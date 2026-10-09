"""
Protocolo de avaliacao honesto + calibracao do Score de Confianca.

A metrica de manchete deste projeto NAO e acuracia media. E:
  1. macro-F1 do PIOR GRUPO (a media esconde o confundimento de proveniencia)
  2. ECE (o score precisa ser calibrado para ser um score, nao um rotulo)
  3. desempenho nos casos limitrofes (meias-verdades)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)


# ---------------------------------------------------------------- calibracao

def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): 1 parametro, nao altera o ranking
    nem a acuracia, so corrige a confianca. Busca em grade + refino ternario."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y).astype(int)

    def nll(T: float) -> float:
        z = logits / max(T, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    # Grade log-espacada e larga: uma cabeca mal calibrada pode precisar de T >> 1,
    # e uma grade curta satura no limite e devolve um numero sem sentido.
    grid = np.geomspace(0.05, 50.0, 120)
    lo_i = int(np.argmin([nll(t) for t in grid]))
    lo = grid[max(lo_i - 1, 0)]
    hi = grid[min(lo_i + 1, len(grid) - 1)]
    for _ in range(60):  # busca ternaria
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Retorna P(classe 1) apos escalonamento."""
    z = np.asarray(logits, dtype=np.float64) / max(T, 1e-3)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling: p = sigmoid(a*z + b), ajustado por maxima verossimilhanca.

    Preferido ao temperature scaling NESTE projeto porque a cabeca DFR e treinada
    com os grupos equilibrados (~47% fake) e implantada sobre uma base 71,5% fake.
    Isso e deslocamento de prior, e temperatura (1 parametro) so consegue achatar
    a confianca — nao deslocar o ponto de corte. O intercepto `b` faz isso.

    Ajustado como uma regressao logistica de uma variavel sobre o valor de decisao,
    que e exatamente a definicao de Platt (1999).
    """
    from sklearn.linear_model import LogisticRegression

    z = _decision(logits).reshape(-1, 1)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    lr.fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    z = _decision(logits)
    return 1.0 / (1.0 + np.exp(-np.clip(a * z + b, -700, 700)))


def _decision(logits: np.ndarray) -> np.ndarray:
    """Reduz logits de 2 colunas ao valor de decisao escalar z = l1 - l0."""
    lg = np.asarray(logits, dtype=np.float64)
    return lg[:, 1] - lg[:, 0] if lg.ndim == 2 else lg.ravel()


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """ECE com bins de largura igual sobre a confianca da classe predita."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        rows.append({"faixa": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(m.sum()),
                     "score_medio": round(float(p[m].mean()), 4),
                     "fake_real": round(float(y[m].mean()), 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ metricas

def core_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")) if len(np.unique(y)) > 1 else float("nan"),
        "f1_fake": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ece": expected_calibration_error(p, y),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:  # grupo degenerado: so acuracia faz sentido
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


# Um grupo so entra na metrica de manchete se tiver massa suficiente para que
# o numero signifique alguma coisa. MuMiN-PT (n=34, com 3 exemplos da classe
# minoritaria) produzia macro-F1 puro ruido e dominava o "pior grupo".
MIN_GROUP_N = 100
MIN_MINORITY_N = 20


def per_group(df: pd.DataFrame, y: np.ndarray, p: np.ndarray,
              col: str = "group", min_n: int = 30) -> pd.DataFrame:
    rows = []
    for name, idx in df.groupby(col).indices.items():
        if len(idx) < min_n:
            continue
        yi = y[idx]
        m = core_metrics(yi, p[idx])
        m[col] = name
        m["fake_%"] = round(float(yi.mean() * 100), 1)
        m["minoria_n"] = int(min(np.bincount(yi, minlength=2)))
        m["confiavel"] = bool(len(idx) >= MIN_GROUP_N and m["minoria_n"] >= MIN_MINORITY_N)
        rows.append(m)
    if not rows:
        return pd.DataFrame()
    cols = [col, "n", "minoria_n", "fake_%", "acc", "macro_f1", "f1_fake",
            "roc_auc", "pr_auc", "ece", "confiavel"]
    return pd.DataFrame(rows)[cols].sort_values("macro_f1", na_position="last")


def worst_group_f1(df: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                   col: str = "group", min_n: int = 30,
                   only_groups: list | None = None) -> float:
    """Metrica de manchete: pior macro-F1 entre grupos ESTATISTICAMENTE confiaveis.

    Ignora grupos degenerados (uma classe so) e grupos pequenos demais, que
    produziriam um minimo dominado por ruido amostral.
    """
    d, yy, pp = df, y, p
    if only_groups is not None:
        m = df[col].isin(only_groups).to_numpy()
        if m.sum() == 0:
            return float("nan")
        d, yy, pp = df[m].reset_index(drop=True), y[m], p[m]
    tab = per_group(d, yy, pp, col=col, min_n=min_n)
    if tab.empty:
        return float("nan")
    tab = tab[tab["confiavel"] & tab["macro_f1"].notna()]
    if tab.empty:
        return float("nan")
    return float(tab["macro_f1"].min())


def report(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, title: str,
           threshold: float = 0.5) -> dict:
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    glob = core_metrics(y, p, threshold)
    print(f"GLOBAL  n={glob['n']}  acc={glob['acc']:.4f}  macro-F1={glob['macro_f1']:.4f}  "
          f"F1(fake)={glob['f1_fake']:.4f}")
    print(f"        PR-AUC={glob['pr_auc']:.4f}  ROC-AUC={glob['roc_auc']:.4f}  "
          f"Brier={glob['brier']:.4f}  ECE={glob['ece']:.4f}")

    tab = per_group(df, y, p)
    if not tab.empty:
        print("\nPOR GRUPO (ordenado do pior para o melhor):")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        frac = tab[~tab["confiavel"]]
        if not frac.empty:
            print(f"  (grupos com confiavel=False sao pequenos demais — "
                  f"n<{MIN_GROUP_N} ou minoria<{MIN_MINORITY_N} — e ficam fora da manchete)")
        wg = worst_group_f1(df, y, p)
        print(f"\n>>> PIOR GRUPO macro-F1 = {wg:.4f}   (metrica de manchete, "
              f"so grupos confiaveis)")
        glob["worst_group_macro_f1"] = wg

        # Restrito aos grupos sem confundimento origem->rotulo: e onde
        # "robustez de grupo" e uma pergunta bem-posta.
        if "is_balanced_group" in df.columns and bool(df["is_balanced_group"].any()):
            m = df["is_balanced_group"].to_numpy()
            wgb = worst_group_f1(df[m].reset_index(drop=True), y[m], p[m])
            print(f">>> PIOR GRUPO (so grupos balanceados) = {wgb:.4f}")
            glob["worst_group_balanced"] = wgb

            # O score e calibrado sob prior balanceado, entao o ECE medido sobre
            # a base inteira (71,5% fake, prior artefatual) penaliza uma escolha
            # deliberada. Este e o ECE no regime em que o score foi definido.
            bm = core_metrics(y[m], p[m])
            print(f"    nos grupos balanceados (n={bm['n']}, "
                  f"{y[m].mean()*100:.1f}% fake): acc={bm['acc']:.4f} "
                  f"macro-F1={bm['macro_f1']:.4f} ECE={bm['ece']:.4f}")
            glob["acc_balanced"] = bm["acc"]
            glob["macro_f1_balanced"] = bm["macro_f1"]
            glob["ece_balanced"] = bm["ece"]

    # Casos limitrofes: onde modelos que decoraram "assinatura gritante" falham.
    if "rating_class" in df.columns:
        hard = np.flatnonzero((df["rating_class"] == "hard").to_numpy())
        pure = np.flatnonzero((df["rating_class"] == "false_pure").to_numpy())
        if len(hard) >= 20:
            hm = core_metrics(y[hard], p[hard])
            print(f"\nCASOS LIMITROFES  n={hm['n']}  acc={hm['acc']:.4f}  "
                  f"score medio={p[hard].mean():.3f}")
            glob["hard_acc"] = hm["acc"]
        if len(pure) >= 20:
            pm = core_metrics(y[pure], p[pure])
            print(f"FALSO PURO        n={pm['n']}  acc={pm['acc']:.4f}  "
                  f"score medio={p[pure].mean():.3f}")
            glob["pure_acc"] = pm["acc"]

    if "is_ptpt" in df.columns:
        pt = np.flatnonzero(df["is_ptpt"].to_numpy())
        br = np.flatnonzero(~df["is_ptpt"].to_numpy())
        if len(pt) >= 30 and len(br) >= 30:
            a, b = core_metrics(y[pt], p[pt]), core_metrics(y[br], p[br])
            print(f"\nDIALETO  PT-PT: n={a['n']} macro-F1={a['macro_f1']:.4f} | "
                  f"resto: n={b['n']} macro-F1={b['macro_f1']:.4f}")
            glob["ptpt_macro_f1"] = a["macro_f1"]

    return glob
