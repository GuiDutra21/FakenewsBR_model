"""
Score de Confianca: cabeca DFR sobre embeddings congelados + features estilisticas
+ calibracao por temperatura.

Mantem o padrao A/B do baseline linear deste projeto — duas variantes treinadas
lado a lado para MEDIR o atalho, nao so para maximizar acuracia:

  ERM  : treina em toda a base, inclusive os grupos degenerados (`fakes` 100%
         falso, `true` 100% verdadeiro). E o modelo "ingenuo": pode aprender
         "estilo de alegacao de agencia => fake".
  DFR  : treina apenas nos grupos onde a origem NAO prediz o rotulo, com
         reamostragem balanceada por celula (grupo x rotulo).
         Kirichenko et al. (2023), "Last Layer Re-Training is Sufficient for
         Robustness to Spurious Correlations".

Espera-se que o DFR tenha acuracia media MENOR e pior-grupo MAIOR. Essa e a
troca que o projeto ja aceitou explicitamente no walkthrough do baseline.

Uso:
    python -m models.score --embeddings models/artifacts/embeddings.npy
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models import data as D
from models import evaluate as E


def _features(emb: np.ndarray, df: pd.DataFrame, style_cols: list[str],
              scaler_emb=None, scaler_sty=None):
    """Concatena embedding padronizado + features estilisticas padronizadas."""
    if scaler_emb is None:
        scaler_emb = StandardScaler().fit(emb)
    X = scaler_emb.transform(emb)
    if style_cols:
        S = D.style_matrix(df, style_cols)
        if scaler_sty is None:
            scaler_sty = StandardScaler().fit(S)
        X = np.hstack([X, scaler_sty.transform(S)])
    return X.astype(np.float32), scaler_emb, scaler_sty


def _logits2(dec: np.ndarray) -> np.ndarray:
    """decision_function -> matriz de 2 colunas, para o temperature scaling."""
    dec = np.asarray(dec, dtype=np.float64).ravel()
    return np.column_stack([np.zeros_like(dec), dec])


def _fit_head(Xtr, ytr, Xva, yva, dva, Cs, selection: str, sw=None):
    """Ajusta a regressao e escolhe C por pior-grupo (DFR) ou macro-F1 (ERM).

    No modo worst_group a selecao olha SO para os grupos balanceados: sao os
    unicos onde "robustez de grupo" e uma pergunta bem-posta. Incluir grupos
    degenerados faria o criterio premiar quem usa o atalho de proveniencia.
    """
    best, best_score = None, -np.inf
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
        clf.fit(Xtr, ytr, sample_weight=sw)
        p = clf.predict_proba(Xva)[:, 1]
        if selection == "worst_group":
            s = E.worst_group_f1(dva, yva, p, only_groups=D.BALANCED_GROUPS)
            if np.isnan(s):
                s = E.core_metrics(yva, p)["macro_f1"]
        else:
            s = E.core_metrics(yva, p)["macro_f1"]
        if s > best_score:
            best, best_score = clf, s
    return best, best_score


def run_variant(name: str, split: D.Split, emb_by_rid: dict, style_cols: list[str],
                group_balanced: bool, seed: int, Cs,
                calib_on: str = "balanced") -> dict:
    tr, va, te = split.train, split.val, split.test

    sw = None
    if group_balanced:
        tr = tr[tr["is_balanced_group"]].reset_index(drop=True)
        if len(tr) == 0:
            raise ValueError("nenhum grupo balanceado no treino deste split")
        sw = D.group_balanced_weights(tr)

    def emb_of(d):
        return np.stack([emb_by_rid[r] for r in d["rid"].to_numpy()])

    Etr, Eva, Ete = emb_of(tr), emb_of(va), emb_of(te)
    Xtr, se, ss = _features(Etr, tr, style_cols)
    Xva, _, _ = _features(Eva, va, style_cols, se, ss)
    Xte, _, _ = _features(Ete, te, style_cols, se, ss)
    ytr, yva, yte = tr["target"].to_numpy(), va["target"].to_numpy(), te["target"].to_numpy()

    print(f"\n### variante {name} — treino n={len(tr)} "
          f"(fake {ytr.mean()*100:.1f}%), grupos={tr['group'].nunique()}"
          f"{', ponderado por celula' if sw is not None else ''}")

    selection = "worst_group" if group_balanced else "macro_f1"
    clf, sel_score = _fit_head(Xtr, ytr, Xva, yva, va, Cs, selection, sw)
    print(f"  C escolhido={clf.C}  criterio={selection}  val={sel_score:.4f}")

    # Calibracao na validacao — nunca no teste.
    #
    # CUIDADO: calibrar na validacao INTEIRA reintroduz o vies que o DFR removeu.
    # A validacao completa e 71,5% fake, mas esse prior e artefato de compilacao
    # do corpus (58% vem de subsets onde a origem determina o rotulo), nao a
    # prevalencia de desinformacao em nenhum fluxo real. Ajustar o intercepto de
    # Platt para reproduzi-lo empurra o corte de volta para "fake" e desfaz o
    # trabalho da cabeca. Medido: pior-grupo caiu de 0,73 para 0,38.
    #
    # Calibramos sob o prior BALANCEADO. O score resultante e essencialmente uma
    # razao de verossimilhanca; para um fluxo de prevalencia conhecida pi, aplica-se
    # a correcao de prior por fora. Isso e a pratica correta quando a prevalencia
    # de implantacao e desconhecida.
    cmask = va["is_balanced_group"].to_numpy() if calib_on == "balanced" \
        else np.ones(len(va), dtype=bool)
    if cmask.sum() < 50 or len(np.unique(yva[cmask])) < 2:
        cmask = np.ones(len(va), dtype=bool)
        print("  AVISO: sem grupos balanceados na validacao; calibrando na val inteira")

    dva_all = _logits2(clf.decision_function(Xva))
    a, b = E.fit_platt(dva_all[cmask], yva[cmask])
    p_va_raw = clf.predict_proba(Xva)[:, 1]
    p_va_cal = E.apply_platt(dva_all, a, b)
    print(f"  Platt a={a:.4f} b={b:+.4f} | calibrado em n={int(cmask.sum())} "
          f"(prior {yva[cmask].mean()*100:.1f}% fake, modo={calib_on})")
    print(f"  ECE val {E.expected_calibration_error(p_va_raw, yva):.4f} -> "
          f"{E.expected_calibration_error(p_va_cal, yva):.4f}")

    p_te = E.apply_platt(_logits2(clf.decision_function(Xte)), a, b)
    metrics = E.report(te, yte, p_te, f"{split.name} | variante {name} (calibrado)")

    if style_cols:
        w = clf.coef_[0][-len(style_cols):]
        print("\n  pesos das features estilisticas (auditoria):")
        for c, v in sorted(zip(style_cols, w), key=lambda kv: -abs(kv[1])):
            print(f"    {c:24s} {v:+.4f}")

    return {"variant": name, "split": split.name, "C": float(clf.C),
            "platt_a": float(a), "platt_b": float(b), "calib_on": calib_on,
            "calib_prior_fake": float(yva[cmask].mean()),
            **{k: v for k, v in metrics.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="models/artifacts/embeddings.npy")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-length", action="store_true",
                    help="inclui log(word_len) — o atalho mais contaminado. So para medir.")
    ap.add_argument("--no-style", action="store_true",
                    help="desliga features estilisticas (embedding puro)")
    ap.add_argument("--ood", nargs="*", default=["whatsapp", "portal"],
                    help="canais para leave-one-channel-out; vazio desliga")
    ap.add_argument("--calib-on", default="balanced", choices=["balanced", "full"],
                    help="onde calibrar: 'balanced' usa so os grupos sem "
                         "confundimento (recomendado); 'full' usa a validacao "
                         "inteira e reintroduz o prior artefatual de 71,5%% fake")
    ap.add_argument("--out", default="models/artifacts/score_results.json")
    a = ap.parse_args()

    emb = np.load(a.embeddings)
    rid_path = a.embeddings.replace(".npy", "_rid.npy")
    if not os.path.exists(rid_path):
        raise SystemExit(f"falta {rid_path} — rode `python -m models.embed run` de novo")
    rids = np.load(rid_path, allow_pickle=True)
    if len(rids) != len(emb):
        raise SystemExit(f"desalinhado: {len(emb)} embeddings vs {len(rids)} rids")

    df = D.load(a.csv or D.DEFAULT_CSV, include_length=a.include_length)
    emb_by_rid = dict(zip(rids, emb))
    missing = set(df["rid"].to_numpy()) - set(rids)
    if missing:
        raise SystemExit(f"{len(missing)} linhas sem embedding — reextraia com o mesmo CSV")

    style_cols = [] if a.no_style else df.attrs["style_cols"]
    print(f"embeddings {emb.shape} | features estilisticas: {style_cols or 'nenhuma'}")

    Cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    results = []

    split = D.iid_split(df, seed=a.seed)
    print("\n" + split.describe())
    results.append(run_variant("ERM (ingenua)", split, emb_by_rid, style_cols, False, a.seed, Cs, a.calib_on))
    results.append(run_variant("DFR (honesta)", split, emb_by_rid, style_cols, True, a.seed, Cs, a.calib_on))

    for ch in (a.ood or []):
        try:
            osplit = D.ood_split(df, ch, seed=a.seed)
        except ValueError as exc:
            print(f"\npulando OOD '{ch}': {exc}")
            continue
        print("\n" + osplit.describe())
        results.append(run_variant("ERM (ingenua)", osplit, emb_by_rid, style_cols, False, a.seed, Cs, a.calib_on))
        results.append(run_variant("DFR (honesta)", osplit, emb_by_rid, style_cols, True, a.seed, Cs, a.calib_on))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}\nRESUMO\n{'='*70}")
    summ = pd.DataFrame(results)
    cols = [c for c in ["split", "variant", "acc", "macro_f1", "worst_group_macro_f1",
                        "worst_group_balanced", "ece", "temperature",
                        "hard_acc"] if c in summ.columns]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nresultados salvos em {a.out}")


if __name__ == "__main__":
    main()
