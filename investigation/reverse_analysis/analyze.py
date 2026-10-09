"""Analise reversa por fatia dos tres leitores do BERTimbau fine-tuned.

Os tres "modelos" que aparecem nos relatorios usam o MESMO encoder ajustado:

  FT-classifier : a cabeca `classifier` treinada junto com o encoder, com Platt
                  na validacao inteira (`5_finetune.log`, acc 0,8655)
  ERM           : regressao logistica sobre embeddings mean-pooled de toda a base
  DFR           : regressao logistica so nos grupos balanceados, ponderada por
                  celula, calibrada sob prior balanceado

Perguntas que este script responde com dado (e nao com leitura de log):

  1. O gap PT-PT e dialeto ou e o subset `true`? Compara PT-PT x PT-BR DENTRO
     do mesmo subset e ROC-AUC dentro do mesmo checador (Poligrafo x Lupa).
  2. Quanto do desempenho vem de quase-duplicatas entre treino e teste?
  3. Onde o erro se concentra: comprimento, ano, rating, ruido de rotulo.
  4. A calibracao do FT-classifier cai na armadilha do prior (Platt na val cheia)?

Uso:
    PYTHONPATH=legacy python -m investigation.reverse_analysis.predict_ft   # uma vez, ~12 min
    PYTHONPATH=legacy python -m investigation.reverse_analysis.analyze
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from models import data as D
from models import evaluate as E
from models import score as S

ART = Path("legacy/models/artifacts")
OUT = Path("investigation/reverse_analysis/output")
CS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
NEAR_DUP_COS = 0.90


# ----------------------------------------------------------------- utilidades

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(mid - half, 4), round(mid + half, 4))


def slice_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    n = len(y)
    k = int((pred == y).sum())
    out = {"n": n, "fake_%": round(float(y.mean() * 100), 1) if n else float("nan"),
           "acc": round(k / n, 4) if n else float("nan"), "acc_ci95": wilson(k, n)}
    if n and len(np.unique(y)) == 2:
        out["macro_f1"] = round(float(f1_score(y, pred, average="macro")), 4)
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
    return out


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def domain(u) -> str:
    m = re.match(r"https?://(?:www\.)?([^/]+)", str(u) if isinstance(u, str) else "")
    return m.group(1) if m else "sem_url"


# --------------------------------------------------------------- predicoes

def head_predictions(split: D.Split, emb_by_rid: dict, group_balanced: bool) -> np.ndarray:
    """Reproduz run_variant de score.py e devolve p(fake) calibrado no teste."""
    tr, va, te = split.train, split.val, split.test
    sw = None
    if group_balanced:
        tr = tr[tr["is_balanced_group"]].reset_index(drop=True)
        sw = D.group_balanced_weights(tr)

    def emb(d):
        return np.stack([emb_by_rid[r] for r in d["rid"].to_numpy()])

    style = D.STYLE_RATE
    Xtr, se, ss = S._features(emb(tr), tr, style)
    Xva, _, _ = S._features(emb(va), va, style, se, ss)
    Xte, _, _ = S._features(emb(te), te, style, se, ss)
    yva = va["target"].to_numpy()
    clf, _ = S._fit_head(Xtr, tr["target"].to_numpy(), Xva, yva, va, CS,
                         "worst_group" if group_balanced else "macro_f1", sw)
    cm = va["is_balanced_group"].to_numpy()
    a, b = E.fit_platt(S._logits2(clf.decision_function(Xva))[cm], yva[cm])
    return E.apply_platt(S._logits2(clf.decision_function(Xte)), a, b)


def near_duplicate_similarity(train_texts, query_texts) -> np.ndarray:
    """Cosseno do vizinho mais proximo no treino (TF-IDF de n-gramas de caractere)."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=2,
                          max_features=400_000, sublinear_tf=True)
    Xtr = vec.fit_transform([norm_text(t) for t in train_texts])
    Xq = vec.transform([norm_text(t) for t in query_texts])
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(Xtr)
    dist, _ = nn.kneighbors(Xq)
    return 1.0 - dist[:, 0]


# ---------------------------------------------------------------- analises

def by(df: pd.DataFrame, key, models: dict, min_n: int = 30) -> list[dict]:
    # `models` esta alinhado ao teste inteiro (RangeIndex); `df` pode ser um filtro
    # dele. `indices` devolve posicoes DENTRO de `df`, entao traduzimos para o
    # rotulo de indice antes de indexar as predicoes.
    rows = []
    for val, idx in df.groupby(key, observed=True).indices.items():
        if len(idx) < min_n:
            continue
        y = df["target"].to_numpy()[idx]
        pos = df.index.to_numpy()[idx]
        row = {"fatia": val if not isinstance(val, tuple) else " | ".join(map(str, val))}
        for name, p in models.items():
            m = slice_metrics(y, p[pos])
            row[name] = m
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = D.load(D.DEFAULT_CSV)
    split = D.iid_split(df, seed=42)
    te = split.test.copy()

    preds = pd.read_csv(ART / "preds_ft_classifier.csv")
    p_ft = te[["rid"]].merge(preds[preds["split"] == "test"], on="rid", how="left")["p_fake"].to_numpy()
    if np.isnan(p_ft).any():
        raise SystemExit("predicoes do FT-classifier nao cobrem o teste; rode predict_ft")

    emb = np.load(ART / "embeddings_ft.npy")
    rids = np.load(ART / "embeddings_ft_rid.npy", allow_pickle=True)
    emb_by_rid = dict(zip(rids, emb))
    print("treinando cabecas ERM e DFR sobre embeddings_ft (reproducao de score.py)...", flush=True)
    p_erm = head_predictions(split, emb_by_rid, group_balanced=False)
    p_dfr = head_predictions(split, emb_by_rid, group_balanced=True)
    models = {"FT": p_ft, "ERM": p_erm, "DFR": p_dfr}
    y = te["target"].to_numpy()

    report: dict = {"reproducao": {k: slice_metrics(y, p) for k, p in models.items()},
                    "esperado_dos_logs": {"FT": 0.8655, "ERM": 0.8451, "DFR": 0.7345}}
    print(json.dumps(report["reproducao"], indent=1), flush=True)

    te["publisher"] = te["url_review"].map(domain)
    te["dialeto"] = np.where(te["is_ptpt"], "PT-PT", "PT-BR")

    # 1. dialeto dentro do subset e dentro do checador
    report["dialeto_por_subset"] = by(te[te["group"].isin(["fakes", "true"])],
                                      ["group", "dialeto"], models)
    report["auc_dentro_do_checador"] = by(
        te[te["publisher"].isin(["poligrafo.sapo.pt", "lupa.uol.com.br", "observador.pt",
                                 "checamos.afp.com", "politica.estadao.com.br"])],
        "publisher", models)
    ptpt_groups = te.loc[te["is_ptpt"], "group"].value_counts().to_dict()
    report["composicao_ptpt_teste"] = {str(k): int(v) for k, v in ptpt_groups.items()}

    # 2. vazamento por quase-duplicata treino -> teste
    print("calculando similaridade treino->teste...", flush=True)
    sim = near_duplicate_similarity(split.train[D.TEXT_COL].astype(str).tolist(),
                                    te[D.TEXT_COL].astype(str).tolist())
    te["sim_treino"] = sim
    te["quase_dup"] = np.where(sim >= NEAR_DUP_COS, "quase-dup no treino", "sem quase-dup")
    report["quase_dup_limiar"] = NEAR_DUP_COS
    report["quase_dup_por_grupo"] = {
        g: {"n": int(len(d)), "pct_quase_dup": round(float((d["sim_treino"] >= NEAR_DUP_COS).mean() * 100), 1)}
        for g, d in te.groupby("group")}
    report["quase_dup_desempenho"] = by(te, ["group", "quase_dup"], models)

    # 3. comprimento, ano, rating, ruido de rotulo
    te["faixa_palavras"] = pd.cut(te["word_len"], [0, 10, 20, 40, 80, 160, 10_000],
                                  labels=["<=10", "11-20", "21-40", "41-80", "81-160", ">160"])
    report["comprimento_por_grupo"] = by(te, ["group", "faixa_palavras"], models)
    te["ano"] = pd.to_datetime(te["date_iso"], errors="coerce").dt.year
    te["ano_faixa"] = pd.cut(te["ano"], [0, 2017, 2018, 2019, 2020, 2021, 2030],
                             labels=["<=2017", "2018", "2019", "2020", "2021", "2022+"])
    te["ano_faixa"] = te["ano_faixa"].cat.add_categories("sem data").fillna("sem data")
    report["ano"] = by(te, "ano_faixa", models)
    report["ano_por_grupo"] = by(te, ["group", "ano_faixa"], models)
    report["rating_class"] = by(te, "rating_class", models, min_n=20)
    contradiz = ((te["rating_class"] == "false_pure") & (te["label"] == "true")) | \
                ((te["rating_class"] == "true_rating") & (te["label"] == "fake"))
    full = df.copy()
    full_contradiz = ((full["rating_class"] == "false_pure") & (full["label"] == "true")) | \
                     ((full["rating_class"] == "true_rating") & (full["label"] == "fake"))
    report["ruido_rotulo_rating_contradiz"] = {
        "base_inteira": int(full_contradiz.sum()),
        "por_grupo": {str(k): int(v) for k, v in full[full_contradiz]["group"].value_counts().items()},
        "no_teste": int(contradiz.sum()),
    }
    te["termina_com_interrogacao"] = te[D.TEXT_COL].astype(str).str.strip().str.endswith("?")
    report["interrogacao_por_subset"] = by(te[te["group"].isin(["fakes", "true"])],
                                           ["group", "publisher", "termina_com_interrogacao"], models)

    # 4. calibracao do FT-classifier: Platt na val cheia x val balanceada
    va = split.val
    pv = preds[preds["split"] == "val"].set_index("rid").loc[va["rid"].to_numpy()]
    lv = pv[["logit_true", "logit_fake"]].to_numpy()
    lt = preds[preds["split"] == "test"].set_index("rid").loc[te["rid"].to_numpy()][["logit_true", "logit_fake"]].to_numpy()
    cm = va["is_balanced_group"].to_numpy()
    a_bal, b_bal = E.fit_platt(lv[cm], va["target"].to_numpy()[cm])
    p_ft_bal = E.apply_platt(lt, a_bal, b_bal)
    bal = te["is_balanced_group"].to_numpy()
    report["calibracao_ft"] = {
        "platt_val_cheia": {
            "pior_grupo": E.worst_group_f1(te, y, p_ft),
            "ece_balanceado": E.expected_calibration_error(p_ft[bal], y[bal]),
            "acc_true_subset": slice_metrics(y[te["group"].eq("true").to_numpy()],
                                             p_ft[te["group"].eq("true").to_numpy()])["acc"]},
        "platt_val_balanceada": {
            "a": a_bal, "b": b_bal,
            "pior_grupo": E.worst_group_f1(te, y, p_ft_bal),
            "ece_balanceado": E.expected_calibration_error(p_ft_bal[bal], y[bal]),
            "acc_true_subset": slice_metrics(y[te["group"].eq("true").to_numpy()],
                                             p_ft_bal[te["group"].eq("true").to_numpy()])["acc"],
            "acc_global": slice_metrics(y, p_ft_bal)["acc"]},
    }

    # amostra de erros confiantes para leitura qualitativa
    err = te.assign(p_ft=p_ft, p_dfr=p_dfr, conf_ft=np.abs(p_ft - 0.5))
    err = err[(err["p_ft"] >= 0.5).astype(int) != err["target"]]
    err.sort_values("conf_ft", ascending=False).groupby("group").head(40)[
        ["rid", "group", "publisher", "dialeto", "label", "factcheck_rating", "p_ft", "p_dfr",
         "sim_treino", "word_len", D.TEXT_COL]].to_csv(OUT / "erros_confiantes.csv", index=False)

    (OUT / "reverse_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"salvo em {OUT}/reverse_analysis.json e erros_confiantes.csv")


if __name__ == "__main__":
    main()
