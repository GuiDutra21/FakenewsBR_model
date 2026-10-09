"""Diagnostico: a fonte da noticia e inferivel apenas do texto?

Se um classificador TF-IDF+LogReg consegue prever `group` (dataset_name) a
partir de `text_no_url`, entao remover a coluna de fonte do treino nao impede o
encoder de aprender o atalho: o estilo do texto carrega a procedencia.
Grupos com n>=200 no treino full_iid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "models/v6/processed/v6_pool.parquet"
SPLITS = ROOT / "models/v6/processed/v6_splits.parquet"
SPLIT_COL = "split_full_iid"
MIN_N = 200


def main() -> None:
    df = pd.read_parquet(DATA)
    sp = pd.read_parquet(SPLITS)
    df = df.merge(sp[["rid", SPLIT_COL]], on="rid", how="left")
    df = df[df[SPLIT_COL].isin(["train", "val_sel", "val_calib", "test"])]
    df = df[~df["has_ufffd"].astype(bool)].reset_index(drop=True)
    count = df[df[SPLIT_COL] == "train"].groupby("group").size()
    keep = count[count >= MIN_N].index
    df = df[df["group"].isin(keep)].reset_index(drop=True)
    tr = df[df[SPLIT_COL] == "train"]
    te = df[df[SPLIT_COL] == "test"]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          lowercase=True)
    Xtr = vec.fit_transform(tr["text_no_url"].astype(str))
    Xte = vec.transform(te["text_no_url"].astype(str))
    clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, random_state=42)
    clf.fit(Xtr, tr["group"].to_numpy())
    pred = clf.predict(Xte)

    acc = accuracy_score(te["group"], pred)
    macro = f1_score(te["group"], pred, average="macro", zero_division=0)
    maj = te["group"].value_counts(normalize=True).iloc[0]
    fake_frac = (te["label"] == "fake").mean()
    base_fake = max(fake_frac, 1 - fake_frac)

    print(f"[source-leak] grupos={len(keep)} | treino={len(tr)} teste={len(te)}")
    print(f"[source-leak] prever GRUPO pelo texto: acc={acc:.4f} "
          f"macro-F1={macro:.4f} | baseline maioria grupo={maj:.4f}")
    print(f"[source-leak] referencia: prever ROTULO pelo texto (sempre fake)="
          f"{base_fake:.4f}")

    # Atalho rotulo~grupo: prever fake/true com um modelo que so usa a fonte
    # (majority por grupo estimado no treino).
    prior = tr.groupby("group")["label"].apply(
        lambda s: (s == "fake").mean())
    p_src = te["group"].map(prior).fillna(0.5).to_numpy()
    acc_src = accuracy_score(te["label"], np.where(p_src >= 0.5, "fake", "true"))
    from models import evaluate as E
    y_te = (te["label"] == "fake").to_numpy().astype(int)
    m_src = E.core_metrics(y_te, p_src)
    wg_src = E.worst_group_f1(te.reset_index(drop=True), y_te, p_src)
    print(f"[source-leak] rotulo previsto SO pela fonte (prior por grupo): "
          f"acc={acc_src:.4f} vs sempre-fake={max(fake_frac, 1-fake_frac):.4f}")
    print(f"[source-leak] procedencia-so: acc={m_src['acc']:.4f} "
          f"macro-F1={m_src['macro_f1']:.4f} F1-fake={m_src['f1_fake']:.4f} "
          f"ECE={m_src['ece']:.4f} pior-grupo={wg_src:.4f}")

    tab = (pd.DataFrame({"group": te["group"], "pred": pred})
           .value_counts().reset_index(name="n").head(15))
    print("\n[source-leak] maiores confusionamentos (grupo real -> previsto):")
    for _, r in tab.iterrows():
        print(f"  {r['group']:<24} -> {r['pred']:<24} n={r['n']}")


if __name__ == "__main__":
    main()
