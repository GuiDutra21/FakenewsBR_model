"""Baseline TF-IDF + Regressao Logistica no pool v6.

Mesmo split (`full_iid`) e mesmo protocolo de avaliacao do fine-tuning
(`models/evaluate.py`): teste de 13.661 linhas, Platt em val_calib, pior-grupo,
ECE. Serve como referencia do que um modelo linear simples alcanca na tarefa.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models import evaluate as E  # noqa: E402

DATA = ROOT / "models/v6/processed/v6_pool.parquet"
SPLITS = ROOT / "models/v6/processed/v6_splits.parquet"
OUT = ROOT / "models/v6/compare/baseline_tfidf"
SPLIT_COL = "split_full_iid"


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def load_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    sp = pd.read_parquet(SPLITS)
    df = df.merge(sp[["rid", SPLIT_COL]], on="rid", how="left")
    df = df[df[SPLIT_COL].isin(["train", "val_sel", "val_calib", "test"])]
    df = df[~df["has_ufffd"].astype(bool)].reset_index(drop=True)
    df["is_ptpt"] = df["is_ptpt_rule"].astype(bool)
    df["y"] = (df["label"] == "fake").astype(int)
    return df


def threshold_val_opt(z_va: np.ndarray, y_va: np.ndarray, a: float, b: float) -> float:
    p = E.apply_platt(z_va, a, b)
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_va, (p >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_df()
    tr = df[df[SPLIT_COL] == "train"].reset_index(drop=True)
    va = df[df[SPLIT_COL] == "val_calib"].reset_index(drop=True)
    te = df[df[SPLIT_COL] == "test"].reset_index(drop=True)
    print(f"[base] treino={len(tr)} val_calib={len(va)} teste={len(te)} "
          f"fake treino={tr.y.mean():.4f}")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          lowercase=True)
    Xtr = vec.fit_transform(tr["text_no_url"].astype(str))
    Xva = vec.transform(va["text_no_url"].astype(str))
    Xte = vec.transform(te["text_no_url"].astype(str))
    print(f"[tfidf] features={Xtr.shape[1]} nnz_treino={Xtr.nnz} "
          f"({time.time()-t0:.1f}s)")

    metrics: dict = {"data": {"n_train": len(tr), "n_val_calib": len(va),
                              "n_test": len(te), "fake_train_pct": float(tr.y.mean() * 100)},
                     "majority": {}, "models": {}}

    maj_pred = np.ones(len(te), dtype=int)
    metrics["majority"] = E.core_metrics(te.y.to_numpy(), maj_pred.astype(float))
    print(f"[majority] acc={metrics['majority']['acc']:.4f} "
          f"macro-F1={metrics['majority']['macro_f1']:.4f}")

    for tag, cw in (("erm", None), ("balanced", "balanced")):
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000,
                                 class_weight=cw, random_state=42)
        t1 = time.time()
        clf.fit(Xtr, tr.y.to_numpy())
        zva = clf.decision_function(Xva)
        zte = clf.decision_function(Xte)
        p_raw = sigmoid(zte)

        mask = va["is_balanced_group"].astype(bool).to_numpy()
        if mask.sum() >= 50 and len(np.unique(va.y.to_numpy()[mask])) > 1:
            zf, yf = zva[mask], va.y.to_numpy()[mask]
        else:
            zf, yf = zva, va.y.to_numpy()
        a, b = E.fit_platt(zf, yf)
        p_cal = E.apply_platt(zte, a, b)
        thr = threshold_val_opt(zva, va.y.to_numpy(), a, b)
        print(f"\n########## modelo={tag} (fit {time.time()-t1:.1f}s) "
              f"platt a={a:.4f} b={b:+.4f} thr_val_opt={thr:.2f} ##########")

        glob_raw = E.report(te, te.y.to_numpy(), p_raw,
                            f"BASELINE TF-IDF+LR ({tag}) | teste full_iid | cru")
        glob_cal = E.report(te, te.y.to_numpy(), p_cal,
                            f"BASELINE TF-IDF+LR ({tag}) | teste full_iid | calibrado 0.5")
        glob_thr = E.core_metrics(te.y.to_numpy(), p_cal, threshold=thr)

        metrics["models"][tag] = {
            "platt_a": a, "platt_b": b, "threshold_val_opt": thr,
            "test_raw": glob_raw, "test_calibrated_050": glob_cal,
            "test_calibrated_val_threshold": glob_thr,
            "per_channel": E.per_group(te, te.y.to_numpy(), p_cal, col="channel").to_dict("records"),
            "per_group": E.per_group(te, te.y.to_numpy(), p_cal, col="group").to_dict("records"),
            "calibration_time_s": round(time.time() - t1, 1),
            "class_weight": cw,
        }
        if tag == "erm":
            pred = pd.DataFrame({
                "rid": te["rid"], "y_true": te["y"], "z": zte,
                "p_raw": p_raw, "p_cal": p_cal,
                "pred_05": (p_cal >= 0.5).astype(int),
                "pred_val_thr": (p_cal >= thr).astype(int),
                "group": te["group"], "channel": te["channel"],
                "publisher": te["publisher"], "era": te["era"],
                "label_tier": te["label_tier"], "is_ptpt": te["is_ptpt"],
                "rating_class": te["rating_class"],
            })
            pred.to_csv(OUT / "predictions.csv", index=False)

    metrics["timing_s"] = round(time.time() - t0, 1)
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[base] artefatos em {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
