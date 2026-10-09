#!/usr/bin/env python
"""Tabela comparativa dos testes em experimentos/runs/.

Uso: python pipeline/experiments/analisar.py

Criterio de escolha: media do val_macro_f1 (epoca escolhida) nas seeds.
val_worst_group e informativo. Metricas de TESTE so sao mostradas com
--com-teste (para o relatorio final; nao usar para escolher).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent


def carregar() -> pd.DataFrame:
    linhas = []
    for run in sorted((EXP_DIR / "runs").glob("*_s*")):
        mp = run / "metrics.json"
        teste, seed = run.name.rsplit("_s", 1)
        if not mp.exists() or not seed.isdigit():  # ex.: D1_dedup_s42_perclass (rerun)
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        hist = m.get("history") or []
        best_ep = m.get("best_epoch") or (len(hist) if hist else None)
        h = next((r for r in hist if r["epoch"] == best_ep), hist[-1] if hist else {})
        ult = hist[-1] if hist else {}
        test = m.get("test") or {}
        linhas.append({
            "teste": teste, "seed": int(seed), "epocas": len(hist), "best_ep": best_ep,
            "val_macro_f1": h.get("val_macro_f1"),
            "val_acc": h.get("val_acc"),
            "val_f1_fake": h.get("val_f1_fake"),
            "val_worst_group": h.get("val_worst_group"),
            "train_macro_f1": h.get("train_macro_f1"),
            "gap_f1": (h["train_macro_f1"] - h["val_macro_f1"])
            if h.get("train_macro_f1") is not None else None,
            "gap_f1_ultima": (ult["train_macro_f1"] - ult["val_macro_f1"])
            if ult.get("train_macro_f1") is not None else None,
            "minutos": round(m["timing"]["total_s"] / 60, 1),
            "test_macro_f1": test.get("macro_f1"),
            "test_acc": test.get("acc"),
            "test_worst_group": test.get("worst_group_macro_f1"),
        })
    return pd.DataFrame(linhas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--com-teste", action="store_true")
    args = ap.parse_args()
    df = carregar()
    if df.empty:
        print("nenhum run concluido em experimentos/runs/")
        return 1
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    esconder = [] if args.com_teste else ["test_macro_f1", "test_acc", "test_worst_group"]

    print("=== POR RUN (metricas da epoca escolhida) ===")
    print(df.drop(columns=esconder).sort_values(["teste", "seed"])
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    agg_cols = ["val_macro_f1", "val_acc", "val_worst_group", "gap_f1"] + \
        ([] if esconder else ["test_macro_f1", "test_acc", "test_worst_group"])
    g = df.groupby("teste")
    res = g[agg_cols].mean()
    res.insert(0, "n_seeds", g.size())
    res.insert(2, "val_macro_f1_dp", g["val_macro_f1"].std())
    res["minutos_medio"] = g["minutos"].mean()
    if "BASE" in res.index:
        base = res.loc["BASE", "val_macro_f1"]
        res.insert(3, "delta_vs_BASE", res["val_macro_f1"] - base)
        # variacao natural: amplitude da BASE entre seeds
        b = df[df.teste == "BASE"]["val_macro_f1"]
        ruido = float(b.max() - b.min()) if len(b) > 1 else np.nan
    else:
        ruido = np.nan
    res = res.sort_values("val_macro_f1", ascending=False)
    print("\n=== POR TESTE (media das seeds; ordenado por val_macro_f1) ===")
    print(res.to_string(float_format=lambda v: f"{v:.4f}"))
    if not np.isnan(ruido):
        print(f"\nvariacao natural da BASE entre seeds (max-min val_macro_f1): {ruido:.4f}")
        print("melhora so conta se delta_vs_BASE > essa variacao (e confirmada em 3 seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
