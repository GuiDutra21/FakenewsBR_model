#!/usr/bin/env python
"""compare_v1_vs_v6.py -- comparacao BASE v1 x BASE v6 nos testes do pool v6.

Avalia o BERTimbau BASE fine-tuned na v1 (`models/artifacts/bertimbau_finetuned`)
nos testes `full_iid` e `ood_wa` do pool v6, usando a calibracao Platt salva no
artefato antigo e apenas `models.evaluate` para as metricas. Mede a contaminacao
do teste v6 pelo TREINO v1 (reconstruido com `models.data.load` +
`iid_split(seed=42)`) e reporta cada teste em dois cortes: completo e limpo
(sem rids que o antigo viu no treino).

Saidas (default):
  models/v6/compare/old_v1_on_v6/predictions.csv
  models/v6/compare/old_v1_on_v6/metrics.json
  models/v6/compare/old_v1_on_v6/ood_wa/{predictions.csv,metrics.json}
  models/v6/compare/COMPARACAO_V1_VS_V6.md

Nao escreve em nenhum outro lugar; nao toca nos parquets nem no pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import data as D
from models import evaluate as EV
from models.encoder import TextDS, infer, mask_entities

TEXT_COL = D.TEXT_COL
CUTS = ("channel", "group", "label_tier")
DEFAULT_OUT = ROOT / "models" / "v6" / "compare" / "old_v1_on_v6"
DEFAULT_MD = ROOT / "models" / "v6" / "compare" / "COMPARACAO_V1_VS_V6.md"
DEFAULT_MODEL = ROOT / "models" / "artifacts" / "bertimbau_finetuned"

REGISTERED_V1_IID = {
    "source": "models/artifacts/logs/5_finetune.log (TESTE | BERTimbau fine-tuned, calibrado)",
    "n": 5920,
    "acc": 0.8655,
    "macro_f1": 0.8214,
    "f1_fake": 0.9102,
    "pr_auc": 0.9615,
    "roc_auc": 0.9162,
    "brier": 0.0962,
    "ece": 0.0105,
    "worst_group_macro_f1": 0.7276,
    "worst_group_balanced": 0.7276,
    "hard_acc": 0.9653,
    "ptpt_macro_f1": 0.6248,
}

PRED_COLS = ["rid", "y_true", "logit0", "logit1", "z", "p_cal", "group",
             "channel", "label_tier", "is_balanced_group", "is_ptpt_rule",
             "era", "publisher", "contam_v1_train"]


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if o is pd.NA or (isinstance(o, float) and math.isnan(o)):
        return None
    return o


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def load_calibration(path: Path) -> tuple[dict, float, float, int, bool]:
    cal = json.loads(path.read_text(encoding="utf-8"))
    a = float(cal["platt_a"])
    b = float(cal["platt_b"])
    max_length = int(cal.get("max_length") or 192)
    mask = bool(cal.get("mask_entities", False))
    return cal, a, b, max_length, mask


def load_v6(pool_path: Path, splits_path: Path,
            split_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    pool = pd.read_parquet(pool_path)
    splits = pd.read_parquet(splits_path)
    pool["rid"] = pool["rid"].astype("int64")
    splits["rid"] = splits["rid"].astype("int64")
    if "target" not in pool.columns:
        pool["target"] = (pool["label"].astype(str) == "fake").astype(int)
    resolved = {}
    for c in split_cols:
        if c in splits.columns:
            resolved[c] = c
        elif f"split_{c}" in splits.columns:
            resolved[c] = f"split_{c}"
        else:
            raise ValueError(
                f"coluna de split '{c}' ausente em {splits_path}: "
                f"{list(splits.columns)}")
    keep = ["rid"] + list(resolved.values())
    pool = pool.merge(splits[keep], on="rid", how="left", validate="one_to_one")
    pool = pool.rename(columns={v: k for k, v in resolved.items()})

    removidas = {}
    n_bruto = int(len(pool))
    if "has_ufffd" in pool.columns:
        u = pool["has_ufffd"].to_numpy().astype(bool)
        if u.any():
            for col in split_cols:
                for val, n in pool.loc[u, col].value_counts().items():
                    removidas[f"{col}={val}"] = int(n)
            print(f"[E2] removendo {int(u.sum())} linhas has_ufffd: {removidas}")
            pool = pool[~u].reset_index(drop=True)
    return pool, {"n_pool_bruto": n_bruto, "n_pool": int(len(pool)),
                  "has_ufffd_removidas": removidas}


def build_v1_split(v1_csv: str, seed: int) -> tuple[pd.DataFrame, D.Split, dict]:
    df = D.load(v1_csv)
    split = D.iid_split(df, seed=seed)
    info = {
        "csv": v1_csv,
        "n": int(len(df)),
        "fake_pct": round(float(df["target"].mean() * 100), 2),
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "fake_pct_train": round(float(split.train["target"].mean() * 100), 2),
        "fake_pct_val": round(float(split.val["target"].mean() * 100), 2),
        "fake_pct_test": round(float(split.test["target"].mean() * 100), 2),
    }
    return df, split, info


def contamination_mask(test_df: pd.DataFrame, v1_split: D.Split) -> pd.DataFrame:
    rid = test_df["rid"]
    return pd.DataFrame({
        "contam_v1_train": rid.isin(set(v1_split.train["rid"])),
        "in_v1_val": rid.isin(set(v1_split.val["rid"])),
        "in_v1_test": rid.isin(set(v1_split.test["rid"])),
    })


def infer_dataset(df: pd.DataFrame, tok, model, max_length: int, batch_size: int,
                  use_mask: bool, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    texts = df[TEXT_COL].astype(str).tolist()
    if use_mask:
        texts = [mask_entities(t) for t in texts]
    y = df["target"].to_numpy().astype(int)
    ds = TextDS(texts, y, tok, max_length)
    collate = DataCollatorWithPadding(tok)
    t0 = time.perf_counter()
    logits = infer(model, ds, collate, batch_size, device)
    dt = time.perf_counter() - t0
    print(f"    inferencia: n={len(ds)} batch={batch_size} "
          f"tempo={dt/60:.1f} min ({len(ds)/max(dt,1e-9):.1f} amostras/s)")
    return logits, y


def evaluate_set(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, title: str,
                 threshold: float) -> tuple[dict, dict]:
    dr = df.copy()
    if "is_ptpt_rule" in dr.columns and "is_ptpt" not in dr.columns:
        dr = dr.rename(columns={"is_ptpt_rule": "is_ptpt"})
    glob = EV.report(dr, y, p, title, threshold=threshold)
    cuts = {}
    for col in CUTS:
        tab = EV.per_group(dr, y, p, col=col)
        cuts[col] = tab.to_dict(orient="records") if len(tab) else []
    return glob, cuts


def build_predictions(df: pd.DataFrame, y: np.ndarray, logits: np.ndarray,
                      p: np.ndarray, contam: pd.DataFrame) -> pd.DataFrame:
    z = (logits[:, 1] - logits[:, 0]).astype(np.float64)
    return pd.DataFrame({
        "rid": df["rid"].to_numpy(),
        "y_true": y,
        "logit0": logits[:, 0].astype(np.float64),
        "logit1": logits[:, 1].astype(np.float64),
        "z": z,
        "p_cal": p,
        "group": df["group"].to_numpy(),
        "channel": df["channel"].to_numpy(),
        "label_tier": df["label_tier"].to_numpy(),
        "is_balanced_group": df["is_balanced_group"].to_numpy(),
        "is_ptpt_rule": df["is_ptpt_rule"].to_numpy(),
        "era": df["era"].to_numpy(),
        "publisher": df["publisher"].to_numpy(),
        "contam_v1_train": contam["contam_v1_train"].to_numpy(),
    })[PRED_COLS]


def run_eval_set(name: str, df_test: pd.DataFrame, tok, model, a: float, b: float,
                 args, device: torch.device, out_dir: Path) -> dict:
    print(f"\n[{name}] teste n={len(df_test)} "
          f"(fake={df_test['target'].mean()*100:.1f}%)")
    logits, y = infer_dataset(df_test, tok, model, args.max_length,
                              args.batch_size, args.use_mask, device)
    p = EV.apply_platt(logits, a, b)
    contam = contamination_mask(df_test, args.v1_split)
    pred = build_predictions(df_test, y, logits, p, contam)
    pred.to_csv(out_dir / "predictions.csv", index=False, float_format="%.6f")
    print(f"    predictions.csv: {out_dir / 'predictions.csv'} (n={len(pred)})")

    m_full, cuts_full = evaluate_set(
        df_test, y, p, f"{name} | TESTE {name} COMPLETO (v1 BASE)", args.threshold)
    clean = ~contam["contam_v1_train"].to_numpy()
    m_clean, cuts_clean = evaluate_set(
        df_test[clean].reset_index(drop=True), y[clean], p[clean],
        f"{name} | TESTE {name} LIMPO (sem treino v1)", args.threshold)

    n = int(len(df_test))
    n_contam = int(contam["contam_v1_train"].sum())
    set_metrics = {
        "n_test": n,
        "n_contam_v1_train": n_contam,
        "pct_contam_v1_train": round(100.0 * n_contam / max(n, 1), 4),
        "n_clean": int(clean.sum()),
        "n_overlap_v1_val": int(contam["in_v1_val"].sum()),
        "n_overlap_v1_test": int(contam["in_v1_test"].sum()),
        "fake_pct": round(float(df_test["target"].mean() * 100), 2),
        "full": {**m_full, "by_cut": cuts_full},
        "clean": {**m_clean, "by_cut": cuts_clean},
    }
    save_json(set_metrics, out_dir / "metrics.json")
    print(f"[{name}] metrics.json: {out_dir / 'metrics.json'}")
    return set_metrics


def fmt_br(v, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and not math.isfinite(v):
        return "-"
    return f"{v:.{nd}f}".replace(".", ",")


def num_br(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and not math.isfinite(v):
        return "-"
    return f"{int(v):,}".replace(",", ".")


def pct_br(v, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and not math.isfinite(v):
        return "-"
    return f"{v:.{nd}f}".replace(".", ",")


def md_metric_block(set_m: dict) -> dict:
    mf = set_m["full"]
    mc = set_m["clean"]
    return {
        "n": (mf.get("n"), mc.get("n")),
        "acc": (mf.get("acc"), mc.get("acc")),
        "macro_f1": (mf.get("macro_f1"), mc.get("macro_f1")),
        "f1_fake": (mf.get("f1_fake"), mc.get("f1_fake")),
        "pr_auc": (mf.get("pr_auc"), mc.get("pr_auc")),
        "roc_auc": (mf.get("roc_auc"), mc.get("roc_auc")),
        "brier": (mf.get("brier"), mc.get("brier")),
        "ece": (mf.get("ece"), mc.get("ece")),
        "worst_group_macro_f1": (mf.get("worst_group_macro_f1"),
                                 mc.get("worst_group_macro_f1")),
        "worst_group_balanced": (mf.get("worst_group_balanced"),
                                 mc.get("worst_group_balanced")),
        "hard_acc": (mf.get("hard_acc"), mc.get("hard_acc")),
        "ptpt_macro_f1": (mf.get("ptpt_macro_f1"), mc.get("ptpt_macro_f1")),
    }


def md_channel_table(set_m: dict) -> str:
    full = {r["channel"]: r for r in set_m["full"]["by_cut"]["channel"]}
    clean = {r["channel"]: r for r in set_m["clean"]["by_cut"]["channel"]}
    names = sorted(set(full) | set(clean),
                   key=lambda c: (full.get(c, {}).get("macro_f1")
                                  if full.get(c, {}).get("macro_f1") is not None
                                  else 2.0, c))
    lines = ["| canal | n (full) | acc (full) | macro-F1 (full) | n (limpo) | "
             "acc (limpo) | macro-F1 (limpo) | novo v6 (a preencher) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in names:
        f = full.get(c)
        cl = clean.get(c)
        lines.append(
            f"| `{c}` | {f['n'] if f else '-'} | "
            f"{fmt_br(f['acc']) if f else '-'} | "
            f"{fmt_br(f['macro_f1']) if f else '-'} | "
            f"{cl['n'] if cl else '-'} | {fmt_br(cl['acc']) if cl else '-'} | "
            f"{fmt_br(cl['macro_f1']) if cl else '-'} | - |")
    return "\n".join(lines)


def write_markdown(path: Path, metrics: dict, meta: dict) -> None:
    sets = metrics["sets"]
    full = metrics["contamination"]
    reg = REGISTERED_V1_IID
    v1r = metrics["v1_reference"]
    blocks = {}
    for key in ("full_iid", "ood_wa"):
        if key in sets:
            blocks[key] = md_metric_block(sets[key])
    b_iid = blocks.get("full_iid", {})
    b_ood = blocks.get("ood_wa", {})

    def cell_iid(metric: str, which: int) -> str:
        vals = b_iid.get(metric)
        if not vals:
            return "-"
        v = vals[which]
        return num_br(v) if metric == "n" else fmt_br(v)

    def row(label: str, metric: str, registered: str = "-") -> str:
        return (f"| {label} | {registered} | {cell_iid(metric, 0)} | "
                f"{cell_iid(metric, 1)} | - |")

    c_iid = full["full_iid"]
    lines = []
    lines.append("# Comparacao BASE v1 x BASE v6 -- testes full_iid e OOD WhatsApp")
    lines.append("")
    lines.append("Gerado por `models/v6/compare_v1_vs_v6.py` em "
                 f"{meta['created_utc']} (CPU, {meta['elapsed_min']:.1f} min).")
    lines.append("")
    lines.append("Modelo avaliado: `models/artifacts/bertimbau_finetuned` "
                 "(v1, hidden 768, 12 camadas), Platt a="
                 f"{fmt_br(meta['platt_a'])} b={fmt_br(meta['platt_b'])} "
                 "(`calibration.json`), max_length "
                 f"{meta['max_length']}, batch {meta['batch_size']}, limiar "
                 f"{fmt_br(meta['threshold'], 2)}. Inferencia em CPU, logits "
                 "fp32.")
    lines.append("")

    lines.append("## 1. Contaminacao do teste v6 pelo treino v1")
    lines.append("")
    lines.append("A v1 foi reconstruida com "
                 "`models.data.load(FakenewsBR_sanitized.csv)` + "
                 "`iid_split(seed=42)`: "
                 f"n={num_br(v1r['n'])} ({pct_br(v1r['fake_pct'])}% fake), "
                 f"treino {num_br(v1r['n_train'])} / val {num_br(v1r['n_val'])} "
                 f"/ teste {num_br(v1r['n_test'])}.")
    lines.append("")
    lines.append("| conjunto | n testado | rids no TREINO v1 | % contaminado | "
                 "n limpo | rids no val v1 (nao treinados) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ("full_iid", "ood_wa"):
        if name not in full:
            continue
        c = full[name]
        lines.append(
            f"| {name} | {num_br(c['n_test'])} | "
            f"{num_br(c['n_contam_v1_train'])} | "
            f"{pct_br(c['pct_contam_v1_train'])}% | "
            f"{num_br(c['n_clean'])} | {num_br(c['n_overlap_v1_val'])} |")
    lines.append("")
    lines.append("`contaminado` = rid presente no TREINO v1 (o antigo viu o "
                 "texto com rotulo no fine-tuning). O corte `limpo` remove "
                 "apenas esses; linhas que ficaram no val/teste v1 nao foram "
                 "treinadas, mas participaram de selecao de checkpoint e "
                 "calibracao do antigo -- limitacao residual registrada.")
    lines.append("")

    lines.append("## 2. Metricas globais (threshold 0,5)")
    lines.append("")
    lines.append("| metrica | antigo v1 no teste v1 (registrado) | "
                 "antigo v1 no teste v6 | antigo v1 no teste v6 limpo | "
                 "novo v6 no teste v6 (a preencher apos Colab) |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(row("n", "n", num_br(reg["n"])))
    lines.append(row("acuracia", "acc", fmt_br(reg["acc"])))
    lines.append(row("macro-F1", "macro_f1", fmt_br(reg["macro_f1"])))
    lines.append(row("F1 (fake)", "f1_fake", fmt_br(reg["f1_fake"])))
    lines.append(row("PR-AUC", "pr_auc", fmt_br(reg["pr_auc"])))
    lines.append(row("ROC-AUC", "roc_auc", fmt_br(reg["roc_auc"])))
    lines.append(row("Brier", "brier", fmt_br(reg["brier"])))
    lines.append(row("ECE (bins 15)", "ece", fmt_br(reg["ece"])))
    lines.append(row("pior-grupo macro-F1", "worst_group_macro_f1",
                     fmt_br(reg["worst_group_macro_f1"])))
    lines.append(row("pior-grupo (balanceados)", "worst_group_balanced",
                     fmt_br(reg["worst_group_balanced"])))
    lines.append(row("casos limitrofes (acc)", "hard_acc",
                     fmt_br(reg["hard_acc"])))
    lines.append(row("PT-PT macro-F1", "ptpt_macro_f1",
                     fmt_br(reg["ptpt_macro_f1"])))
    lines.append("")
    lines.append(f"Registrado de: `{reg['source']}`. Teste v1 n=5.920; "
                 f"teste v6 full_iid n={num_br(c_iid['n_test'])} pos-U+FFFD "
                 f"({num_br(c_iid['n_contam_v1_train'])} linhas contaminadas, "
                 f"{pct_br(c_iid['pct_contam_v1_train'])}%), limpo "
                 f"n={num_br(c_iid['n_clean'])}.")
    lines.append("")

    if "full_iid" in sets:
        lines.append("## 3. Por canal -- teste full_iid (macro-F1 via "
                     "`models.evaluate.per_group`, min_n=30)")
        lines.append("")
        lines.append(md_channel_table(sets["full_iid"]))
        lines.append("")

    if "ood_wa" in sets:
        lines.append("## 4. OOD WhatsApp -- teste ood_wa "
                     "(leave-one-channel-out)")
        lines.append("")
        lines.append("| metrica | antigo v1 no ood_wa v6 | antigo v1 no "
                     "ood_wa v6 limpo | novo v6 (a preencher apos Colab) |")
        lines.append("|---|---:|---:|---:|")
        names = [("n", "n"), ("acuracia", "acc"), ("macro-F1", "macro_f1"),
                 ("F1 (fake)", "f1_fake"), ("PR-AUC", "pr_auc"),
                 ("ROC-AUC", "roc_auc"), ("Brier", "brier"),
                 ("ECE (bins 15)", "ece"),
                 ("pior-grupo macro-F1", "worst_group_macro_f1")]
        for label, key in names:
            vals = b_ood.get(key)
            if key == "n":
                a_s, b_s = num_br(vals[0]), num_br(vals[1])
            else:
                a_s, b_s = fmt_br(vals[0]), fmt_br(vals[1])
            lines.append(f"| {label} | {a_s} | {b_s} | - |")
        lines.append("")
        lines.append("Nao ha coluna `registrado` para o OOD: os numeros OOD da "
                     "v1 em `RELATORIO_DADOS.md`/`score_finetuned.json` vem do "
                     "pipeline de score (regressao logistica sobre embeddings + "
                     "calibracao sob prior balanceado), nao do encoder puro "
                     "avaliado aqui -- comparacao direta seria enganosa.")
        lines.append("")

    lines.append("## 5. Nota metodologica (diferencas entre os dois runs)")
    lines.append("")
    lines.append("1. **Dados**: v1 = 39.466 linhas "
                 "(`FakenewsBR_sanitized.csv`, 71,5% fake, sem dedup por "
                 "cluster); v6 = 91.080 linhas (`v6_pool.parquet`, com dedup "
                 "exata/quase-dup e rotulos estratificados por tier). O teste "
                 "v6 nao e amostra do teste v1; a v6 e uma expansao com novas "
                 "fontes (FC_*, NEWS_*).")
    lines.append("2. **DFR on/off**: o antigo e BASE puro (encoder + Platt). "
                 "O default do trainer v6 liga DFR de celula "
                 "(`--dfr-weights cell --weight-clip 25`), que reequilibra os "
                 "grupos e muda o prior efetivo (~0,60) e a calibracao. Para "
                 "um confronto BASE x BASE, o run v6 deve ser "
                 "`--dfr-weights off`; a coluna final do MD deve registrar "
                 "qual foi usado.")
    lines.append("3. **Splits**: v1 usa IID estratificado por (grupo x rotulo) "
                 "em uma base sem controle de quase-duplicata; v6 usa "
                 "`full_iid` (estratificado por grupo x rotulo com dedup/"
                 "clusters) e `ood_wa` (canal WhatsApp inteiramente retido). "
                 f"O teste v1 e ~5.920 linhas; o full_iid v6 tem "
                 f"{num_br(c_iid['n_test'])} (pos-U+FFFD).")
    lines.append("4. **Contaminacao medida**: "
                 f"{pct_br(c_iid['pct_contam_v1_train'])}% do teste full_iid "
                 "v6 estava no treino v1; o corte `limpo` existe para isolar "
                 "essa memoria. A contaminacao e estrutural: o pool v6 "
                 "reaproveita as linhas da v1.")
    lines.append("5. **Metricas**: todas de `models/evaluate.py` "
                 "(`core_metrics`, `worst_group_f1`, `per_group`, "
                 "`expected_calibration_error`); ECE com 15 bins; pior-grupo "
                 "restrito a grupos com n>=100 e minoria>=20. Platt e limiar "
                 "0,5 identicos aos do artefato antigo.")
    lines.append("6. **U+FFFD**: removidas defensivamente do pool (4 linhas), "
                 "como faz `train_bertimbau_v6.py`; o teste full_iid fica com "
                 f"{num_br(c_iid['n_test'])} linhas.")
    lines.append("")
    lines.append("## 6. Como preencher a coluna do v6 novo (Colab)")
    lines.append("")
    lines.append("Depois do run no Colab, copiar de "
                 "`runs/<run_id>/metrics.json` o bloco `test` "
                 "(e `ood` para o OOD) para a ultima coluna das tabelas:")
    lines.append("")
    lines.append("```text")
    lines.append("R0 default (full_iid + DFR cell, 2 epocas, freeze 6, ml192):")
    lines.append("  runs/full_iid_ml192_f6_on_seed42/metrics.json -> test.*")
    lines.append("BASE puro equivalente ao antigo (sem DFR):")
    lines.append("  python models/v6/train_bertimbau_v6.py --data ... --splits ... \\")
    lines.append("    --split-col full_iid --dfr-weights off --max-length 192 \\")
    lines.append("    --batch-size 32 --epochs 2 --freeze-layers 6 --seed 42")
    lines.append("```")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--v1-csv", default=D.DEFAULT_CSV)
    ap.add_argument("--pool", default=str(ROOT / "models/v6/processed/v6_pool.parquet"))
    ap.add_argument("--splits", default=str(ROOT / "models/v6/processed/v6_splits.parquet"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--md", default=str(DEFAULT_MD))
    ap.add_argument("--split-col", default="full_iid")
    ap.add_argument("--ood-col", default="ood_wa")
    ap.add_argument("--ood", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval-value", default="test")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: avalia uma amostra de N linhas de cada teste")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t_start = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("[AVISO] CUDA disponivel, mas a comparacao roda em CPU por spec")

    model_dir = Path(args.model)
    cal_path = Path(args.calibration) if args.calibration \
        else model_dir / "calibration.json"
    cal, a, b, max_length_cal, use_mask = load_calibration(cal_path)
    if args.max_length is None:
        args.max_length = max_length_cal
    args.use_mask = use_mask
    print(f"[E0] model={model_dir} calib={cal_path} "
          f"platt_a={a:.4f} platt_b={b:.4f} max_length={args.max_length} "
          f"batch={args.batch_size} threads={args.threads} device={device} "
          f"mask_entities={use_mask}")

    split_cols = [args.split_col] + ([args.ood_col] if args.ood else [])
    pool, pool_info = load_v6(Path(args.pool), Path(args.splits), split_cols)

    print(f"[E1] reconstruindo split v1: {args.v1_csv}")
    v1_df, v1_split, v1_info = build_v1_split(args.v1_csv, args.seed)
    print(f"[E1] v1: n={v1_info['n']} ({v1_info['fake_pct']}% fake) "
          f"treino={v1_info['n_train']} val={v1_info['n_val']} "
          f"teste={v1_info['n_test']}")
    expected = {"n_train": 27626, "n_val": 5920, "n_test": 5920}
    v1_ok = all(v1_info[k] == v for k, v in expected.items())

    print(f"[E2] carregando encoder antigo de {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    args.v1_split = v1_split
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [("full_iid", args.split_col)]
    if args.ood:
        specs.append(("ood_wa", args.ood_col))
    sets = {}
    contamination = {}
    for name, col in specs:
        df_test = pool[pool[col] == args.eval_value].reset_index(drop=True)
        if args.limit and args.limit < len(df_test):
            df_test = df_test.sample(n=args.limit,
                                     random_state=args.seed).reset_index(drop=True)
        if df_test.empty:
            print(f"[ERRO] conjunto vazio para {col}={args.eval_value}")
            return 2
        sub = out_dir if name == "full_iid" else out_dir / "ood_wa"
        sub.mkdir(parents=True, exist_ok=True)
        set_m = run_eval_set(name, df_test, tok, model, a, b, args, device, sub)
        sets[name] = set_m
        contamination[name] = {
            "split_col": col,
            "n_test": set_m["n_test"],
            "n_contam_v1_train": set_m["n_contam_v1_train"],
            "pct_contam_v1_train": set_m["pct_contam_v1_train"],
            "n_clean": set_m["n_clean"],
            "n_overlap_v1_val": set_m["n_overlap_v1_val"],
            "n_overlap_v1_test": set_m["n_overlap_v1_test"],
        }

    elapsed = time.perf_counter() - t_start
    metrics = {
        "meta": {
            "created_utc": utcnow(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "model_dir": str(model_dir),
            "calibration": cal,
            "platt_a": a,
            "platt_b": b,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "threshold": args.threshold,
            "threads": args.threads,
            "seed": args.seed,
            "device": str(device),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "elapsed_min": round(elapsed / 60, 2),
            "limit": args.limit,
            "ood": bool(args.ood),
            "pool": str(args.pool),
            "splits": str(args.splits),
            "pool_info": pool_info,
        },
        "v1_reference": {**v1_info, "registered_test_iid": REGISTERED_V1_IID},
        "contamination": contamination,
        "sets": sets,
        "checks": {},
    }

    checks = {"v1_split_sizes_esperados": {
        "esperado": expected,
        "obtido": {k: v1_info[k] for k in expected},
        "ok": bool(v1_ok)}}
    required = ("n", "acc", "macro_f1", "f1_fake", "brier", "ece",
                "roc_auc", "pr_auc")
    optional = ("worst_group_macro_f1", "worst_group_balanced", "hard_acc",
                "ptpt_macro_f1")
    fatal, notes = [], []
    if not v1_ok:
        fatal.append("tamanhos do split v1 divergem do esperado")
    for name, sm in sets.items():
        for cut in ("full", "clean"):
            m = sm[cut]
            for k in required:
                v = m.get(k)
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    fatal.append(f"{name}.{cut}.{k} ausente/nao-finito")
            for k in optional:
                v = m.get(k)
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    notes.append(f"{name}.{cut}.{k} indefinido "
                                 "(sem grupo confiavel ou corte degenerado)")
        if sm["n_contam_v1_train"] + sm["n_clean"] != sm["n_test"]:
            fatal.append(f"{name}: contaminado + limpo != n_test")
        if sm["full"]["n"] != sm["n_test"]:
            fatal.append(f"{name}: full.n != n_test")
        if sm["clean"]["n"] != sm["n_clean"]:
            fatal.append(f"{name}: clean.n != n_clean")
    checks["sem_nan_silencioso"] = {
        "metricas_obrigatorias_finitas": not fatal,
        "n_consistente": not fatal,
        "indefinidos_explicitos": notes,
    }
    checks["warnings"] = fatal + notes
    metrics["checks"] = checks
    save_json(metrics, out_dir / "metrics.json")

    md_meta = {"created_utc": metrics["meta"]["created_utc"],
               "elapsed_min": metrics["meta"]["elapsed_min"],
               "platt_a": a, "platt_b": b, "max_length": args.max_length,
               "batch_size": args.batch_size, "threshold": args.threshold}
    write_markdown(Path(args.md), metrics, md_meta)

    print("\n" + "=" * 70)
    print("RESUMO COMPARACAO v1 BASE nos testes v6")
    print("=" * 70)
    for name, sm in sets.items():
        f, c = sm["full"], sm["clean"]
        print(f"{name}: n={sm['n_test']} contaminado="
              f"{sm['n_contam_v1_train']} ({sm['pct_contam_v1_train']:.2f}%) "
              f"limpo={sm['n_clean']}")
        print(f"  full : acc={fmt_br(f['acc'])} macro-F1={fmt_br(f['macro_f1'])} "
              f"F1fake={fmt_br(f['f1_fake'])} PR-AUC={fmt_br(f['pr_auc'])} "
              f"ROC-AUC={fmt_br(f['roc_auc'])} Brier={fmt_br(f['brier'])} "
              f"ECE={fmt_br(f['ece'])} pior-grupo={fmt_br(f['worst_group_macro_f1'])}")
        print(f"  limpo: acc={fmt_br(c['acc'])} macro-F1={fmt_br(c['macro_f1'])} "
              f"F1fake={fmt_br(c['f1_fake'])} PR-AUC={fmt_br(c['pr_auc'])} "
              f"ROC-AUC={fmt_br(c['roc_auc'])} Brier={fmt_br(c['brier'])} "
              f"ECE={fmt_br(c['ece'])} pior-grupo={fmt_br(c['worst_group_macro_f1'])}")
    print(f"\nmetrics.json: {out_dir / 'metrics.json'}")
    print(f"markdown: {args.md}")
    print(f"tempo total: {elapsed/60:.1f} min")
    if fatal:
        print(f"[ERRO] {len(fatal)} checagem(ns) obrigatoria(s) falhou:")
        for w in fatal:
            print("  -", w)
        return 1
    if notes:
        print(f"[ATENCAO] {len(notes)} metrica(s) de grupo indefinida(s):")
        for w in notes:
            print("  -", w)
    print("[OK] n consistente e metricas globais finitas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
