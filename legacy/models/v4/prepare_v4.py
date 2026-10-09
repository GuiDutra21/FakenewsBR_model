#!/usr/bin/env python
"""prepare_v4.py -- CSVs v4 -> parquet enxuto + splits 4 vias + estatisticas.

Roda UMA vez, no repo local (depende de `models.data` para reproduzir as regras
de `train_label`, grupo, canal, `is_balanced_group` e `_safe_strat`). As saidas
ficam fora do Git (`models/.gitignore` ignora `artifacts/`).

Decisoes seguem `models/v4/PLANO_FT_V4_T4.md`; contrato em
`models/v4/spec_treino_v4.md` secao 3. Em caso de conflito, o PLANO vence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import data as D  # noqa: E402  (precisa do sys.path acima)

TEXT_COL = D.TEXT_COL
SPLIT_COLS = ("split_bal_iid", "split_bal_ood_wa", "split_full_iid")

EXPECTED_POOL = 85_212
EXPECTED_FAKE = 60_991
EXPECTED_TRUE = 24_221
EXPECTED_INFO = 36_896
EXPECTED_CONSTANT = 39_351
EXPECTED_UFFFD = 4

EXPECTED_COUNTS = {
    "split_bal_iid": {"train": 25_827, "val_sel": 3_689, "val_calib": 1_845,
                      "test": 5_535, "unused": EXPECTED_POOL - EXPECTED_INFO},
    "split_bal_ood_wa": {"train": 25_937, "val_sel": 3_052, "val_calib": 1_526,
                         "test": 6_381, "unused": EXPECTED_POOL - EXPECTED_INFO},
    "split_full_iid": {"train": 67_725, "val_sel": 7_968, "val_calib": 3_984,
                       "test": 0, "unused": EXPECTED_POOL - 67_725 - 7_968 - 3_984},
}

# Latin Extended-A/B: regex com caracteres literais (pyarrow nao aceita \u).
LATIN_EXT_RE = "[" + chr(0x0100) + "-" + chr(0x024F) + "]"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_era(date_iso) -> str:
    s = "" if date_iso is None else str(date_iso).strip()
    if len(s) < 4 or not s[:4].isdigit():
        return "sem_data"
    y = int(s[:4])
    if y <= 2017:
        return "<=2017"
    if y <= 2022:
        return "2018-2022"
    return ">=2023"


def four_way(df: pd.DataFrame, val_sel_frac: float, val_calib_frac: float,
             test_frac: float, seed: int):
    """Split aninhado 70/10/5/15 (ou 85/10/5) replicando `_safe_strat`.

    Ordem exata que reproduz as contagens do PLANO 2.1 (tolerancia 0):
      1. separa hold = val_sel + val_calib + test (ceil de sklearn);
      2. tira val_calib de hold com fracao val_calib/hold;
      3. se test_frac > 0, tira test do restante com test/(val_sel+test).
    """
    label = df["label"].astype(str)
    strat = D._safe_strat(df["group"] + "|" + label, label, min_count=10)
    hold = val_sel_frac + val_calib_frac + test_frac
    tr, tmp = train_test_split(df, test_size=hold, random_state=seed,
                               stratify=strat)
    strat_tmp = D._safe_strat(strat.loc[tmp.index], label.loc[tmp.index],
                              min_count=2)
    rest, vc = train_test_split(tmp, test_size=val_calib_frac / hold,
                                random_state=seed, stratify=strat_tmp)
    if test_frac > 0:
        strat_rest = D._safe_strat(strat.loc[rest.index], label.loc[rest.index],
                                   min_count=2)
        vs, te = train_test_split(rest, test_size=test_frac / (val_sel_frac + test_frac),
                                  random_state=seed, stratify=strat_rest)
    else:
        vs, te = rest, None
    return tr, vs, vc, te


def assign(col: pd.Series, df: pd.DataFrame, value: str) -> None:
    col.loc[df.index] = value


def split_counts(splits: pd.DataFrame, pool: pd.DataFrame) -> dict:
    merged = pool[["rid", "label"]].merge(splits, on="rid", how="left")
    out = {}
    for col in SPLIT_COLS:
        vc = merged[col].value_counts(dropna=False)
        d = {}
        for val in ("train", "val_sel", "val_calib", "test", "unused"):
            sub = merged[merged[col] == val]
            d[val] = {
                "n": int(len(sub)),
                "fake": int((sub["label"] == "fake").sum()),
                "true": int((sub["label"] == "true").sum()),
                "fake_pct": round(float((sub["label"] == "fake").mean() * 100)
                                  if len(sub) else 0.0, 2),
            }
        d["_null"] = int(vc.get(np.nan, 0)) if vc.index.hasnans else 0
        out[col] = d
    return out


def check_equal(name: str, got, want) -> dict:
    ok = got == want
    return {"ok": bool(ok), "got": got, "want": want, "name": name}


# --------------------------------------------------------------- quase-dup
def shingles(text: str, n: int = 5) -> set:
    ws = str(text).split()
    if len(ws) < n:
        return set()
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def near_dup_probe(train_texts, test_texts, n_sample: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    tr = list(train_texts)
    te = list(test_texts)
    tr_sample = [tr[i] for i in rng.choice(len(tr), size=min(n_sample, len(tr)), replace=False)]
    te_sample = [te[i] for i in rng.choice(len(te), size=min(n_sample, len(te)), replace=False)]
    t0 = time.time()
    tr_sets = [shingles(t) for t in tr_sample]
    te_sets = [shingles(t) for t in te_sample]
    inv: dict = {}
    for j, s in enumerate(te_sets):
        for g in s:
            inv.setdefault(g, []).append(j)
    max_j, n_above, n_pairs = 0.0, 0, 0
    for s in tr_sets:
        if not s:
            continue
        hits: Counter = Counter()
        for g in s:
            for j in inv.get(g, ()):
                hits[j] += 1
        for j, inter in hits.items():
            if inter < 2:
                continue
            n_pairs += 1
            union = len(s) + len(te_sets[j]) - inter
            jac = inter / union if union else 0.0
            if jac > max_j:
                max_j = jac
            if jac > 0.8:
                n_above += 1
    return {
        "metric": "5-gram word shingles, Jaccard exato na amostra (indice invertido)",
        "n_train_sample": len(tr_sample),
        "n_test_sample": len(te_sample),
        "n_pares_com_intersecao": int(n_pairs),
        "max_jaccard": round(float(max_j), 6),
        "n_pares_gt_0.8": int(n_above),
        "seconds": round(time.time() - t0, 2),
    }


# ------------------------------------------------------------- token stats
def length_grouped_batches(lengths, batch_size, generator, mega=50):
    """Copia verbatim de models/encoder.py:83 (mesma simulacao do PLANO)."""
    idx = torch.randperm(len(lengths), generator=generator).tolist()
    span = batch_size * mega
    out = []
    for i in range(0, len(idx), span):
        chunk = sorted(idx[i:i + span], key=lambda j: lengths[j])
        out += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    perm = torch.randperm(len(out), generator=generator).tolist()
    return [out[i] for i in perm]


def dist_stats(lens: np.ndarray) -> dict:
    lens = np.asarray(lens, dtype=np.int64)
    return {
        "n": int(len(lens)),
        "mean": round(float(lens.mean()), 2),
        "p50": round(float(np.percentile(lens, 50)), 2),
        "p90": round(float(np.percentile(lens, 90)), 2),
        "p95": round(float(np.percentile(lens, 95)), 2),
        "p99": round(float(np.percentile(lens, 99)), 2),
        "max": int(lens.max()),
        "pct_gt_128": round(float((lens > 128).mean() * 100), 2),
        "pct_gt_192": round(float((lens > 192).mean() * 100), 2),
        "pct_gt_256": round(float((lens > 256).mean() * 100), 2),
        "pct_gt_512": round(float((lens > 512).mean() * 100), 2),
    }


def cap_cost(lens: np.ndarray, cap: int, batch_size: int = 32, mega: int = 50,
             seed: int = 42) -> dict:
    clipped = np.minimum(np.asarray(lens, dtype=np.int64), cap)
    g = torch.Generator().manual_seed(seed)
    batches = length_grouped_batches(clipped.tolist(), batch_size, g, mega)
    paid = 0
    at_cap = 0
    for b in batches:
        m = max(clipped[j] for j in b)
        paid += len(b) * int(m)
        if m >= cap:
            at_cap += 1
    n = int(len(clipped))
    return {
        "cap": int(cap),
        "batch_size": batch_size,
        "mega": mega,
        "tokens_pagos_por_epoca": int(paid),
        "tokens_pagos_por_epoca_milhoes": round(paid / 1e6, 3),
        "media_por_amostra": round(paid / max(n, 1), 2),
        "n_batches": len(batches),
        "pct_batches_no_cap": round(100.0 * at_cap / max(len(batches), 1), 2),
    }


def compute_token_stats(tokenizer, texts_by_cut: dict, cost_texts) -> dict:
    result = {"tokenizer": tokenizer.name_or_path, "add_special_tokens": True,
              "truncation": False, "distributions": {}, "cost": {}}
    lens_by_cut = {}
    for name, texts in texts_by_cut.items():
        t0 = time.time()
        lens = np.empty(len(texts), dtype=np.int64)
        texts = list(texts)
        for i in range(0, len(texts), 1024):
            enc = tokenizer(texts[i:i + 1024], add_special_tokens=True,
                            truncation=False)
            for j, ids in enumerate(enc["input_ids"]):
                lens[i + j] = len(ids)
        lens_by_cut[name] = lens
        d = dist_stats(lens)
        d["tokenize_seconds"] = round(time.time() - t0, 2)
        result["distributions"][name] = d
    # custo de batching no recorte pedido (treino do split selecionado)
    lens = lens_by_cut["train"] if "train" in lens_by_cut else np.asarray(
        list(map(len, [])), dtype=np.int64)
    for cap in (128, 192, 256):
        result["cost"][str(cap)] = cap_cost(lens, cap)
    result["cap_tokens_pagos_por_epoca"] = {
        str(c): result["cost"][str(c)]["tokens_pagos_por_epoca"] for c in (128, 192, 256)}
    base = result["cost"]["192"]["tokens_pagos_por_epoca"]
    result["razao_256_192"] = round(
        result["cost"]["256"]["tokens_pagos_por_epoca"] / base, 4) if base else None
    result["assumptions"] = {
        "cost_split": "bal_iid train",
        "batching": "length_grouped_batches(batch=32, mega=50, seed=42), comprimentos truncados no cap",
        "tokens_pagos": "sum(len(batch) * max(len_no_batch))",
    }
    return result


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sanitized", default=str(ROOT / "data" / "FakenewsBR_sanitized_v4.csv"))
    ap.add_argument("--labels", default=str(ROOT / "data" / "FakenewsBR_v4_labels.csv"))
    ap.add_argument("--provenance", default=str(ROOT / "data" / "FakenewsBR_v4_provenance.csv"))
    ap.add_argument("--out-dir", default="models/v4/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-sel-frac", type=float, default=0.10)
    ap.add_argument("--val-calib-frac", type=float, default=0.05)
    ap.add_argument("--ood-channel", default="whatsapp")
    ap.add_argument("--token-stats", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    pool_path = out_dir / "v4_pool.parquet"
    splits_path = out_dir / "v4_splits.parquet"
    stats_path = out_dir / "prepare_stats.json"
    tok_path = out_dir / "token_stats.json"
    outputs = [pool_path, splits_path, stats_path]
    if not args.force and any(p.exists() for p in outputs):
        print(f"[ERRO] saidas ja existem em {out_dir} (use --force para sobrescrever):")
        for p in outputs:
            if p.exists():
                print("  -", p)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    versions = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "pyarrow": __import__("pyarrow").__version__,
        "torch": torch.__version__,
    }

    print(f"[1/6] load({Path(args.sanitized).name}, ...)")
    t1 = time.time()
    df = D.load(csv=args.sanitized, labels_csv=args.labels,
                provenance_csv=args.provenance)
    load_s = time.time() - t1
    n_raw = len(df)
    fake_raw = int((df["label"] == "fake").sum())
    true_raw = int((df["label"] == "true").sum())
    print(f"      n={n_raw} fake={fake_raw} true={true_raw} ({load_s:.1f}s)")

    checks: dict[str, dict] = {}
    checks["pool_raw_85212"] = check_equal("pool_raw", n_raw, EXPECTED_POOL)
    checks["fake_raw_60991"] = check_equal("fake_raw", fake_raw, EXPECTED_FAKE)
    checks["true_raw_24221"] = check_equal("true_raw", true_raw, EXPECTED_TRUE)
    checks["rid_unico"] = {"name": "rid_unico", "ok": bool(df["rid"].is_unique),
                           "got": int(df["rid"].nunique()), "want": n_raw}

    print("[2/6] flags de ruido (U+FFFD, mojibake)")
    has_ufffd = df[TEXT_COL].astype(str).str.contains(chr(0xFFFD), regex=False)
    mojibake = df[TEXT_COL].astype(str).str.contains(LATIN_EXT_RE, regex=True)
    ufffd_rids = df.loc[has_ufffd, "rid"].astype("int64").tolist()
    checks["ufffd_4"] = check_equal("ufffd", int(has_ufffd.sum()), EXPECTED_UFFFD)
    print(f"      U+FFFD={int(has_ufffd.sum())} rids={ufffd_rids}")
    print(f"      mojibake(Latin Ext)= {int(mojibake.sum())}")

    print("[3/6] colunas derivadas")
    pool = pd.DataFrame({
        "rid": df["rid"].astype("int64"),
        "text_no_url": df[TEXT_COL].astype(str),
        "group": df["group"].astype(str),
        "channel": df["channel"].astype(str),
        "label": df["label"].astype(str),
        "label_tier": df["label_tier"].fillna("none").astype(str),
        "label_source": df["label_source"].astype(str),
        "is_balanced_group": df["is_balanced_group"].astype(bool),
        "is_ptpt_rule": df["is_ptpt"].astype(bool),
        "lang_variant": df["lang_variant"].fillna("desconhecido").astype(str),
        "publisher": df["publisher"].fillna("desconhecido").astype(str),
        "rating_class": df["rating_class"].astype(str),
        "era": df["date_iso"].map(parse_era).astype(str),
        "word_len": df["word_len"].fillna(0).astype("int32"),
        "num_exclamations": df["num_exclamations"].fillna(0).astype("int32"),
        "num_questions": df["num_questions"].fillna(0).astype("int32"),
        "num_ellipsis": df["num_ellipsis"].fillna(0).astype("int32"),
        "uppercase_word_ratio": df["uppercase_word_ratio"].fillna(0.0).astype("float32"),
        "has_ufffd": has_ufffd.astype(bool),
        "mojibake_flag": mojibake.astype(bool),
    })
    n_info = int(pool["is_balanced_group"].sum())
    # grupos de rotulo constante: um unico rotulo no grupo (PLANO 1.1)
    grp = pool.groupby("group")["label"]
    const_n = int(sum(len(sub) for _, sub in grp if sub.nunique() == 1))
    checks["informativos_36896"] = check_equal("informativos", n_info, EXPECTED_INFO)
    checks["constantes_39351"] = check_equal("constantes", const_n, EXPECTED_CONSTANT)
    print(f"      informativos={n_info} constantes={const_n}")

    print("[4/6] splits (estratificados grupo|rotulo, seed=%d)" % args.seed)
    bal = pool[pool["is_balanced_group"]]
    tr, vs, vc, te = four_way(bal, args.val_sel_frac, args.val_calib_frac,
                              args.test_frac, args.seed)
    splits = pd.DataFrame({"rid": pool["rid"].to_numpy(),
                           "split_bal_iid": "unused",
                           "split_bal_ood_wa": "unused",
                           "split_full_iid": "unused"})
    for name, sub in (("train", tr), ("val_sel", vs), ("val_calib", vc), ("test", te)):
        if sub is not None:
            splits.loc[sub.index, "split_bal_iid"] = name

    wa = bal[bal["channel"] == args.ood_channel]
    rest_info = bal[bal["channel"] != args.ood_channel]
    if len(wa) == 0:
        print(f"[ERRO] canal OOD '{args.ood_channel}' vazio")
        return 2
    tr2, vs2, vc2, _ = four_way(rest_info, args.val_sel_frac, args.val_calib_frac,
                                0.0, args.seed)
    for name, sub in (("train", tr2), ("val_sel", vs2), ("val_calib", vc2),
                      ("test", wa)):
        splits.loc[sub.index, "split_bal_ood_wa"] = name

    holdout_rids = set(pool.loc[te.index, "rid"].tolist())
    remaining = pool[~pool["rid"].isin(holdout_rids)]
    tr3, vs3, vc3, _ = four_way(remaining, args.val_sel_frac, args.val_calib_frac,
                                0.0, args.seed)
    for name, sub in (("train", tr3), ("val_sel", vs3), ("val_calib", vc3)):
        splits.loc[sub.index, "split_full_iid"] = name
    # holdout (teste canonico) fica "unused" em full_iid
    counts = split_counts(splits, pool)
    for col, exp in EXPECTED_COUNTS.items():
        for val, want in exp.items():
            got = counts[col][val]["n"]
            checks[f"{col}.{val}"] = check_equal(f"{col}.{val}", got, want)
            print(f"      {col}.{val}: {got} (esperado {want})"
                  f"{'' if got == want else '   <<< DIVERGE'}")

    test_rids = set(pool.loc[te.index, "rid"])
    tr_rids = set(pool.loc[tr3.index, "rid"])
    for part, sub in (("train", tr3), ("val_sel", vs3), ("val_calib", vc3)):
        inter = set(pool.loc[sub.index, "rid"]) & test_rids
        checks[f"holdout_x_teste.{part}"] = check_equal(
            f"holdout_x_teste.{part}", len(inter), 0)
    checks["full_train_x_teste_conf"] = check_equal(
        "full_train_x_teste_conf", len(tr_rids & test_rids), 0)

    null_counts = {c: int(splits[c].isna().sum()) for c in SPLIT_COLS}
    checks["sem_nulos"] = {"name": "sem_nulos", "ok": all(v == 0 for v in null_counts.values()),
                           "got": null_counts, "want": {c: 0 for c in SPLIT_COLS}}
    bad_info = int((splits.loc[pool["is_balanced_group"].to_numpy(), "split_bal_iid"] == "unused").sum())
    checks["info_sem_unused_bal_iid"] = check_equal("info_sem_unused_bal_iid", bad_info, 0)
    wa_rids = set(pool.loc[wa.index, "rid"])
    ood_test_rids = set(pool["rid"][splits["split_bal_ood_wa"] == "test"])
    checks["ood_test_e_whatsapp"] = check_equal("ood_test_e_whatsapp",
                                                len(wa_rids ^ ood_test_rids), 0)

    probe = near_dup_probe(pool.loc[tr.index, TEXT_COL], pool.loc[te.index, TEXT_COL],
                           n_sample=5_000, seed=args.seed)
    print(f"      sonda quase-dup: max_jaccard={probe['max_jaccard']} "
          f"pares>0.8={probe['n_pares_gt_0.8']}")

    print("[5/6] escrevendo parquets")
    pool.to_parquet(pool_path, index=False, compression="snappy")
    splits.to_parquet(splits_path, index=False, compression="snappy")
    pool_mb = pool_path.stat().st_size / 1e6
    checks["parquet_le_25mb"] = {"name": "parquet_le_25mb",
                                 "ok": pool_mb <= 25.0,
                                 "got": round(pool_mb, 2), "want": 25.0}
    print(f"      v4_pool.parquet = {pool_mb:.2f} MB")
    print(f"      v4_splits.parquet = {splits_path.stat().st_size/1e3:.1f} KB")

    token_stats = None
    if args.token_stats:
        print("[5b/6] token stats (tokenizer BERTimbau cacheado)")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        cost_texts = pool.loc[tr.index, TEXT_COL]
        token_stats = compute_token_stats(
            tok,
            {"informativos": pool.loc[bal.index, TEXT_COL],
             "pool": pool[TEXT_COL],
             "train": cost_texts},
            cost_texts)
        info = token_stats["distributions"]["informativos"]
        print(f"      informativos: media={info['mean']} p95={info['p95']} "
              f">192={info['pct_gt_192']}%")
        print(f"      razao_256_192={token_stats['razao_256_192']}")
        tok_path.write_text(json.dumps(token_stats, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    failed = [k for k, v in checks.items() if not v["ok"]]
    stats = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": str(Path(__file__).resolve()),
        "seed": args.seed,
        "inputs": {
            "sanitized": {"path": str(Path(args.sanitized).resolve()),
                          "sha256": sha256_file(Path(args.sanitized))},
            "labels": {"path": str(Path(args.labels).resolve()),
                       "sha256": sha256_file(Path(args.labels))},
            "provenance": {"path": str(Path(args.provenance).resolve()),
                           "sha256": sha256_file(Path(args.provenance))},
        },
        "load_seconds": round(load_s, 1),
        "counts": {
            "pool_raw": n_raw, "fake_raw": fake_raw, "true_raw": true_raw,
            "pool": len(pool),
            "fake": int((pool["label"] == "fake").sum()),
            "true": int((pool["label"] == "true").sum()),
            "fake_pct": round(float((pool["label"] == "fake").mean() * 100), 2),
            "informativos": n_info,
            "informativos_fake_pct": round(
                float((pool.loc[bal.index, "label"] == "fake").mean() * 100), 2),
            "constant_groups": const_n,
            "ufffd_marked": int(has_ufffd.sum()),
            "ufffd_rids": ufffd_rids,
            "ufffd_policy": (
                "linhas U+FFFD ficam no pool marcadas has_ufffd=True e sao "
                "removidas do treino/eval pelo train_bertimbau_v4.py (E2). "
                "Remocao no prepare quebraria as contagens canonicas do PLANO "
                "1.1/2.1 (2 dos 4 rids sao informativos); desvio documentado em "
                "models/v4/BUILD_NOTES.md."),
            "mojibake_flagged": int(mojibake.sum()),
        },
        "splits": counts,
        "validation": {"checks": checks, "all_pass": not failed, "failed": failed},
        "near_dup_probe": probe,
        "sha256": {
            "v4_pool.parquet": sha256_file(pool_path),
            "v4_splits.parquet": sha256_file(splits_path),
        },
        "files": {
            "v4_pool.parquet": str(pool_path.resolve()),
            "v4_splits.parquet": str(splits_path.resolve()),
            "v4_pool_bytes": pool_path.stat().st_size,
            "v4_splits_bytes": splits_path.stat().st_size,
        },
        "versions": versions,
        "total_seconds": round(time.time() - t0, 1),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    print("[6/6] resumo")
    print(f"      all_pass={not failed} failed={failed}")
    print(f"      arquivos: {pool_path} | {splits_path} | {stats_path}"
          + (f" | {tok_path}" if token_stats else ""))
    if failed:
        print("[ERRO] validacoes falharam (codigo 2)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
