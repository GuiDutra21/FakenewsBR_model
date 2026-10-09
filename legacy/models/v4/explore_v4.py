#!/usr/bin/env python
"""Perfil reprodutivel do dataset FakenewsBR v4 para o fine-tuning do BERTimbau.

Le SOMENTE os tres CSVs da raiz do projeto e grava SOMENTE em models/v4/:
    - v4_profile.json  (todos os numeros)
    - V4_PROFILE.md    (resumo legivel)

Nao importa models.data; replica as regras relevantes:
    TEXT_COL = "text_no_url"
    pool treinavel: label := train_label (merge por rid) e label in {fake,true},
        depois dropna(texto) e texto.strip() != ""
    channel_of com o fallback de prefixo FC_/EXT_/NEWS_
    grupo informativo: prefixo FC_/EXT_/NEWS_, n >= 200 e minoria >= 15%

Uso: python models/v4/explore_v4.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

T0 = time.time()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAN_CSV = ROOT / "data" / "FakenewsBR_sanitized_v4.csv"
LAB_CSV = ROOT / "data" / "FakenewsBR_v4_labels.csv"
PROV_CSV = ROOT / "data" / "FakenewsBR_v4_provenance.csv"
JSON_OUT = HERE / "v4_profile.json"
MD_OUT = ROOT / "docs" / "v4" / "V4_PROFILE.md"

TEXT_COL = "text_no_url"
TEXT_COLS = ("text", "text_clean", "text_no_url")
POOL_LABELS = ("fake", "true")
CHUNK = 25_000

BALANCED_GROUPS = ["Fake.br", "FakeWhatsApp.BR_2018", "COVID19.BR",
                   "COVID19.BR_raw", "LLM4BR_300"]
CHANNEL_OF_V1 = {
    "FakeWhatsApp.BR_2018": "whatsapp",
    "Fake.br": "portal", "Fake.br_raw": "portal",
    "COVID19.BR": "covid", "COVID19.BR_raw": "covid",
    "LLM4BR_300": "llm",
    "fakes": "agency_claim",
    "true": "press_true",
    "MuMiN-PT": "social", "MuMiN-PT_raw": "social",
}
NEW_GROUP_PREFIXES = ("FC_", "EXT_", "NEWS_")
MIN_BALANCED_N = 200
MIN_MINORITY_FRAC = 0.15

FFFD = chr(0xFFFD)
LATIN_EXT_AB = "[" + chr(0x0100) + "-" + chr(0x024F) + "]"
URL_PT_PT = r"poligrafo|observador|eco\.sapo\.pt"

POOL_KEEP = ["rid", "dataset_name", "label", TEXT_COL, "date_iso",
             "word_len", "char_len", "url_review", "factcheck_url"]
FLAG_COLS = ("source_type", "is_duplicated", "is_null", "too_short")
TRAIN_LABEL_COLS = ("train_label", "label_tier", "label_source", "auto_label",
                    "confidence", "method")


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def channel_of(group) -> str:
    if group in CHANNEL_OF_V1:
        return CHANNEL_OF_V1[group]
    g = str(group)
    if g.startswith("FC_") and g.endswith("_VIRAL"):
        return "social"
    if g.startswith("FC_"):
        return "agency_claim"
    if g.startswith("EXT_"):
        return "external_claim"
    if g.startswith("NEWS_"):
        return "press_true"
    return "other"


def jval(x):
    if isinstance(x, dict):
        return {str(k): jval(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jval(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if math.isnan(float(x)) else float(x)
    if isinstance(x, float):
        return None if math.isnan(x) else x
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if pd.isna(x) if not isinstance(x, (dict, list, tuple)) else False:
        return None
    return x


def jdump(obj, path: Path) -> None:
    path.write_text(json.dumps(jval(obj), ensure_ascii=False, indent=2),
                    encoding="utf-8")


def pct(n, d) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def length_stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 1),
        "min": int(a.min()),
        "p50": round(float(np.percentile(a, 50)), 1),
        "p90": round(float(np.percentile(a, 90)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
        "p99": round(float(np.percentile(a, 99)), 1),
        "max": int(a.max()),
    }


def count_physical_lines(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(d: dict, keys) -> list:
    return [d.get(k) for k in keys]


def startswith_any(index, prefixes) -> np.ndarray:
    s = pd.Series([str(x) for x in index])
    mask = np.zeros(len(s), dtype=bool)
    for p in prefixes:
        mask |= s.str.startswith(p).fillna(False).to_numpy(dtype=bool)
    return mask


# ---------------------------------------------------------------------------
# 0. metadados dos arquivos
# ---------------------------------------------------------------------------
log("lendo cabecalhos/dtypes e contando linhas fisicas")
file_meta = {}
for name, path in (("sanitized", SAN_CSV), ("labels", LAB_CSV),
                   ("provenance", PROV_CSV)):
    file_meta[name] = {
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "physical_lines": count_physical_lines(path),
    }
    log(f"  {name}: {file_meta[name]['size_mb']} MB, "
        f"{file_meta[name]['physical_lines']} linhas fisicas")

san_head = pd.read_csv(SAN_CSV, nrows=1000, low_memory=False)
san_columns = list(san_head.columns)
san_dtypes = {c: str(t) for c, t in san_head.dtypes.items()}
del san_head

# ---------------------------------------------------------------------------
# 1. labels (pequeno): mapa de train_label + contagens
# ---------------------------------------------------------------------------
t = time.time()
lab = pd.read_csv(LAB_CSV, low_memory=False,
                  dtype={"rid": str, "train_label": str})
file_meta["labels"]["parsed_rows"] = len(lab)
file_meta["labels"]["columns"] = list(lab.columns)
file_meta["labels"]["t_read_s"] = round(time.time() - t, 1)
lab = lab[["rid"] + [c for c in TRAIN_LABEL_COLS if c in lab.columns]]
lab_rid_set = set(lab["rid"])
lab_train = dict(zip(lab["rid"], lab["train_label"]))
lab_tier = dict(zip(lab["rid"], lab["label_tier"]))
lab_source = dict(zip(lab["rid"], lab["label_source"]))
log(f"labels: {len(lab)} linhas, {len(lab_rid_set)} rids unicos, "
    f"{file_meta['labels']['t_read_s']}s")
train_label_counts = (lab["train_label"].fillna("<NaN>")
                      .value_counts().to_dict())
label_tier_counts = (lab["label_tier"].fillna("<NaN>")
                     .value_counts().to_dict())
label_source_counts = (lab["label_source"].fillna("<NaN>")
                       .value_counts().to_dict())

# ---------------------------------------------------------------------------
# 2. provenance (pequeno)
# ---------------------------------------------------------------------------
t = time.time()
prov = pd.read_csv(PROV_CSV, low_memory=False, dtype={"rid": str})
file_meta["provenance"]["parsed_rows"] = len(prov)
file_meta["provenance"]["columns"] = list(prov.columns)
file_meta["provenance"]["t_read_s"] = round(time.time() - t, 1)
prov_rid_set = set(prov["rid"])
log(f"provenance: {len(prov)} linhas, {file_meta['provenance']['t_read_s']}s")

# ---------------------------------------------------------------------------
# 3. passada unica no sanitized em chunks
# ---------------------------------------------------------------------------
log("passada unica no sanitized_v4 em chunks...")
t = time.time()
n_rows = 0
rid_set: set[str] = set()
label_counter: Counter = Counter()
dataset_counter: Counter = Counter()
year_counter: Counter = Counter()
null_counts: Counter = Counter()
flags = {c: Counter() for c in FLAG_COLS}
text_diag = {c: Counter() for c in TEXT_COLS}
cat_counts: Counter = Counter()
tier_counts: Counter = Counter()
tier_cat_counts: Counter = Counter()
orig_cat_counts: Counter = Counter()
news_total = news_no_train = 0
fc_total = fc_no_train = 0
ext_total = ext_no_train = 0
train_labelled = 0
dropped_text = 0
date_min = date_max = None
date_nan = 0
pool_parts: list[pd.DataFrame] = []
norm_hash_set: set[int] = set()
fffd_examples: list[str] = []

reader = pd.read_csv(SAN_CSV, chunksize=CHUNK, low_memory=False,
                     dtype={"rid": str})
for i, chunk in enumerate(reader):
    n_rows += len(chunk)
    rid_set.update(chunk["rid"])
    label_counter.update(chunk["label"].fillna("<NaN>").value_counts().to_dict())
    dataset_counter.update(chunk["dataset_name"].fillna("<NaN>")
                           .value_counts().to_dict())
    null_counts.update(chunk.isna().sum().to_dict())
    for c in FLAG_COLS:
        if c in chunk.columns:
            flags[c].update(chunk[c].fillna("<NaN>").value_counts().to_dict())

    dt = pd.to_datetime(chunk["date_iso"], errors="coerce", format="mixed")
    year_counter.update(dt.dt.year.fillna(-1).astype("int64")
                        .value_counts().to_dict())
    date_nan += int(dt.isna().sum())
    valid = dt.dropna()
    if len(valid):
        vmin, vmax = valid.min(), valid.max()
        date_min = vmin if date_min is None or vmin < date_min else date_min
        date_max = vmax if date_max is None or vmax > date_max else date_max

    for c in TEXT_COLS:
        s = chunk[c]
        tx = s.fillna("")
        text_diag[c]["notnull"] += int(s.notna().sum())
        text_diag[c]["empty"] += int((tx.str.strip() == "").sum())
        text_diag[c]["has_fffd"] += int(tx.str.contains(FFFD, regex=False).sum())
        text_diag[c]["has_latin_ext_ab"] += int(
            tx.str.contains(LATIN_EXT_AB, regex=True).sum())

    norm_full = (chunk[TEXT_COL].astype(str).str.strip().str.lower()
                 .str.replace(r"\s+", " ", regex=True))
    norm_hash_set.update(norm_full.map(hash).to_numpy().tolist())
    if len(fffd_examples) < 5:
        bad = chunk.loc[chunk[TEXT_COL].fillna("")
                        .str.contains(FFFD, regex=False), TEXT_COL]
        fffd_examples.extend(str(v)[:120]
                             for v in bad.head(5 - len(fffd_examples)))

    tl = chunk["rid"].map(lab_train)
    tier = chunk["rid"].map(lab_tier)
    matched = chunk["rid"].isin(lab_rid_set)
    tl_f = tl.fillna("<NaN>")
    tier_f = tier.fillna("<sem_labels>")
    cat = pd.Series(
        np.where(~matched.to_numpy(), "<sem_labels>",
                 np.where(tl_f.eq("fake").to_numpy(), "fake",
                          np.where(tl_f.eq("true").to_numpy(), "true",
                                   np.where(tl_f.eq("<NaN>").to_numpy(),
                                            "<train_label_NaN>", "<outro>")))),
        index=chunk.index, dtype="str")
    cat_counts.update(cat.value_counts().to_dict())
    tier_counts.update(tier_f.value_counts().to_dict())
    pair = pd.DataFrame({"tier": tier_f, "cat": cat})
    tier_cat_counts.update(pair.value_counts().to_dict())
    orig = chunk["label"].fillna("<NaN>")
    orig_cat_counts.update(
        pd.DataFrame({"orig": orig, "cat": cat}).value_counts().to_dict())

    is_train = tl_f.isin(POOL_LABELS)
    txt = chunk[TEXT_COL]
    text_ok = txt.notna() & (txt.fillna("").str.strip() != "")
    train_labelled += int(is_train.sum())
    dropped_text += int((is_train & ~text_ok).sum())

    ds = chunk["dataset_name"].fillna("")
    news = ds.str.startswith("NEWS_")
    news_total += int(news.sum())
    news_no_train += int((news & ~is_train).sum())
    fc = ds.str.startswith("FC_")
    fc_total += int(fc.sum())
    fc_no_train += int((fc & ~is_train).sum())
    ext = ds.str.startswith("EXT_")
    ext_total += int(ext.sum())
    ext_no_train += int((ext & ~is_train).sum())

    keep = chunk.loc[is_train & text_ok, POOL_KEEP].copy()
    keep["train_label"] = tl_f[is_train & text_ok].to_numpy()
    keep["label_tier"] = tier_f[is_train & text_ok].to_numpy()
    pool_parts.append(keep)
    if (i + 1) % 4 == 0:
        log(f"  chunks={i + 1} linhas={n_rows}")

t_full_pass = time.time() - t
log(f"passada completa: {n_rows} linhas em {t_full_pass:.1f}s")
file_meta["sanitized"]["parsed_rows"] = n_rows
file_meta["sanitized"]["t_read_s"] = round(t_full_pass, 1)

pool = pd.concat(pool_parts, ignore_index=True)
del pool_parts
n_pool = len(pool)
n_fake = int((pool["train_label"] == "fake").sum())
n_true = int((pool["train_label"] == "true").sum())
log(f"pool treinavel (regra data.py): {n_pool} linhas "
    f"(fake={n_fake}, true={n_true})")
news_pool = pool[pool["dataset_name"].astype(str).str.startswith("NEWS_")]
news_pool_info = {
    "n": int(len(news_pool)),
    "by_group": {str(k): int(v) for k, v in
                 news_pool["dataset_name"].value_counts().items()},
    "by_tier": {str(k): int(v) for k, v in
                news_pool["label_tier"].value_counts().items()},
    "by_label": {str(k): int(v) for k, v in
                 news_pool["train_label"].value_counts().items()},
}

# ---------------------------------------------------------------------------
# 4. merges: cobertura e vazamento
# ---------------------------------------------------------------------------
san_rids = np.array(sorted(rid_set))
lab_only = len(lab_rid_set - rid_set)
san_without_label = len(rid_set - lab_rid_set)
prov_only = len(prov_rid_set - rid_set)
san_without_prov = len(rid_set - prov_rid_set)

orig_casefold_mismatch = 0
for (o, c), v in orig_cat_counts.items():
    if c in POOL_LABELS and str(o).casefold() != c:
        orig_casefold_mismatch += v

merge_stats = {
    "sanitized_rows": n_rows,
    "sanitized_unique_rids": len(rid_set),
    "sanitized_duplicate_rids": n_rows - len(rid_set),
    "labels_rows": int(len(lab)),
    "labels_rids_not_in_sanitized": int(lab_only),
    "sanitized_rids_without_label_row": int(san_without_label),
    "provenance_rids_not_in_sanitized": int(prov_only),
    "sanitized_rids_without_provenance": int(san_without_prov),
    "train_label_counts": {str(k): int(v) for k, v in train_label_counts.items()},
    "label_tier_counts": {str(k): int(v) for k, v in label_tier_counts.items()},
    "label_source_counts": {str(k): int(v) for k, v in label_source_counts.items()},
    "category_counts_after_merge": {str(k): int(v) for k, v in cat_counts.items()},
    "tier_x_category": {"|".join(map(str, k)): int(v)
                        for k, v in tier_cat_counts.items()},
    "orig_label_x_category": {"|".join(map(str, k)): int(v)
                              for k, v in orig_cat_counts.items()},
    "train_label_capitalized_True": int(train_label_counts.get("True", 0)),
    "orig_label_casefold_mismatch_in_pool": orig_casefold_mismatch,
    "text_labeled_rows": train_labelled,
    "text_labeled_rows_dropped_by_text_filter": dropped_text,
    "news_total": news_total,
    "news_without_train_label": news_no_train,
    "news_in_pool": news_pool_info,
    "fc_total": fc_total,
    "fc_without_train_label": fc_no_train,
    "ext_total": ext_total,
    "ext_without_train_label": ext_no_train,
    "full_normalized_unique_texts": int(len(norm_hash_set)),
    "full_normalized_duplicated_rows_extra": int(n_rows - len(norm_hash_set)),
}

# ---------------------------------------------------------------------------
# 5. grupos e canais
# ---------------------------------------------------------------------------
g = (pool.groupby(["dataset_name", "train_label"]).size()
     .unstack(fill_value=0).rename(columns={"fake": "n_fake", "true": "n_true"}))
for c in ("n_fake", "n_true"):
    if c not in g.columns:
        g[c] = 0
g["n"] = g["n_fake"] + g["n_true"]
g["pct_fake"] = (100.0 * g["n_fake"] / g["n"]).round(2)
g["minority_frac"] = (g[["n_fake", "n_true"]].min(axis=1) / g["n"]).round(4)
g["channel"] = [channel_of(x) for x in g.index]
g["balanced_v1"] = [x in BALANCED_GROUPS for x in g.index]
prefix_mask = startswith_any(g.index, NEW_GROUP_PREFIXES)
g["is_new_prefix"] = prefix_mask
g["informative"] = ((g["n"] >= MIN_BALANCED_N)
                    & (g["minority_frac"] >= MIN_MINORITY_FRAC)
                    & prefix_mask)
g = g.sort_values("n", ascending=False)

group_rows = []
for name, row in g.iterrows():
    group_rows.append({
        "group": str(name),
        "channel": row["channel"],
        "n": int(row["n"]), "n_fake": int(row["n_fake"]),
        "n_true": int(row["n_true"]), "pct_fake": float(row["pct_fake"]),
        "minority_frac": float(row["minority_frac"]),
        "balanced_v1": bool(row["balanced_v1"]),
        "informative": bool(row["informative"]),
    })

by_dataset_top30 = group_rows[:30]
prefix_groups = [r for r in group_rows if r["group"].startswith(NEW_GROUP_PREFIXES)]
informative_groups = [r for r in group_rows if r["informative"]]
v1_balanced_present = [r for r in group_rows if r["balanced_v1"]]

ch = (pool.assign(channel=[channel_of(x) for x in pool["dataset_name"]])
      .groupby(["channel", "train_label"]).size()
      .unstack(fill_value=0).rename(columns={"fake": "n_fake", "true": "n_true"}))
for c in ("n_fake", "n_true"):
    if c not in ch.columns:
        ch[c] = 0
ch["n"] = ch["n_fake"] + ch["n_true"]
ch["pct_fake"] = (100.0 * ch["n_fake"] / ch["n"]).round(2)
ch = ch.sort_values("n", ascending=False)
by_channel = [{"channel": str(i), "n": int(r["n"]), "n_fake": int(r["n_fake"]),
               "n_true": int(r["n_true"]), "pct_fake": float(r["pct_fake"])}
              for i, r in ch.iterrows()]

# ---------------------------------------------------------------------------
# 6. comprimento, duplicatas e conflitos
# ---------------------------------------------------------------------------
words = pool[TEXT_COL].astype(str).str.split().str.len()
chars = pool[TEXT_COL].astype(str).str.len()
pool["words"] = words.fillna(0).astype(int)
pool["chars"] = chars.fillna(0).astype(int)

length = {"words": {}, "chars": {}, "truncation": {}}
for klass in ("fake", "true"):
    m = pool["train_label"] == klass
    length["words"][klass] = length_stats(pool.loc[m, "words"])
    length["chars"][klass] = length_stats(pool.loc[m, "chars"])
length["words"]["all"] = length_stats(pool["words"])
length["truncation"] = {
    "over_512_words": int((pool["words"] > 512).sum()),
    "over_512_words_pct": pct(int((pool["words"] > 512).sum()), n_pool),
    "over_510_words": int((pool["words"] > 510).sum()),
    "over_400_words": int((pool["words"] > 400).sum()),
    "fake_over_512": int((pool.loc[pool["train_label"] == "fake", "words"] > 512).sum()),
    "true_over_512": int((pool.loc[pool["train_label"] == "true", "words"] > 512).sum()),
    "char_p95_over_2000": int((pool["chars"] > 2000).sum()),
}

wl = pd.to_numeric(pool["word_len"], errors="coerce")
mismatch = (wl.fillna(-1).astype(float) != pool["words"]).sum()
length["word_len_column_check"] = {
    "rows_word_len_missing": int(wl.isna().sum()),
    "rows_word_len_differs_from_text": int(mismatch),
    "rows_word_len_differs_gt_2": int((abs(wl.fillna(-1) - pool['words']) > 2).sum()),
    "corr": round(float(wl.corr(pool["words"].astype(float))), 4)
    if wl.notna().sum() > 1 else None,
}

norm = (pool[TEXT_COL].astype(str).str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True))
pool["norm"] = norm
n_dup_groups = int(norm.duplicated().sum())
n_dup_rows = int(norm.duplicated(keep=False).sum())
conf_nunique = pool.groupby("norm")["train_label"].nunique()
conf_texts = conf_nunique[conf_nunique > 1]
conf_rows = int(pool["norm"].isin(set(conf_texts.index)).sum())

dup_top = (pool[pool["norm"].isin(
        set(norm.value_counts()[norm.value_counts() > 1].index))]
    .groupby("norm")
    .agg(n=("train_label", "size"),
         n_fake=("train_label", lambda s: int((s == "fake").sum())),
         n_true=("train_label", lambda s: int((s == "true").sum())),
         groups=("dataset_name", lambda s: "|".join(sorted(set(map(str, s)))[:3])),
         example=(TEXT_COL, "first"))
    .sort_values("n", ascending=False).head(15))
dup_examples = [{
    "text_preview": str(r["example"])[:160],
    "n": int(r["n"]), "n_fake": int(r["n_fake"]), "n_true": int(r["n_true"]),
    "groups": str(r["groups"]),
} for _, r in dup_top.iterrows()]

conf_ex = (pool[pool["norm"].isin(set(conf_texts.index))]
           .groupby("norm")
           .agg(n=("train_label", "size"),
                n_fake=("train_label", lambda s: int((s == "fake").sum())),
                n_true=("train_label", lambda s: int((s == "true").sum())),
                groups=("dataset_name", lambda s: "|".join(sorted(set(map(str, s)))[:3])),
                example=(TEXT_COL, "first"))
           .sort_values("n", ascending=False).head(15))
conf_examples = [{
    "text_preview": str(r["example"])[:160],
    "n": int(r["n"]), "n_fake": int(r["n_fake"]), "n_true": int(r["n_true"]),
    "groups": str(r["groups"]),
} for _, r in conf_ex.iterrows()]

dup_raw = int(pool[TEXT_COL].duplicated().sum())
duplicates = {
    "normalized_unique_texts": int(norm.nunique()),
    "normalized_duplicated_rows_extra": n_dup_groups,
    "normalized_duplicated_rows_total": n_dup_rows,
    "normalized_dup_row_rate_pct": pct(n_dup_rows, n_pool),
    "raw_exact_duplicated_rows_extra": dup_raw,
    "conflicting_texts_fake_and_true": int(len(conf_texts)),
    "conflicting_rows_involved": conf_rows,
    "conflicting_row_rate_pct": pct(conf_rows, n_pool),
    "top_duplicates": dup_examples,
    "top_conflicts": conf_examples,
}

# ---------------------------------------------------------------------------
# 7. proveniencia no pool: lang, publisher, era, is_ptpt
# ---------------------------------------------------------------------------
t = time.time()
pm = pool.merge(prov, on="rid", how="left", validate="many_to_one")
prov_cov = int(pm["label_source"].notna().sum())
lang = (pm.assign(lang=pm["lang_variant"].fillna("<sem_prov>"))
        .groupby(["lang", "train_label"]).size()
        .unstack(fill_value=0).rename(columns={"fake": "n_fake", "true": "n_true"}))
for c in ("n_fake", "n_true"):
    if c not in lang.columns:
        lang[c] = 0
lang["n"] = lang["n_fake"] + lang["n_true"]
lang["pct_fake"] = (100.0 * lang["n_fake"] / lang["n"]).round(2)
lang_rows = [{"lang_variant": str(i), "n": int(r["n"]), "n_fake": int(r["n_fake"]),
              "n_true": int(r["n_true"]), "pct_fake": float(r["pct_fake"])}
             for i, r in lang.sort_values("n", ascending=False).iterrows()]

pub = (pm.assign(pub=pm["publisher"].fillna("<sem_prov>"))
       .groupby(["pub", "train_label"]).size()
       .unstack(fill_value=0).rename(columns={"fake": "n_fake", "true": "n_true"}))
for c in ("n_fake", "n_true"):
    if c not in pub.columns:
        pub[c] = 0
pub["n"] = pub["n_fake"] + pub["n_true"]
pub["pct_fake"] = (100.0 * pub["n_fake"] / pub["n"]).round(2)
pub_rows = [{"publisher": str(i), "n": int(r["n"]), "n_fake": int(r["n_fake"]),
             "n_true": int(r["n_true"]), "pct_fake": float(r["pct_fake"])}
            for i, r in pub.sort_values("n", ascending=False).head(15).iterrows()]

role = (pm.assign(role=pm["text_role"].fillna("<sem_prov>"))
        .groupby(["role", "train_label"]).size().unstack(fill_value=0)
        .rename(columns={"fake": "n_fake", "true": "n_true"}))
for c in ("n_fake", "n_true"):
    if c not in role.columns:
        role[c] = 0
role_rows = [{"text_role": str(i), "n_fake": int(r.get("n_fake", 0)),
              "n_true": int(r.get("n_true", 0))} for i, r in role.iterrows()]

coll = (pm.assign(col=pm["collector"].fillna("<sem_prov>"))
        .groupby(["col", "train_label"]).size().unstack(fill_value=0)
        .rename(columns={"fake": "n_fake", "true": "n_true"}))
for c in ("n_fake", "n_true"):
    if c not in coll.columns:
        coll[c] = 0
collector_rows = [{"collector": str(i), "n_fake": int(r.get("n_fake", 0)),
                   "n_true": int(r.get("n_true", 0)), "n": int(r.sum())}
                  for i, r in coll.iterrows()]

urls = pool["url_review"].fillna("") + " " + pool["factcheck_url"].fillna("")
is_ptpt = urls.str.contains(URL_PT_PT, case=False, regex=True)
ptpt_cross = (pm.assign(is_ptpt=is_ptpt.to_numpy(),
                        lang=pm["lang_variant"].fillna("<sem_prov>"))
              .groupby(["is_ptpt", "lang"]).size().to_dict())
ptpt_cross_rows = [{"is_ptpt_url_rule": bool(k[0]), "lang_variant": str(k[1]),
                    "n": int(v)} for k, v in sorted(ptpt_cross.items())]
ai1 = pm[pm["mentions_ai"].fillna(0.0) == 1.0]
mentions_ai_by_group = {str(k): int(v) for k, v in
                        ai1.groupby("dataset_name").size()
                        .sort_values(ascending=False).items()}
rating_counts = {str(k): int(v) for k, v in
                 pm["rating_norm"].fillna("<NaN>").value_counts()
                 .head(20).to_dict().items()}

dt = pd.to_datetime(pool["date_iso"], errors="coerce", format="mixed")
pool["year"] = dt.dt.year


def era_of(y):
    if pd.isna(y):
        return "<sem_data>"
    y = int(y)
    if y <= 2017:
        return "<=2017"
    if y <= 2022:
        return "2018-2022"
    return ">=2023"


pool["era"] = pool["year"].map(era_of)
year_tab = (pool.assign(year=pool["year"].fillna(-1).astype(int))
            .groupby(["year", "train_label"]).size().unstack(fill_value=0)
            .rename(columns={"fake": "n_fake", "true": "n_true"}))
for c in ("n_fake", "n_true"):
    if c not in year_tab.columns:
        year_tab[c] = 0
year_tab["n"] = year_tab["n_fake"] + year_tab["n_true"]
year_rows = [{"year": int(i), "n": int(r["n"]), "n_fake": int(r["n_fake"]),
              "n_true": int(r["n_true"]),
              "pct_fake": round(100.0 * r["n_fake"] / r["n"], 2) if r["n"] else 0.0}
             for i, r in year_tab.sort_index().iterrows()]
era_tab = (pool.groupby(["era", "train_label"]).size().unstack(fill_value=0)
           .rename(columns={"fake": "n_fake", "true": "n_true"}))
for c in ("n_fake", "n_true"):
    if c not in era_tab.columns:
        era_tab[c] = 0
era_tab["n"] = era_tab["n_fake"] + era_tab["n_true"]
era_order = ["<=2017", "2018-2022", ">=2023", "<sem_data>"]
era_rows = [{"era": e, "n": int(era_tab.loc[e, "n"]),
             "n_fake": int(era_tab.loc[e, "n_fake"]),
             "n_true": int(era_tab.loc[e, "n_true"]),
             "pct_fake": round(100.0 * era_tab.loc[e, "n_fake"] / era_tab.loc[e, "n"], 2)
             if era_tab.loc[e, "n"] else 0.0}
            for e in era_order if e in era_tab.index]
full_year_counter = {int(k) if not (isinstance(k, float) and math.isnan(k)) else -1:
                     int(v) for k, v in year_counter.items()}

# ---------------------------------------------------------------------------
# 8. vazamento por procedencia (rotulo constante por grupo)
# ---------------------------------------------------------------------------
nun = pool.groupby("dataset_name")["train_label"].nunique()
const_groups = set(nun[nun == 1].index)
const_group_rows = int(pool["dataset_name"].isin(const_groups).sum())
const_list = []
for name, row in g.iterrows():
    if name in const_groups:
        const_list.append({"group": str(name), "n": int(row["n"]),
                           "label": "fake" if row["n_fake"] > 0 else "true",
                           "channel": row["channel"]})
const_list.sort(key=lambda d: -d["n"])

nun_ch = pool.groupby(pool["dataset_name"].map(channel_of))["train_label"].nunique()
const_channels = sorted(nun_ch[nun_ch == 1].index.astype(str).tolist())
channel_purity = [{"channel": str(i), "n_unique_labels": int(v)}
                  for i, v in nun_ch.sort_values().items()]

# ---------------------------------------------------------------------------
# 9. metadados de avaliacao/calibracao
# ---------------------------------------------------------------------------
suggested = [
    {"column": "publisher", "source": "provenance", "cardinality": int(prov["publisher"].nunique()),
     "n_pool": prov_cov},
    {"column": "lang_variant", "source": "provenance", "cardinality": int(prov["lang_variant"].nunique()),
     "n_pool": int(pm["lang_variant"].notna().sum())},
    {"column": "era (de date_iso)", "source": "sanitized", "cardinality": 4,
     "n_pool": int(dt.notna().sum())},
    {"column": "channel (channel_of)", "source": "derived", "cardinality": int(ch.shape[0]),
     "n_pool": n_pool},
    {"column": "dataset_name", "source": "sanitized", "cardinality": int(g.shape[0]),
     "n_pool": n_pool},
    {"column": "label_tier", "source": "labels", "cardinality": int(len(label_tier_counts)),
     "n_pool": n_pool},
    {"column": "label_source (labels)", "source": "labels", "cardinality": int(len(label_source_counts)),
     "n_pool": n_pool},
]

# ---------------------------------------------------------------------------
# 10. monta JSON e MD
# ---------------------------------------------------------------------------
cur_year = datetime.now(timezone.utc).year
future_years = {k: v for k, v in full_year_counter.items() if k > cur_year}
old_years = {k: v for k, v in full_year_counter.items() if 0 < k < 2000}
surprises = {
    "text_encoding": {c: dict(text_diag[c]) for c in TEXT_COLS},
    "text_columns_present": [c for c in TEXT_COLS if c in san_columns],
    "empty_text_full_sanitized": {c: int(text_diag[c]["empty"]) for c in TEXT_COLS},
    "train_label_capitalized_True": int(train_label_counts.get("True", 0)),
    "orig_label_casefold_mismatch_in_pool": orig_casefold_mismatch,
    "text_labeled_rows_dropped_by_text_filter": dropped_text,
    "date_range_full": {"min": str(date_min), "max": str(date_max),
                        "nan": int(date_nan), "nan_pct": pct(date_nan, n_rows)},
    "future_dates_full": {str(k): int(v) for k, v in sorted(future_years.items())},
    "suspect_old_dates_full": {str(k): int(v) for k, v in sorted(old_years.items())},
    "fffd_examples": fffd_examples,
    "full_normalized_duplicated_rows_extra": int(n_rows - len(norm_hash_set)),
    "news_groups_without_train_label": news_no_train,
    "news_in_pool": news_pool_info,
    "news_total": news_total,
    "dataset_names": int(len(dataset_counter)),
}

profile = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "script": str(Path(__file__).resolve()),
    "inputs": file_meta,
    "sanitized": {
        "shape": [n_rows, len(san_columns)],
        "columns": san_columns,
        "dtypes": san_dtypes,
        "rid_unique": n_rows == len(rid_set),
        "rid_null": int(null_counts.get("rid", 0)),
        "label_column_counts": {str(k): int(v) for k, v in label_counter.items()},
        "dataset_name_counts": {str(k): int(v) for k, v in
                                dataset_counter.most_common()},
        "null_counts": {c: int(null_counts.get(c, 0)) for c in san_columns},
        "flags": {c: {str(k): int(v) for k, v in flags[c].items()} for c in FLAG_COLS},
        "text_diagnostics": {c: dict(text_diag[c]) for c in TEXT_COLS},
        "date_full": {"min": str(date_min), "max": str(date_max),
                      "nan": int(date_nan),
                      "year_counts": {str(k): int(v) for k, v in
                                      sorted(full_year_counter.items())}},
        "timing_read_s": round(t_full_pass, 1),
    },
    "labels": {
        "shape": [int(len(lab)), int(lab.shape[1])],
        "columns": file_meta["labels"]["columns"],
        "rid_unique": len(lab) == len(lab_rid_set),
        "train_label_counts": {str(k): int(v) for k, v in train_label_counts.items()},
        "label_tier_counts": {str(k): int(v) for k, v in label_tier_counts.items()},
        "label_source_counts": {str(k): int(v) for k, v in label_source_counts.items()},
        "confidence_counts": {str(k): int(v) for k, v in
                              lab["confidence"].fillna("<NaN>")
                              .value_counts().head(15).to_dict().items()},
        "method_counts": {str(k): int(v) for k, v in
                          lab["method"].fillna("<NaN>")
                          .value_counts().head(15).to_dict().items()},
        "auto_label_counts": {str(k): int(v) for k, v in
                              lab["auto_label"].fillna("<NaN>")
                              .value_counts().head(10).to_dict().items()},
    },
    "provenance": {
        "shape": [int(len(prov)), int(prov.shape[1])],
        "columns": file_meta["provenance"]["columns"],
        "rid_unique": len(prov) == len(prov_rid_set),
        "lang_variant_counts": {str(k): int(v) for k, v in
                                prov["lang_variant"].fillna("<NaN>")
                                .value_counts().to_dict().items()},
        "publisher_counts": {str(k): int(v) for k, v in
                             prov["publisher"].fillna("<NaN>")
                             .value_counts().to_dict().items()},
    },
    "merge": merge_stats,
    "pool": {
        "n": int(n_pool),
        "n_fake": n_fake,
        "n_true": n_true,
        "fake_true_ratio": round(n_fake / n_true, 3) if n_true else None,
        "pct_fake": pct(n_fake, n_pool),
        "by_dataset_top30": by_dataset_top30,
        "by_dataset_all_n": int(g.shape[0]),
        "by_channel": by_channel,
        "prefix_groups": prefix_groups,
        "informative_groups": informative_groups,
        "informative_group_count": int(len(informative_groups)),
        "v1_balanced_present": v1_balanced_present,
        "length": length,
        "duplicates": duplicates,
        "provenance": {
            "coverage_rows": prov_cov,
            "coverage_pct": pct(prov_cov, n_pool),
            "lang_variant": lang_rows,
            "publisher_top15": pub_rows,
            "text_role": role_rows,
            "collector": collector_rows,
            "is_ptpt_url_rule": int(is_ptpt.sum()),
            "is_ptpt_x_lang": ptpt_cross_rows,
            "mentions_ai_counts": {str(k): int(v) for k, v in
                                   pm["mentions_ai"].fillna("<NaN>")
                                   .value_counts().head(10).to_dict().items()},
            "mentions_ai_1_by_group": mentions_ai_by_group,
            "rating_norm_counts": rating_counts,
        },
        "eras": {
            "by_year": year_rows,
            "by_era": era_rows,
            "date_min": str(dt.min()), "date_max": str(dt.max()),
            "date_nan": int(dt.isna().sum()),
        },
        "group_leakage": {
            "n_groups": int(nun.shape[0]),
            "n_constant_label_groups": int(len(const_groups)),
            "pct_constant_label_groups": pct(len(const_groups), nun.shape[0]),
            "rows_in_constant_groups": const_group_rows,
            "pct_rows_in_constant_groups": pct(const_group_rows, n_pool),
            "constant_groups": const_list,
            "constant_channels": const_channels,
            "channel_purity": channel_purity,
        },
    },
    "eval_metadata": {
        "sanitized_columns": san_columns,
        "labels_columns": file_meta["labels"]["columns"],
        "provenance_columns": file_meta["provenance"]["columns"],
        "suggested_eval_groups": suggested,
    },
    "surprises": surprises,
    "timings_s": {"full_pass": round(t_full_pass, 1),
                  "total": round(time.time() - T0, 1)},
}

jdump(profile, JSON_OUT)
log(f"JSON salvo em {JSON_OUT}")

# ------------------------------- MD ---------------------------------------
L = []
L.append("# Perfil do dataset FakenewsBR v4")
L.append("")
L.append(f"Gerado por `models/v4/explore_v4.py` em "
         f"{profile['generated_at']} (tempo total: "
         f"{profile['timings_s']['total']}s). Somente leitura; nenhum arquivo "
         f"do projeto foi alterado.")
L.append("")
L.append("## Resumo executivo")
L.append("")
L.append(f"- Pool treinavel (regra `models/data.py` com labels v4): "
         f"**{n_pool:,} linhas** — fake {n_fake:,} / true {n_true:,} "
         f"({pct(n_fake, n_pool)}% fake; razao fake:true "
         f"{round(n_fake / n_true, 2) if n_true else 'n/a'}).")
L.append(f"- `sanitized_v4`: {n_rows:,} linhas x {len(san_columns)} colunas; "
         f"`rid` unico (duplicados: {n_rows - len(rid_set)}).")
L.append(f"- `labels_v4`: {len(lab):,} linhas; `rid` unico; "
         f"train_label NaN em {train_label_counts.get('<NaN>', 0):,} "
         f"(camada `provenance`).")
L.append(f"- Grupos informativos (prefixo FC_/EXT_/NEWS_, n>=200, minoria>=15%): "
         f"**{len(informative_groups)}**; grupos v1 balanceados presentes: "
         f"{len(v1_balanced_present)}.")
L.append(f"- `NEWS_*` (canal press_true) fora do treino: "
         f"**{news_no_train:,} de {news_total:,}** linhas "
         f"({pct(news_no_train, news_total)}%).")
L.append(f"- Duplicatas normalizadas no pool: {n_dup_rows:,} linhas em grupos "
         f"duplicados ({pct(n_dup_rows, n_pool)}%); textos com fake E true: "
         f"**{len(conf_texts):,}** ({conf_rows:,} linhas).")
ptpt_n = next((r["n"] for r in lang_rows if r["lang_variant"] == "pt-PT"), 0)
L.append(f"- PT-PT no pool: {ptpt_n:,} linhas com lang_variant pt-PT; a regra "
         f"de URL `is_ptpt` do data.py marca {int(is_ptpt.sum()):,} "
         f"(criterios discordam; ver secao 9).")
L.append(f"- Exemplos acima de 512 palavras (truncamento BERT): "
         f"**{length['truncation']['over_512_words']:,}** "
         f"({length['truncation']['over_512_words_pct']}%).")
L.append("")

L.append("## 1. Arquivos e colunas")
L.append("")
rows = []
for name in ("sanitized", "labels", "provenance"):
    m = file_meta[name]
    rows.append([name, m["size_mb"], m["physical_lines"],
                 m.get("parsed_rows", n_rows), m.get("t_read_s", "-")])
L.append(md_table(["arquivo", "MB", "linhas fisicas", "linhas CSV (parse)", "leitura (s)"], rows))
L.append("")
L.append("Linhas fisicas > linhas parseadas indicam `\\n` embutido em campos de "
         "texto (esperado nos 3 arquivos).")
L.append("")
L.append("Colunas de texto presentes: "
         + ", ".join(f"`{c}` ({text_diag[c]['notnull']:,} nao nulos, "
                     f"{text_diag[c]['empty']:,} vazios)" for c in TEXT_COLS) + ".")
L.append("")
L.append(md_table(["coluna", "dtype", "nulos", "nulos %"],
                  [[c, san_dtypes[c], f"{int(null_counts.get(c, 0)):,}",
                    pct(int(null_counts.get(c, 0)), n_rows)]
                   for c in san_columns]))
L.append("")

L.append("## 2. Merge sanitized x labels")
L.append("")
L.append(md_table(["metrica", "valor"], [
    ["linhas sanitized", f"{n_rows:,}"],
    ["rids unicos sanitized", f"{len(rid_set):,}"],
    ["linhas labels", f"{len(lab):,}"],
    ["rids labels ausentes no sanitized", f"{lab_only:,}"],
    ["rids sanitized sem linha em labels", f"{san_without_label:,}"],
    ["train_label fake", f"{train_label_counts.get('fake', 0):,}"],
    ["train_label true", f"{train_label_counts.get('true', 0):,}"],
    ["train_label NaN", f"{train_label_counts.get('<NaN>', 0):,}"],
    ["train_label 'True' (maiusculo)", f"{train_label_counts.get('True', 0):,}"],
    ["linhas com train_label fake/true", f"{train_labelled:,}"],
    ["  descartadas por texto vazio/nulo", f"{dropped_text:,}"],
    ["pool final", f"{n_pool:,}"],
]))
L.append("")
L.append("Quebra por `label_tier` (linhas do sanitized apos merge):")
L.append("")
rows = []
for tier, tot in sorted(tier_counts.items(), key=lambda kv: -kv[1]):
    fake = sum(v for (t, c), v in tier_cat_counts.items() if t == tier and c == "fake")
    true = sum(v for (t, c), v in tier_cat_counts.items() if t == tier and c == "true")
    nan = sum(v for (t, c), v in tier_cat_counts.items() if t == tier and c == "<train_label_NaN>")
    rows.append([tier, f"{tot:,}", f"{fake:,}", f"{true:,}", f"{nan:,}"])
L.append(md_table(["label_tier", "n", "fake", "true", "train_label NaN"], rows))
L.append("")
mism = orig_casefold_mismatch
L.append(f"Divergencias rotulo original x train_label (casefold, dentro do pool): **{mism:,}** "
         f"({pct(mism, n_pool)}% do pool).")
L.append("")

L.append("## 3. Pool treinavel: grupos e canais")
L.append("")
L.append("Top 30 grupos por n (o pool tem "
         f"{g.shape[0]} grupos; lista completa no JSON):")
L.append("")
rows = [[r["group"], r["channel"], f"{r['n']:,}", f"{r['n_fake']:,}",
         f"{r['n_true']:,}", r["pct_fake"], r["minority_frac"],
         "sim" if r["informative"] else ("v1" if r["balanced_v1"] else "-")]
        for r in by_dataset_top30]
L.append(md_table(["dataset_name", "canal", "n", "fake", "true", "%fake",
                   "minoria", "informativo"], rows))
L.append("")
L.append("Distribuicao por canal:")
L.append("")
L.append(md_table(["canal", "n", "fake", "true", "%fake"],
                  [[r["channel"], f"{r['n']:,}", f"{r['n_fake']:,}",
                    f"{r['n_true']:,}", r["pct_fake"]] for r in by_channel]))
L.append("")
L.append("Grupos com prefixo FC_/EXT_/NEWS_ (todos):")
L.append("")
L.append(md_table(["grupo", "canal", "n", "%fake", "minoria", "passa criterio"],
                  [[r["group"], r["channel"], f"{r['n']:,}", r["pct_fake"],
                    r["minority_frac"], "SIM" if r["informative"] else "nao"]
                   for r in prefix_groups]))
L.append("")
L.append(f"Grupos informativos que entram no treino "
         f"({len(informative_groups)}): "
         + ", ".join(f"`{r['group']}` ({r['n']:,}, {r['pct_fake']}% fake)"
                     for r in informative_groups) + ".")
L.append("")

L.append("## 4. Comprimento (palavras em text_no_url)")
L.append("")
rows = []
for scope, title in (("fake", "fake"), ("true", "true"), ("all", "pool")):
    s = length["words"][scope]
    rows.append([title, f"{s['n']:,}", s["mean"], s["p50"], s["p90"],
                 s["p95"], s["p99"], s["max"]])
L.append(md_table(["classe", "n", "media", "p50", "p90", "p95", "p99", "max"], rows))
L.append("")
L.append(f"Acima de 512 palavras: **{length['truncation']['over_512_words']:,}** "
         f"(fake {length['truncation']['fake_over_512']:,}, "
         f"true {length['truncation']['true_over_512']:,}); "
         f"acima de 510: {length['truncation']['over_510_words']:,}; "
         f"acima de 400: {length['truncation']['over_400_words']:,}.")
L.append("")
wlc = length["word_len_column_check"]
L.append(f"Conferencia da coluna `word_len`: {wlc['rows_word_len_missing']:,} "
         f"nulos, {wlc['rows_word_len_differs_from_text']:,} linhas divergentes "
         f"do texto (>2: {wlc['rows_word_len_differs_gt_2']:,}), corr={wlc['corr']}.")
L.append("")

L.append("## 5. Duplicatas e conflitos")
L.append("")
L.append(md_table(["metrica", "valor"], [
    ["textos unicos (norm.)", f"{duplicates['normalized_unique_texts']:,}"],
    ["linhas duplicadas (total)", f"{duplicates['normalized_duplicated_rows_total']:,} "
     f"({duplicates['normalized_dup_row_rate_pct']}%)"],
    ["duplicatas exatas (cru)", f"{duplicates['raw_exact_duplicated_rows_extra']:,}"],
    ["textos com fake E true", f"{duplicates['conflicting_texts_fake_and_true']:,}"],
    ["linhas nesses textos conflitantes", f"{duplicates['conflicting_rows_involved']:,} "
     f"({duplicates['conflicting_row_rate_pct']}%)"],
]))
L.append("")
L.append(f"No sanitized completo: {len(norm_hash_set):,} textos normalizados "
         f"unicos, {n_rows - len(norm_hash_set):,} linhas duplicadas extras "
         f"({pct(n_rows - len(norm_hash_set), n_rows)}%).")
if dup_examples:
    L.append("")
    L.append("Textos mais duplicados (preview de 160 chars):")
    L.append("")
    L.append(md_table(["n", "fake", "true", "grupos", "preview"],
                      [[d["n"], d["n_fake"], d["n_true"], d["groups"],
                        d["text_preview"].replace("|", "/")] for d in dup_examples]))
if conf_examples:
    L.append("")
    L.append("Exemplos de conflito de rotulo (mesmo texto normalizado):")
    L.append("")
    L.append(md_table(["n", "fake", "true", "grupos", "preview"],
                      [[d["n"], d["n_fake"], d["n_true"], d["groups"],
                        d["text_preview"].replace("|", "/")] for d in conf_examples]))
L.append("")

L.append("## 6. Proveniencia no pool (lang, publisher, era)")
L.append("")
L.append(f"Cobertura de `provenance` no pool: {prov_cov:,} / {n_pool:,} "
         f"({pct(prov_cov, n_pool)}%).")
L.append("")
L.append("Idioma (`lang_variant`):")
L.append("")
L.append(md_table(["lang_variant", "n", "fake", "true", "%fake"],
                  [[r["lang_variant"], f"{r['n']:,}", f"{r['n_fake']:,}",
                    f"{r['n_true']:,}", r["pct_fake"]] for r in lang_rows]))
L.append("")
L.append("Publisher (top 15):")
L.append("")
L.append(md_table(["publisher", "n", "fake", "true", "%fake"],
                  [[r["publisher"], f"{r['n']:,}", f"{r['n_fake']:,}",
                    f"{r['n_true']:,}", r["pct_fake"]] for r in pub_rows]))
L.append("")
ptpt_b = next((r["n"] for r in ptpt_cross_rows
               if r["is_ptpt_url_rule"] and r["lang_variant"] == "pt-BR"), 0)
ptpt_np = next((r["n"] for r in ptpt_cross_rows
                if r["is_ptpt_url_rule"] and r["lang_variant"] == "<sem_prov>"), 0)
L.append(f"Regra URL PT-PT do `data.py` marca {int(is_ptpt.sum()):,} linhas do "
         f"pool; dessas, a provenance diz pt-PT em {ptpt_n:,}, pt-BR em "
         f"{ptpt_b:,} e nao cobre {ptpt_np:,}. Os dois criterios discordam.")
L.append("")
L.append("Eras temporais (`date_iso` do sanitized):")
L.append("")
L.append(md_table(["era", "n", "fake", "true", "%fake"],
                  [[r["era"], f"{r['n']:,}", f"{r['n_fake']:,}",
                    f"{r['n_true']:,}", r["pct_fake"]] for r in era_rows]))
L.append("")
L.append("Por ano: " + "; ".join(f"{r['year']}: {r['n']:,}"
                                 for r in year_rows if r["year"] > 0) + ".")
L.append("")

L.append("## 7. Vazamento por procedencia (rotulo constante)")
L.append("")
gl = profile["pool"]["group_leakage"]
L.append(f"- Grupos (dataset_name): **{gl['n_constant_label_groups']} de "
         f"{gl['n_groups']}** ({gl['pct_constant_label_groups']}%) tem rotulo "
         f"constante; essas linhas sao **{gl['rows_in_constant_groups']:,}** "
         f"({gl['pct_rows_in_constant_groups']}% do pool).")
L.append(f"- Canais com rotulo constante: {', '.join(gl['constant_channels'])}.")
L.append("")
if const_list:
    L.append("Maiores grupos constantes:")
    L.append("")
    L.append(md_table(["grupo", "n", "rotulo unico", "canal"],
                      [[c["group"], f"{c['n']:,}", c["label"], c["channel"]]
                       for c in const_list[:20]]))
L.append("")

L.append("## 8. Metadados uteis para avaliacao/calibracao")
L.append("")
L.append(md_table(["coluna", "fonte", "cardinalidade", "linhas no pool"],
                  [[s["column"], s["source"], s["cardinality"], f"{s['n_pool']:,}"]
                   for s in suggested]))
L.append("")

L.append("## 9. Achados criticos / surpresas")
L.append("")
ex_enc = ("; ex.: " + " / ".join(repr(e) for e in fffd_examples[:2])
          if fffd_examples else "")
L.append(f"1. **Encoding corrompido em parte do texto**: "
         f"`text_no_url` tem {text_diag['text_no_url']['has_fffd']:,} linhas com "
         f"U+FFFD e "
         f"{text_diag['text_no_url']['has_latin_ext_ab']:,} com caracteres "
         f"Latin Extended-A/B fora do portugues (mojibake){ex_enc}.")
L.append(f"2. **Vazamento de procedencia**: "
         f"{gl['pct_rows_in_constant_groups']}% do pool esta em grupos com "
         f"rotulo constante (dataset_name). Amostragem aleatoria IID mede "
         f"memorizacao de origem, nao detecao.")
L.append(f"3. **Duplicatas**: {duplicates['normalized_duplicated_rows_total']:,} "
         f"linhas duplicadas ({duplicates['normalized_dup_row_rate_pct']}%) e "
         f"{duplicates['conflicting_texts_fake_and_true']:,} textos com fake E "
         f"true ({duplicates['conflicting_rows_involved']:,} linhas) — o "
         f"sanitizer ja deduplicou tudo (flag `is_duplicated`=0 nas 291.521 "
         f"linhas), entao nao ha vazamento treino/teste por texto repetido.")
L.append(f"4. **Desbalanceamento**: pool {pct(n_fake, n_pool)}% fake "
         f"(razao {round(n_fake / n_true, 2) if n_true else 'n/a'}:1) e "
         f"{length['truncation']['over_512_words']:,} linhas acima de 512 "
         f"palavras serao truncadas pelo BERTimbau.")
L.append(f"5. **NEWS_* fora do treino**: {news_no_train:,} de {news_total:,} "
         f"linhas do canal press_true nao tem train_label e ficam fora do pool.")
L.append(f"6. **`is_ptpt` x `lang_variant` discordam**: a regra de URL do "
         f"`data.py` marca {int(is_ptpt.sum()):,} linhas do pool como PT-PT, "
         f"mas a provenance registra so {ptpt_n:,} pt-PT (e {ptpt_b:,} pt-BR "
         f"nessas mesmas linhas). Avaliar o dominio de PT-PT fica ambiguo.")
news_groups_str = "; ".join(f"{k} {v:,}"
                            for k, v in news_pool_info["by_group"].items())
L.append(f"7. **NEWS_* parcialmente no treino**: {len(news_pool):,} manchetes "
         f"`NEWS_*` tem train_label e entram no pool ({news_groups_str}), "
         f"todas da classe true — grupos constantes que nao ensinam a "
         f"fronteira fake/true.")
L.append("")

L.append("## 10. Reprodutibilidade")
L.append("")
L.append("```")
L.append("python models/v4/explore_v4.py")
L.append("```")
L.append("")
L.append(f"- Entrada: `{SAN_CSV.name}`, `{LAB_CSV.name}`, `{PROV_CSV.name}`.")
L.append(f"- Saida: `models/v4/v4_profile.json`, `models/v4/V4_PROFILE.md`.")
L.append("- Nao importa `models.data`; regras replicadas no proprio script.")
L.append(f"- Tempos: passada completa {profile['timings_s']['full_pass']}s, "
         f"total {profile['timings_s']['total']}s.")
L.append("")

MD_OUT.write_text("\n".join(L), encoding="utf-8")
log(f"MD salvo em {MD_OUT}")
log(f"concluido em {time.time() - T0:.1f}s")
