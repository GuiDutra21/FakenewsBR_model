#!/usr/bin/env python
"""prepare_v6.py -- CSVs v6 -> parquet enxuto + splits 3 vias + estatisticas.

Roda UMA vez, no repo local (depende de `data.py`, copia de `legacy/models/data.py`,
para reproduzir as regras de `train_label`, grupo, canal, `is_balanced_group` e
`_safe_strat`). As saidas ficam fora do Git (`.gitignore` ignora `processed/`).

Adaptacao do pipeline v4 (`legacy/models/v4/prepare_v4.py`, testado) para o POOL
COMPLETO da v6 (91.080 linhas rotuladas, 36 grupos). Deltas v6:

  - splits: `full_iid` (pool completo 70/10/5/15, default de treino),
    `ood_wa` (teste = canal whatsapp 6.381; restante 85/10/5) e
    `bal_iid` (informativos `is_balanced_group`, 70/10/5/15, referencia);
  - duplicatas exatas normalizadas E quase-duplicatas (Jaccard >= 0,8) sao
    agrupadas em CLUSTERS (MinHash+LSH em numpy, verificacao exata, union-find)
    e cada cluster inteiro cai em um unico lado do split; a atribuicao usa um
    greedy que preserva as proporcoes por rotulo e por grupo (alvo ~1 p.p.);
  - `near_dup_cluster` persistido no `v6_splits.parquet` e nos stats;
  - varredura EXATA (indice invertido de 5-gramas) confirma 0 pares
    Jaccard >= 0,8 cruzando treino<->teste e treino<->val nos splits; se algum
    par escapar do LSH, os clusters sao unidos e o split refeito;
  - stats de pesos DFR cell/clip 25 por split (prior efetivo, percentis);
  - token stats por split de treino.

Contrato v4 preservado: mesmas 20 colunas de `v6_pool.parquet` (inclui
`label_tier`), `v6_splits.parquet` com `rid` + 3 colunas de split + 
`near_dup_cluster`, `prepare_stats.json` com sha256 e validacoes que abortam
com codigo 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data as D  # noqa: E402  (copia de legacy/models/data.py; precisa do sys.path acima)

TEXT_COL = D.TEXT_COL
SPLIT_COLS = ("split_full_iid", "split_ood_wa", "split_bal_iid")
SIDES_4 = ("train", "val_sel", "val_calib", "test")
SIDES_3 = ("train", "val_sel", "val_calib")

# Invariantes medidos no pool v6 (tolerancia 0; ver docs/v6/BUILD_NOTES.md).
EXPECTED_SANITIZED = 297_672
EXPECTED_POOL = 91_080
EXPECTED_FAKE = 66_772
EXPECTED_TRUE = 24_308
EXPECTED_INFO = 36_896
EXPECTED_GROUPS = 36
EXPECTED_CONST_GROUPS = 11
EXPECTED_CONST_LINES = 39_373
EXPECTED_UFFFD = 4
EXPECTED_SANITIZED_TIERS = {
    "provenance": 206_592, "checker": 44_347, "v1": 39_466,
    "checker_match": 4_166, "llm_local": 3_092, "corroborated": 9,
}
EXPECTED_POOL_TIERS = {
    "checker": 44_347, "v1": 39_466, "checker_match": 4_166,
    "llm_local": 3_092, "corroborated": 9,
}
EXPECTED_INFO_GROUPS = {
    "FC_POLIGRAFO": 10_831, "Fake.br": 7_160, "EXT_LIARBR": 6_996,
    "FakeWhatsApp.BR_2018": 6_381, "EXT_AVERITECBR": 2_925, "COVID19.BR": 1_931,
    "COVID19.BR_raw": 373, "LLM4BR_300": 299,
}
EXPECTED_CONST_GROUPS_DETAIL = {
    "fakes": (20_347, "fake"), "FC_BOATOS": (10_772, "fake"),
    "true": (2_710, "true"), "FC_BOATOS_VIRAL": (2_360, "fake"),
    "NEWS_PODER360": (1_338, "true"), "NEWS_BRASILDEFATO": (909, "true"),
    "NEWS_OECO": (468, "true"), "NEWS_ECO": (453, "true"),
    "FC_ALETHEIA": (9, "fake"), "MuMiN-PT_raw": (4, "fake"),
    "FC_NEXO": (3, "fake"),
}
# Duplicatas exatas normalizadas medidas no pool (a v6 tem 473 copias extras).
EXPECTED_DUP_GROUPS = 427
EXPECTED_DUP_ROWS = 900
EXPECTED_DUP_EXTRA = 473

# Near-duplicatas: 5-gramas de palavra normalizados, MinHash K=64, LSH 16x4,
# verificacao exata Jaccard; anel de referencia <0,7 no stats da varredura.
MINHASH_K = 64
LSH_BANDS = 16
LSH_ROWS = 4
NEAR_DUP_J = 0.8
NEAR_DUP_REF = 0.7
_MINHASH_PRIME = (1 << 31) - 1
_MINHASH_RNG = np.random.default_rng(20260912)
_MINHASH_A = _MINHASH_RNG.integers(1, _MINHASH_PRIME, size=MINHASH_K,
                                   dtype=np.uint64)
_MINHASH_B = _MINHASH_RNG.integers(0, _MINHASH_PRIME, size=MINHASH_K,
                                   dtype=np.uint64)

# Latin Extended-A/B: regex com caracteres literais (pyarrow nao aceita \u).
LATIN_EXT_RE = "[" + chr(0x0100) + "-" + chr(0x024F) + "]"

# Normalizacao identica a `investigation/expansion/dedup.py::normalize_text`
# (a mesma que o quality report da v6 usou para medir as 473 duplicatas).
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Minuscula, sem acento, apenas [a-z0-9 ] colapsado (copia do dedup v6)."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _NON_ALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


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


# ------------------------------------------------------- near-dup clustering
def shingles(text: str, n: int = 5) -> set:
    """5-gramas de palavra sobre `normalize_text` (near-dup normalizado)."""
    ws = normalize_text(text).split()
    if len(ws) < n:
        return set()
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def shingle_hashes_from_words(ws: list, n: int = 5) -> np.ndarray:
    """Hashes crc32 (estaveis) dos 5-gramas; unicos e ordenados (uint64)."""
    if len(ws) < n:
        return np.zeros(0, dtype=np.uint64)
    arr = np.fromiter(
        (zlib.crc32(" ".join(ws[i:i + n]).encode("utf-8"))
         for i in range(len(ws) - n + 1)),
        dtype=np.uint64, count=len(ws) - n + 1)
    return np.unique(arr)


def minhash_signature(hashes: np.ndarray) -> np.ndarray:
    """Assinatura MinHash de K permutacoes (a*h+b mod p, hashes uint32)."""
    if len(hashes) == 0:
        return np.full(MINHASH_K, _MINHASH_PRIME, dtype=np.uint64)
    h = hashes.astype(np.uint64)
    prod = (_MINHASH_A[:, None] * h[None, :] + _MINHASH_B[:, None])
    return (prod % np.uint64(_MINHASH_PRIME)).min(axis=1)


def lsh_candidates(signatures: np.ndarray, bands: int = LSH_BANDS,
                   rows: int = LSH_ROWS) -> set:
    """Pares candidatos por bandas LSH (16x4 por default)."""
    n = signatures.shape[0]
    pairs: set = set()
    for b in range(bands):
        block = signatures[:, b * rows:(b + 1) * rows]
        key = block[:, 0].copy()
        for r in range(1, rows):
            key = key * np.uint64(1_000_003) ^ block[:, r]
        buckets: dict = {}
        for i, k in enumerate(key.tolist()):
            buckets.setdefault(k, []).append(i)
        for idxs in buckets.values():
            m = len(idxs)
            if m < 2:
                continue
            for x in range(m):
                for y in range(x + 1, m):
                    a, c = idxs[x], idxs[y]
                    pairs.add((a, c) if a < c else (c, a))
    return pairs


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def jaccard_exact(a: np.ndarray, b: np.ndarray) -> tuple:
    inter = int(np.intersect1d(a, b, assume_unique=True).size)
    union = len(a) + len(b) - inter
    return inter, (inter / union if union else 0.0)


def build_near_dup_clusters(pool: pd.DataFrame, seed: int = 42) -> tuple:
    """Clusters de duplicata exata + quase-duplicata (Jaccard >= 0.8).

    Etapas (por variante de shingle): shingles -> MinHash K=64 -> LSH 16x4 ->
    verificacao exata dos candidatos -> uniao via union-find. Roda DUAS
    variantes de 5-gramas sobre o mesmo union-find:

      - `norm`: sobre `normalize_text` (NFKD sem acento, [a-z0-9 ], espacos
        colapsados) — definicao principal do V2;
      - `raw`: palavras cruas (so split) — fecha a varredura historica do
        revisor, que usava 5-gramas exatos.

    A uniao cobre as duas definicoes de quase-duplicata. Tambem une os grupos
    de texto exato (`text_key`). Devolve (cluster_ids, hashes_por_variante,
    stats).
    """
    t0 = time.time()
    n = len(pool)
    texts = pool[TEXT_COL].astype(str).tolist()
    uf = UnionFind(n)
    n_exact_groups = 0
    for _, idxs in pool.groupby("text_key").indices.items():
        if len(idxs) > 1:
            n_exact_groups += 1
            for j in idxs[1:]:
                uf.union(int(idxs[0]), int(j))

    words_by_variant = {
        "norm": [normalize_text(t).split() for t in texts],
        "raw": [str(t).split() for t in texts],
    }
    hashes_by_variant: dict = {}
    variant_stats: dict = {}
    for name, words in words_by_variant.items():
        tv = time.time()
        hashes = [shingle_hashes_from_words(ws) for ws in words]
        n_single = int(sum(1 for h in hashes if len(h) < 2))
        sig = np.empty((n, MINHASH_K), dtype=np.uint64)
        for i, h in enumerate(hashes):
            sig[i] = minhash_signature(h)
        t_sig = time.time()
        candidates = lsh_candidates(sig)
        # docs com <2 shingles nao formam par Jaccard >= 0.8 com outro texto
        # (so duplicata exata, ja tratada por text_key).
        candidates = {(i, j) for (i, j) in candidates
                      if len(hashes[i]) >= 2 and len(hashes[j]) >= 2}
        t_cand = time.time()
        n_verified = 0
        for i, j in candidates:
            inter, jac = jaccard_exact(hashes[i], hashes[j])
            if inter >= 2 and jac >= NEAR_DUP_J:
                if uf.union(i, j):
                    n_verified += 1
        t_ver = time.time()
        hashes_by_variant[name] = hashes
        variant_stats[name] = {
            "shingle": ("5-gramas de palavra sobre normalize_text"
                        if name == "norm" else "5-gramas de palavra cruas (split)"),
            "docs_com_menos_de_2_shingles": n_single,
            "n_candidate_pairs": int(len(candidates)),
            "n_pairs_verified_ge_0.8": int(n_verified),
            "seconds": {
                "minhash": round(t_sig - tv, 1),
                "lsh_candidates": round(t_cand - t_sig, 1),
                "exact_verify": round(t_ver - t_cand, 1),
            },
        }

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    _, cluster = np.unique(roots, return_inverse=True)
    cluster = cluster.astype(np.int32)
    sizes = np.bincount(cluster)
    multi = sizes > 1
    hist = Counter(int(s) for s in sizes[sizes > 1])
    stats = {
        "method": ("MinHash+LSH numpy + verificacao exata Jaccard>=%.2f nas "
                   "variantes norm e raw; uniao com grupos de texto exato "
                   "(dedup.normalize_text)" % NEAR_DUP_J),
        "k": MINHASH_K, "bands": LSH_BANDS, "rows": LSH_ROWS,
        "variants": variant_stats,
        "n_exact_dup_groups": int(n_exact_groups),
        "n_clusters": int(len(sizes)),
        "n_multi_clusters": int(multi.sum()),
        "max_cluster_size": int(sizes.max()) if len(sizes) else 0,
        "n_rows_in_multi_clusters": int(sizes[multi].sum()),
        "multi_cluster_size_hist": {str(k): int(v) for k, v in sorted(hist.items())},
        "seconds": {"total": round(time.time() - t0, 1)},
    }
    return cluster, hashes_by_variant, stats


def exhaustive_cross_scan(hashes_a: list, hashes_b: list,
                          high: float = NEAR_DUP_J,
                          ref: float = NEAR_DUP_REF,
                          label: str = "") -> dict:
    """Varredura completa treino x teste/val (exata para Jaccard >= `high`).

    Indexa o lado B (menor) por shingle e acumula interseccoes para cada doc de
    A. Todo par com Jaccard >= `high` tem interseccao >= 2 (para >= 3 shingles;
    com < 3 shingles o par e duplicata exata e cai no mesmo cluster), entao a
    varredura e completa para o criterio. Devolve contagens e os pares >= high
    (indices locais) para o loop de uniao.
    """
    t0 = time.time()
    inv: dict = defaultdict(list)
    for j, h in enumerate(hashes_b):
        for g in h.tolist():
            inv[g].append(j)
    n_pairs, n_high, n_ref, n_mid, max_j = 0, 0, 0, 0, 0.0
    violations = []
    for i, h in enumerate(hashes_a):
        if len(h) < 2:
            continue
        hits: dict = {}
        for g in h.tolist():
            for j in inv.get(g, ()):
                hits[j] = hits.get(j, 0) + 1
        la = len(h)
        for j, inter in hits.items():
            if inter < 2:
                continue
            n_pairs += 1
            union = la + len(hashes_b[j]) - inter
            jac = inter / union if union else 0.0
            if jac > max_j:
                max_j = jac
            if jac >= high:
                n_high += 1
                violations.append((i, j))
            elif jac >= ref:
                n_ref += 1
            elif jac >= 0.5:
                n_mid += 1
    return {
        "comparison": label,
        "n_a": len(hashes_a), "n_b": len(hashes_b),
        "n_pares_com_intersecao": int(n_pairs),
        "max_jaccard": round(float(max_j), 6),
        "n_ge_0.8": int(n_high),
        "n_0.7_0.8": int(n_ref),
        "n_0.5_0.7": int(n_mid),
        "seconds": round(time.time() - t0, 1),
        "violations": violations,
    }


# ------------------------------------------------------- cluster split greed
def _cluster_units(cluster: np.ndarray, n_clusters: int, size_w: np.ndarray,
                   fake_w: np.ndarray, group_code: np.ndarray,
                   n_groups: int) -> dict:
    """Agrega contagens por cluster (arrays; mistos em dict separado)."""
    size = np.bincount(cluster, weights=size_w, minlength=n_clusters).astype(np.int64)
    fake = np.bincount(cluster, weights=fake_w, minlength=n_clusters).astype(np.int64)
    # grupo dominante por cluster; clusters multi-grupo vao para `mixed`
    tmp = pd.DataFrame({"c": cluster, "g": group_code, "n": size_w})
    ngroups = tmp.groupby("c")["g"].nunique()
    mixed_ids = set(ngroups[ngroups > 1].index.tolist())
    group_of = np.full(n_clusters, -1, dtype=np.int32)
    single = tmp[~tmp["c"].isin(mixed_ids)] if mixed_ids else tmp
    for c, g in single.groupby("c")["g"].first().items():
        group_of[int(c)] = int(g)
    mixed: dict = {}
    if mixed_ids:
        sub = tmp[tmp["c"].isin(mixed_ids)]
        for (c, g), cnt in sub.groupby(["c", "g"])["n"].sum().items():
            mixed.setdefault(int(c), {})[int(g)] = int(cnt)
    return {"size": size, "fake": fake, "group_of": group_of, "mixed": mixed}


def greedy_cluster_assign(units: dict, n_groups: int, sides: tuple,
                          targets: dict, seed: int) -> np.ndarray:
    """Atribui cada cluster a um lado preservando rotulo e grupo (~1 p.p.).

    Greedy deterministico por MAIOR DEFICIT normalizado: para cada cluster
    (maiores primeiro) escolhe o lado mais abaixo do alvo para o rotulo e o
    grupo do cluster (`target - atribuido/total`), em vez de minimizar o desvio
    absoluto (que encheria primeiro o menor lado). Jitter semeado desempata.
    Devolve um array lado->indice por cluster.
    """
    size = units["size"]
    fake = units["fake"]
    true = size - fake
    group_of = units["group_of"]
    mixed = units["mixed"]
    n_clusters = len(size)
    tot_fake = max(int(fake.sum()), 1)
    tot_true = max(int(true.sum()), 1)
    # totais de grupo: soma dos tamanhos dos clusters por grupo dominante +
    # parcelas dos clusters mistos
    tot_group_arr = np.zeros(n_groups, dtype=np.int64)
    valid = group_of >= 0
    np.add.at(tot_group_arr, group_of[valid], size[valid])
    for c, gc in mixed.items():
        for g, cnt in gc.items():
            tot_group_arr[g] += cnt
    tot_group = np.maximum(tot_group_arr, 1)

    rng = np.random.default_rng(seed)
    jitter = rng.random((n_clusters, len(sides)))
    order = sorted(range(n_clusters),
                   key=lambda i: (-int(size[i]), float(jitter[i, 0])))
    side_fake = {s: 0 for s in sides}
    side_true = {s: 0 for s in sides}
    side_group = {s: np.zeros(n_groups, dtype=np.int64) for s in sides}
    side_of = np.empty(n_clusters, dtype=np.int8)
    for i in order:
        best_s, best_score = None, -float("inf")
        fi, ti = int(fake[i]), int(true[i])
        for si, s in enumerate(sides):
            score = 0.0
            if fi:
                score += fi * (targets[s] - (side_fake[s] + fi) / tot_fake)
            if ti:
                score += ti * (targets[s] - (side_true[s] + ti) / tot_true)
            if group_of[i] >= 0:
                g = int(group_of[i])
                score += 0.5 * size[i] * (
                    targets[s] - (side_group[s][g] + size[i]) / tot_group[g])
            else:
                for g, cnt in mixed.get(i, {}).items():
                    score += 0.5 * cnt * (
                        targets[s] - (side_group[s][g] + cnt) / tot_group[g])
            score += 1e-6 * float(jitter[i, si])
            if score > best_score:
                best_score, best_s = score, s
        side_fake[best_s] += fi
        side_true[best_s] += ti
        if group_of[i] >= 0:
            side_group[best_s][int(group_of[i])] += int(size[i])
        else:
            for g, cnt in mixed.get(i, {}).items():
                side_group[best_s][g] += cnt
        side_of[i] = sides.index(best_s)
    return side_of


def cluster_split(pool: pd.DataFrame, cluster: np.ndarray, n_clusters: int,
                  subset_mask: np.ndarray, sides: tuple, targets: dict,
                  seed: int) -> np.ndarray:
    """Atribui clusters (restritos a `subset_mask`) aos lados do split."""
    sub = pool.loc[subset_mask]
    cl = cluster[subset_mask]
    size_w = np.ones(len(sub), dtype=np.int64)
    fake_w = (sub["label"].to_numpy() == "fake").astype(np.int64)
    codes = pd.factorize(pool["group"].astype(str))[0]
    n_groups = int(pool["group"].nunique())
    units = _cluster_units(cl, n_clusters, size_w, fake_w, codes[subset_mask],
                           n_groups)
    present = units["size"] > 0
    old_ids = np.flatnonzero(present)
    remap = {int(old): int(new) for new, old in enumerate(old_ids)}
    units = {
        "size": units["size"][present],
        "fake": units["fake"][present],
        "group_of": units["group_of"][present],
        "mixed": {remap[c]: v for c, v in units["mixed"].items() if present[c]},
    }
    side_local = greedy_cluster_assign(units, n_groups, sides, targets, seed)
    side_global = np.full(n_clusters, -1, dtype=np.int8)
    for new, old in enumerate(old_ids):
        side_global[int(old)] = side_local[new]
    return side_global


# --------------------------------------------------------------- DFR stats
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


def check_true(name: str, ok: bool, got=None, want=None) -> dict:
    return {"ok": bool(ok), "name": name, "got": got, "want": want}


def _weight_stats(w: np.ndarray, y: np.ndarray) -> dict:
    qs = np.percentile(w, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "min": round(float(qs[0]), 6),
        "p1": round(float(qs[1]), 6),
        "p5": round(float(qs[2]), 6),
        "p25": round(float(qs[3]), 6),
        "p50": round(float(qs[4]), 6),
        "p75": round(float(qs[5]), 6),
        "p95": round(float(qs[6]), 6),
        "p99": round(float(qs[7]), 6),
        "max": round(float(qs[8]), 6),
        "mean": round(float(w.mean()), 6),
        "prior_efetivo_fake": round(float(w[y == 1].sum() / w.sum()), 6),
    }


def dfr_weight_stats(train_df: pd.DataFrame, clip: float) -> dict:
    """Reproduz `build_sample_weights` (dfr=cell) do trainer v6, sem GPU."""
    d = pd.DataFrame({"group": train_df["group"].astype(str),
                      "target": (train_df["label"].astype(str) == "fake").astype(int)})
    y = d["target"].to_numpy()
    cells = d.groupby(["group", "target"]).size()
    w = D.group_balanced_weights(d)
    before = _weight_stats(w, y)
    w = np.minimum(w, clip)
    w = w * (len(w) / w.sum())
    after = _weight_stats(w, y)
    return {
        "n": int(len(d)),
        "n_cells": int(len(cells)),
        "clip": float(clip),
        "before_clip": before,
        "after_clip": after,
        "largest_cells": [
            {"group": str(g), "target": int(t), "n": int(n),
             "raw_weight": round(1.0 / n, 6)}
            for (g, t), n in cells.sort_values(ascending=False).head(5).items()
        ],
        "smallest_cells": [
            {"group": str(g), "target": int(t), "n": int(n),
             "raw_weight": round(1.0 / n, 6)}
            for (g, t), n in cells.sort_values().head(5).items()
        ],
    }


# --------------------------------------------------------------- quase-dup
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
        "metric": "5-gram word shingles normalizados, Jaccard exato na amostra (indice invertido)",
        "n_train_sample": len(tr_sample),
        "n_test_sample": len(te_sample),
        "n_pares_com_intersecao": int(n_pairs),
        "max_jaccard": round(float(max_j), 6),
        "n_pares_gt_0.8": int(n_above),
        "seconds": round(time.time() - t0, 2),
    }


# ------------------------------------------------------------- token stats
def length_grouped_batches(lengths, batch_size, generator, mega=50):
    """Copia verbatim de models/encoder.py:83 (mesma simulacao do PLANO v4)."""
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


def tokenize_lengths(tokenizer, texts) -> np.ndarray:
    texts = list(texts)
    lens = np.empty(len(texts), dtype=np.int64)
    for i in range(0, len(texts), 1024):
        enc = tokenizer(texts[i:i + 1024], add_special_tokens=True,
                        truncation=False)
        for j, ids in enumerate(enc["input_ids"]):
            lens[i + j] = len(ids)
    return lens


def compute_token_stats(tokenizer, texts_by_cut: dict,
                        cost_lens_by_split: dict) -> dict:
    """Distribuicoes por recorte + custo por split de treino (128/192/256)."""
    result = {"tokenizer": tokenizer.name_or_path, "add_special_tokens": True,
              "truncation": False, "distributions": {}, "cost_by_split": {}}
    for name, texts in texts_by_cut.items():
        t0 = time.time()
        lens = tokenize_lengths(tokenizer, texts)
        d = dist_stats(lens)
        d["tokenize_seconds"] = round(time.time() - t0, 2)
        result["distributions"][name] = d
    for split, lens in cost_lens_by_split.items():
        result["cost_by_split"][split] = {
            str(cap): cap_cost(lens, cap) for cap in (128, 192, 256)}
        base = result["cost_by_split"][split]["192"]["tokens_pagos_por_epoca"]
        result["cost_by_split"][split]["razao_256_192"] = round(
            result["cost_by_split"][split]["256"]["tokens_pagos_por_epoca"] / base, 4) if base else None
    result["cap_tokens_pagos_por_epoca"] = {
        split: {str(c): result["cost_by_split"][split][str(c)]["tokens_pagos_por_epoca"]
                for c in (128, 192, 256)}
        for split in result["cost_by_split"]}
    result["assumptions"] = {
        "batching": "length_grouped_batches(batch=32, mega=50, seed=42), comprimentos truncados no cap",
        "tokens_pagos": "sum(len(batch) * max(len_no_batch))",
        "cost_split": "trem de cada split (full_iid/ood_wa/bal_iid)",
    }
    return result


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sanitized", default=str(ROOT / "data" / "FakenewsBR_sanitized_v6.csv"))
    ap.add_argument("--labels", default=str(ROOT / "data" / "FakenewsBR_v6_labels.csv"))
    ap.add_argument("--provenance", default=str(ROOT / "data" / "FakenewsBR_v6_provenance.csv"))
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "pipeline" / "prepare" / "processed"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-sel-frac", type=float, default=0.10)
    ap.add_argument("--val-calib-frac", type=float, default=0.05)
    ap.add_argument("--ood-channel", default="whatsapp")
    ap.add_argument("--token-stats", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    pool_path = out_dir / "v6_pool.parquet"
    splits_path = out_dir / "v6_splits.parquet"
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

    print(f"[1/8] load({Path(args.sanitized).name}, ...)")
    t1 = time.time()
    df = D.load(csv=args.sanitized, labels_csv=args.labels,
                provenance_csv=args.provenance)
    load_s = time.time() - t1
    n_raw = len(df)
    fake_raw = int((df["label"] == "fake").sum())
    true_raw = int((df["label"] == "true").sum())
    print(f"      n={n_raw} fake={fake_raw} true={true_raw} ({load_s:.1f}s)")

    checks: dict[str, dict] = {}
    checks["pool_91080"] = check_equal("pool", n_raw, EXPECTED_POOL)
    checks["fake_66772"] = check_equal("fake", fake_raw, EXPECTED_FAKE)
    checks["true_24308"] = check_equal("true", true_raw, EXPECTED_TRUE)
    checks["rid_unico"] = {"name": "rid_unico", "ok": bool(df["rid"].is_unique),
                           "got": int(df["rid"].nunique()), "want": n_raw}

    # invariantes do labels CSV completo (297.672 linhas rotuladas ou nao)
    lab = pd.read_csv(args.labels, low_memory=False)
    lab_tiers = lab["label_tier"].fillna("none").value_counts().to_dict()
    checks["sanitized_297672"] = check_equal("sanitized", len(lab), EXPECTED_SANITIZED)
    checks["sanitized_rid_unico"] = check_true("sanitized_rid_unico",
                                               bool(lab["rid"].is_unique),
                                               int(lab["rid"].nunique()), len(lab))
    checks["sanitized_tiers"] = check_equal("sanitized_tiers",
                                            {k: int(v) for k, v in lab_tiers.items()},
                                            EXPECTED_SANITIZED_TIERS)
    train_label_vc = lab["train_label"].value_counts(dropna=False).to_dict()
    checks["train_label_counts"] = check_equal(
        "train_label_counts",
        {("nan" if (isinstance(k, float) and np.isnan(k)) else k): int(v)
         for k, v in train_label_vc.items()},
        {"nan": EXPECTED_SANITIZED - EXPECTED_POOL, "fake": EXPECTED_FAKE,
         "true": EXPECTED_TRUE})
    n_verified = int(lab["verified_label"].notna().sum())
    checks["verified_label_cobertura"] = check_true(
        "verified_label_cobertura", n_verified == EXPECTED_SANITIZED,
        n_verified, EXPECTED_SANITIZED)

    print("[2/8] flags de ruido (U+FFFD, mojibake)")
    has_ufffd = df[TEXT_COL].astype(str).str.contains(chr(0xFFFD), regex=False)
    mojibake = df[TEXT_COL].astype(str).str.contains(LATIN_EXT_RE, regex=True)
    ufffd_rids = df.loc[has_ufffd, "rid"].astype("int64").tolist()
    checks["ufffd_4"] = check_equal("ufffd", int(has_ufffd.sum()), EXPECTED_UFFFD)
    print(f"      U+FFFD={int(has_ufffd.sum())} rids={ufffd_rids}")
    print(f"      mojibake(Latin Ext)= {int(mojibake.sum())}")

    print("[3/8] colunas derivadas")
    pool = pd.DataFrame({
        "rid": df["rid"].astype("int64"),
        "text_no_url": df[TEXT_COL].astype(str),
        "group": df["group"].astype(str),
        "channel": df["channel"].astype(str),
        "label": df["label"].astype(str),
        "label_tier": df["label_tier"].fillna("none").astype(str),
        "label_source": df["label_source"].fillna("desconhecido").astype(str),
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
    assert list(pool.columns)[:20] == [
        "rid", "text_no_url", "group", "channel", "label", "label_tier",
        "label_source", "is_balanced_group", "is_ptpt_rule", "lang_variant",
        "publisher", "rating_class", "era", "word_len", "num_exclamations",
        "num_questions", "num_ellipsis", "uppercase_word_ratio", "has_ufffd",
        "mojibake_flag"]
    n_info = int(pool["is_balanced_group"].sum())
    const_detail = {}
    for g, sub in pool.groupby("group")["label"]:
        if sub.nunique() == 1:
            const_detail[g] = int(len(sub))
    const_n = int(sum(const_detail.values()))
    checks["informativos_36896"] = check_equal("informativos", n_info, EXPECTED_INFO)
    checks["grupos_36"] = check_equal("grupos", int(pool["group"].nunique()),
                                      EXPECTED_GROUPS)
    checks["constantes_11"] = check_equal("constantes_grupos", len(const_detail),
                                          EXPECTED_CONST_GROUPS)
    checks["constantes_39373"] = check_equal("constantes_linhas", const_n,
                                             EXPECTED_CONST_LINES)
    checks["constantes_detalhe"] = check_equal(
        "constantes_detalhe", {k: v for k, v in sorted(const_detail.items())},
        {k: v[0] for k, v in sorted(EXPECTED_CONST_GROUPS_DETAIL.items())})
    info_detail = pool[pool["is_balanced_group"]].groupby("group").size().to_dict()
    checks["informativos_detalhe"] = check_equal(
        "informativos_detalhe", {k: int(v) for k, v in sorted(info_detail.items())},
        {k: v for k, v in sorted(EXPECTED_INFO_GROUPS.items())})
    tier_pool = pool["label_tier"].value_counts().to_dict()
    checks["tiers_pool"] = check_equal("tiers_pool",
                                       {k: int(v) for k, v in tier_pool.items()},
                                       EXPECTED_POOL_TIERS)
    print(f"      informativos={n_info} constantes={const_n} grupos={pool['group'].nunique()}")

    print("[4/8] duplicatas exatas normalizadas + clusters near-dup")
    pool["text_key"] = pool["text_no_url"].map(normalize_text)
    tk_vc = pool["text_key"].value_counts()
    dup_keys = tk_vc[tk_vc > 1].index
    dup_rows = int(tk_vc[tk_vc > 1].sum())
    dup_groups = int(len(dup_keys))
    dup_extra = int(dup_rows - dup_groups)
    dups = pool[pool["text_key"].isin(dup_keys)]
    dup_conf = dups.groupby("text_key")["label"].nunique()
    dup_channels = dups.groupby("text_key")["channel"].nunique()
    n_conf = int((dup_conf > 1).sum())
    n_chan = int((dup_channels > 1).sum())
    checks["dups_groups_427"] = check_equal("dups_groups", dup_groups, EXPECTED_DUP_GROUPS)
    checks["dups_rows_900"] = check_equal("dups_rows", dup_rows, EXPECTED_DUP_ROWS)
    checks["dups_extra_473"] = check_equal("dups_extra", dup_extra, EXPECTED_DUP_EXTRA)
    print(f"      exatas: grupos={dup_groups} linhas={dup_rows} extras={dup_extra} "
          f"conflitos_rotulo={n_conf} multi_canal={n_chan}")

    cluster, hashes_by_variant, nd_stats = build_near_dup_clusters(pool, seed=args.seed)
    pool["near_dup_cluster"] = cluster
    n_ver = {k: v["n_pairs_verified_ge_0.8"]
             for k, v in nd_stats["variants"].items()}
    print(f"      clusters: total={nd_stats['n_clusters']} "
          f"multi={nd_stats['n_multi_clusters']} max_size={nd_stats['max_cluster_size']} "
          f"linhas_em_multi={nd_stats['n_rows_in_multi_clusters']} "
          f"pares_verificados={n_ver} "
          f"({nd_stats['seconds']['total']}s)")

    print("[5/8] splits por cluster (greedy rotulo+grupo; seed=%d)" % args.seed)
    splits = pd.DataFrame({"rid": pool["rid"].to_numpy(),
                           "split_full_iid": "unused",
                           "split_ood_wa": "unused",
                           "split_bal_iid": "unused",
                           "near_dup_cluster": cluster.astype("int32")})

    targets4 = {"train": 0.70, "val_sel": 0.10, "val_calib": 0.05, "test": 0.15}
    targets3 = {"train": 0.85, "val_sel": 0.10, "val_calib": 0.05}
    n_clusters = int(nd_stats["n_clusters"])
    codes, _ = pd.factorize(pool["group"].astype(str))
    n_groups = int(pool["group"].nunique())
    all_mask = np.ones(len(pool), dtype=bool)

    wa_mask = (pool["channel"] == args.ood_channel).to_numpy()
    if wa_mask.sum() == 0:
        print(f"[ERRO] canal OOD '{args.ood_channel}' vazio")
        return 2
    wa_clusters = set(np.unique(cluster[wa_mask]).tolist())
    rest_mask = ~wa_mask
    rest_clean_mask = rest_mask & ~np.isin(cluster, list(wa_clusters))
    n_ood_excluded = int(rest_mask.sum() - rest_clean_mask.sum())

    def build_sides():
        side_full = cluster_split(pool, cluster, n_clusters, all_mask,
                                  SIDES_4, targets4, args.seed)
        row_full = np.array(SIDES_4, dtype=object)[side_full[cluster]]
        row_ood = np.full(len(pool), "unused", dtype=object)
        row_ood[wa_mask] = "test"
        side_rest = cluster_split(pool, cluster, n_clusters, rest_clean_mask,
                                  SIDES_3, targets3, args.seed + 1)
        row_ood[rest_clean_mask] = np.array(SIDES_3, dtype=object)[
            side_rest[cluster[rest_clean_mask]]]
        bal_mask = pool["is_balanced_group"].to_numpy()
        side_bal = cluster_split(pool, cluster, n_clusters, bal_mask,
                                 SIDES_4, targets4, args.seed + 2)
        row_bal = np.full(len(pool), "unused", dtype=object)
        row_bal[bal_mask] = np.array(SIDES_4, dtype=object)[
            side_bal[cluster[bal_mask]]]
        return row_full, row_ood, row_bal

    row_full, row_ood, row_bal = build_sides()

    def run_cross_scan(idx_a, idx_b, label):
        res = {"comparison": label, "variants": {}, "violations": [],
               "n_ge_0.8": 0}
        for name, hashes in hashes_by_variant.items():
            r = exhaustive_cross_scan([hashes[i] for i in idx_a],
                                      [hashes[i] for i in idx_b],
                                      label=f"{label}.{name}")
            res["violations"].extend(
                (name, int(x), int(y)) for x, y in r["violations"])
            res["n_ge_0.8"] += r["n_ge_0.8"]
            res["variants"][name] = {k: v for k, v in r.items()
                                     if k != "violations"}
        return res

    # Varredura exata pos-split; se o LSH perdeu algum par >=0.8, unimos os
    # clusters dos pares e refazemos os splits (garante 0 por construcao).
    scans: dict = {}
    attempts = 0
    while True:
        attempts += 1
        rows = {"full_iid": row_full, "ood_wa": row_ood, "bal_iid": row_bal}
        slices = {}
        for split_name, row in rows.items():
            slices[split_name] = {
                "train": np.flatnonzero(row == "train"),
                "val": np.flatnonzero((row == "val_sel") | (row == "val_calib")),
                "test": np.flatnonzero(row == "test"),
            }
        scans = {}
        for split_name, s in slices.items():
            scans[f"{split_name}.train_x_test"] = run_cross_scan(
                s["train"], s["test"], f"{split_name}.train_x_test")
            scans[f"{split_name}.train_x_val"] = run_cross_scan(
                s["train"], s["val"], f"{split_name}.train_x_val")
        n_viol = sum(v["n_ge_0.8"] for v in scans.values())
        if n_viol == 0 or attempts >= 3:
            break
        print(f"      [aviso] {n_viol} pares >=0.8 cruzando lados; "
              f"unindo clusters e refazendo splits (tentativa {attempts})")
        uf2 = UnionFind(n_clusters)
        for split_name, s in slices.items():
            for key, other in ((f"{split_name}.train_x_test", "test"),
                               (f"{split_name}.train_x_val", "val")):
                ia, ib = s["train"], s[other]
                for _variant, x, y in scans[key]["violations"]:
                    uf2.union(int(cluster[ia[x]]), int(cluster[ib[y]]))
        roots = np.array([uf2.find(int(c)) for c in cluster], dtype=np.int64)
        _, cluster = np.unique(roots, return_inverse=True)
        cluster = cluster.astype(np.int32)
        n_clusters = int(cluster.max()) + 1
        pool["near_dup_cluster"] = cluster
        splits["near_dup_cluster"] = cluster.astype("int32")
        row_full, row_ood, row_bal = build_sides()
        print(f"      clusters apos uniao: {n_clusters}")

    splits["split_full_iid"] = row_full
    splits["split_ood_wa"] = row_ood
    splits["split_bal_iid"] = row_bal

    print(f"      ood_wa: teste={int(wa_mask.sum())} whatsapp; copias nao-wa "
          f"excluidas do treino (unused)={n_ood_excluded}")

    counts = split_counts(splits, pool)
    for col in SPLIT_COLS:
        c = counts[col]
        print(f"      {col}: train={c['train']['n']}({c['train']['fake_pct']}%fake) "
              f"val_sel={c['val_sel']['n']}({c['val_sel']['fake_pct']}%) "
              f"val_calib={c['val_calib']['n']}({c['val_calib']['fake_pct']}%) "
              f"test={c['test']['n']}({c['test']['fake_pct']}%) "
              f"unused={c['unused']['n']}")

    # ---------------- validacoes estruturais dos splits
    null_counts = {c: int(splits[c].isna().sum()) for c in SPLIT_COLS}
    checks["sem_nulos"] = check_true("sem_nulos", all(v == 0 for v in null_counts.values()),
                                     null_counts, {c: 0 for c in SPLIT_COLS})

    active = {"train", "val_sel", "val_calib", "test"}
    for col in SPLIT_COLS:
        d = pd.DataFrame({"c": splits["near_dup_cluster"].to_numpy(),
                          "side": splits[col].to_numpy()})
        d = d[d["side"].isin(active)]
        n_multi = int((d.groupby("c")["side"].nunique() > 1).sum())
        checks[f"{col}.cluster_um_lado"] = check_equal(
            f"{col}.cluster_um_lado", n_multi, 0)
        learn = d[d["side"].isin(["train", "val_sel", "val_calib"])]["c"]
        test = set(d.loc[d["side"] == "test", "c"])
        checks[f"{col}.sem_cluster_treino_teste"] = check_equal(
            f"{col}.sem_cluster_treino_teste", int(learn.isin(test).sum()), 0)
        # texto exato: nenhum text_key cruzando lados ativos
        k = pd.DataFrame({"tk": pool["text_key"].to_numpy(),
                          "side": splits[col].to_numpy()})
        k = k[k["side"].isin(active)]
        kk = k.groupby("tk")["side"].nunique()
        checks[f"{col}.textkey_um_lado"] = check_equal(
            f"{col}.textkey_um_lado", int((kk > 1).sum()), 0)

    # proporcoes por linha e por rotulo (tolerancia 1,0 p.p.)
    for col, targets in (("split_full_iid", targets4), ("split_ood_wa", targets3),
                         ("split_bal_iid", targets4)):
        c = counts[col]
        base = sum(c[s]["n"] for s in targets)
        for name, want in targets.items():
            got = c[name]["n"] / base if base else 0.0
            checks[f"{col}.{name}_frac"] = check_true(
                f"{col}.{name}_frac", abs(got - want) <= 0.01,
                round(got, 5), want)
        fake_global = (sum(c[s]["fake"] for s in targets) / base) if base else 0.0
        for name in targets:
            if c[name]["n"] == 0:
                continue
            dev = abs(c[name]["fake_pct"] / 100.0 - fake_global)
            checks[f"{col}.{name}_fake_dev"] = check_true(
                f"{col}.{name}_fake_dev", dev <= 0.01, round(dev, 5), 0.01)

    # proporcoes por grupo (max desvio) -- registrado e checado em n>=200
    group_dev = {}
    for col, targets in (("split_full_iid", targets4), ("split_ood_wa", targets3),
                         ("split_bal_iid", targets4)):
        dd = pd.DataFrame({"g": pool["group"].to_numpy(),
                           "side": splits[col].to_numpy()})
        dd = dd[dd["side"].isin(targets)]
        tot = dd.groupby("g").size()
        for g in tot.index:
            if tot[g] < 200:
                continue
            for s, want in targets.items():
                frac = (dd[(dd["g"] == g) & (dd["side"] == s)].shape[0]
                        / max(tot[g], 1))
                group_dev.setdefault(col, 0.0)
                group_dev[col] = max(group_dev[col], abs(frac - want))
    for col, dev in group_dev.items():
        checks[f"{col}.grupo_max_dev"] = check_true(
            f"{col}.grupo_max_dev", dev <= 0.01, round(dev, 5), 0.01)
    print(f"      max desvio por grupo (n>=200): { {k: round(v,4) for k,v in group_dev.items()} }")

    # varredura exata cross-side (near-dup) -- ja executada no loop acima.
    # Exige 0 pares >=0.8 em AMBAS as variantes (norm e raw).
    for key, res in scans.items():
        for vname, vres in res["variants"].items():
            checks[f"scan.{key}.{vname}.ge_0.8"] = check_equal(
                f"scan.{key}.{vname}.ge_0.8", vres["n_ge_0.8"], 0)
    for key, res in scans.items():
        for vname, vres in res["variants"].items():
            print(f"      varredura {key} [{vname}]: max_j={vres['max_jaccard']} "
                  f">=0.8={vres['n_ge_0.8']} 0.7-0.8={vres['n_0.7_0.8']} "
                  f"0.5-0.7={vres['n_0.5_0.7']} ({vres['seconds']}s)")

    probe = near_dup_probe(pool.loc[splits["split_full_iid"] == "train", TEXT_COL],
                           pool.loc[splits["split_full_iid"] == "test", TEXT_COL],
                           n_sample=5_000, seed=args.seed)
    print(f"      sonda quase-dup amostral (full_iid): max_jaccard={probe['max_jaccard']} "
          f"pares>0.8={probe['n_pares_gt_0.8']}")

    print("[6/8] pesos DFR cell clip=25 por split de treino")
    dfr_stats = {}
    for col in SPLIT_COLS:
        tr_sub = pool[splits[col].to_numpy() == "train"]
        dfr_stats[col] = dfr_weight_stats(tr_sub, clip=25.0)
        a = dfr_stats[col]["after_clip"]
        print(f"      {col}: n={dfr_stats[col]['n']} cells={dfr_stats[col]['n_cells']} "
              f"prior_fake_efetivo={a['prior_efetivo_fake']:.4f} "
              f"w_min={a['min']:.4f} w_max={a['max']:.2f} w_p50={a['p50']:.3f}")

    print("[7/8] escrevendo parquets")
    pool_out = pool.drop(columns=["text_key", "near_dup_cluster"])
    pool_out.to_parquet(pool_path, index=False, compression="snappy")
    splits.to_parquet(splits_path, index=False, compression="snappy")
    pool_mb = pool_path.stat().st_size / 1e6
    checks["pool_le_25mb"] = check_true("pool_le_25mb", pool_mb <= 25.0,
                                        round(pool_mb, 2), 25.0)
    print(f"      v6_pool.parquet = {pool_mb:.2f} MB")
    print(f"      v6_splits.parquet = {splits_path.stat().st_size/1e3:.1f} KB")

    token_stats = None
    if args.token_stats:
        print("[7b/8] token stats (tokenizer BERTimbau cacheado)")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        texts_by_cut = {
            "pool": pool_out[TEXT_COL],
            "informativos": pool_out.loc[pool_out["is_balanced_group"], TEXT_COL],
        }
        cost_lens = {
            "train_full_iid": tokenize_lengths(
                tok, pool_out.loc[splits["split_full_iid"] == "train", TEXT_COL]),
            "train_ood_wa": tokenize_lengths(
                tok, pool_out.loc[splits["split_ood_wa"] == "train", TEXT_COL]),
            "train_bal_iid": tokenize_lengths(
                tok, pool_out.loc[splits["split_bal_iid"] == "train", TEXT_COL]),
        }
        token_stats = compute_token_stats(tok, texts_by_cut, cost_lens)
        for cut, d in token_stats["distributions"].items():
            print(f"      {cut}: n={d['n']} media={d['mean']} p50={d['p50']} "
                  f"p95={d['p95']} p99={d['p99']} >192={d['pct_gt_192']}%")
        for split, c in token_stats["cost_by_split"].items():
            print(f"      {split}: 192={c['192']['tokens_pagos_por_epoca_milhoes']}M "
                  f"tokens/epoca razao_256_192={c['razao_256_192']}")
        tok_path.write_text(json.dumps(token_stats, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    failed = [k for k, v in checks.items() if not v["ok"]]
    dup_info = {
        "normalization": "investigation/expansion/dedup.py::normalize_text "
                         "(NFKD sem acento, [a-z0-9 ], espacos colapsados)",
        "n_groups": dup_groups,
        "n_rows": dup_rows,
        "n_extra_copies": dup_extra,
        "n_conflict_groups_fake_true": n_conf,
        "n_groups_multi_channel": n_chan,
        "policy": (
            "NAO remove linhas: o split e feito no nivel do cluster near-dup "
            "(MinHash+LSH+text_key), entao todas as copias e gemeas ficam no "
            "mesmo lado. No ood_wa, os clusters que tocam o canal whatsapp tem "
            "as linhas nao-whatsapp 'unused' (nunca entram no treino)."),
        "ood_wa_excluded_nao_whatsapp": n_ood_excluded,
    }
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
            "sanitized_rows": int(len(lab)),
            "sanitized_label_tiers": {k: int(v) for k, v in lab_tiers.items()},
            "sanitized_train_label": {
                ("unknown" if (isinstance(k, float) and np.isnan(k)) else str(k)): int(v)
                for k, v in train_label_vc.items()},
            "verified_label_notna": n_verified,
            "pool_raw": n_raw, "fake_raw": fake_raw, "true_raw": true_raw,
            "pool": len(pool_out),
            "fake": int((pool_out["label"] == "fake").sum()),
            "true": int((pool_out["label"] == "true").sum()),
            "fake_pct": round(float((pool_out["label"] == "fake").mean() * 100), 2),
            "informativos": n_info,
            "informativos_fake_pct": round(
                float((pool_out.loc[pool_out["is_balanced_group"], "label"] == "fake")
                      .mean() * 100), 2),
            "informativos_detail": {k: int(v) for k, v in sorted(info_detail.items())},
            "groups": int(pool_out["group"].nunique()),
            "constant_groups": len(const_detail),
            "constant_lines": const_n,
            "constant_groups_detail": {k: int(v) for k, v in sorted(const_detail.items())},
            "ufffd_marked": int(has_ufffd.sum()),
            "ufffd_rids": ufffd_rids,
            "ufffd_policy": (
                "linhas U+FFFD ficam no pool marcadas has_ufffd=True e sao "
                "removidas do treino/eval pelo train_bertimbau_v6.py (E2); "
                "o metrics.json registra n_test_bruto e n_ufffd_removidas."),
            "mojibake_flagged": int(mojibake.sum()),
            "pool_label_tiers": {k: int(v) for k, v in tier_pool.items()},
        },
        "duplicates": dup_info,
        "near_dup": nd_stats,
        "splits": counts,
        "near_dup_scans": {k: {kk: vv for kk, vv in v.items() if kk != "violations"}
                           for k, v in scans.items()},
        "cluster_split_attempts": attempts,
        "dfr_weights": dfr_stats,
        "validation": {"checks": checks, "all_pass": not failed, "failed": failed},
        "near_dup_probe": probe,
        "sha256": {
            "v6_pool.parquet": sha256_file(pool_path),
            "v6_splits.parquet": sha256_file(splits_path),
        },
        "files": {
            "v6_pool.parquet": str(pool_path.resolve()),
            "v6_splits.parquet": str(splits_path.resolve()),
            "v6_pool_bytes": pool_path.stat().st_size,
            "v6_splits_bytes": splits_path.stat().st_size,
        },
        "versions": versions,
        "total_seconds": round(time.time() - t0, 1),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    print("[8/8] resumo")
    print(f"      all_pass={not failed} failed={failed}")
    print(f"      arquivos: {pool_path} | {splits_path} | {stats_path}"
          + (f" | {tok_path}" if token_stats else ""))
    if failed:
        print("[ERRO] validacoes falharam (codigo 2)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
