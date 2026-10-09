#!/usr/bin/env python
"""train_bertimbau_v6.py -- treino/avaliacao/calibracao/retomada do BERTimbau v6.

Single-file, Colab-ready: NAO importa nada do repositorio no caminho de treino.
O bloco VENDOR abaixo e copia verbatim de `models/evaluate.py` (e de
`models/data.py::group_balanced_weights`), para rodar no Colab apenas com os
parquets. `--check-vendor` (somente local) compara bloco embutido x original.

Adaptacao do trainer v4 (`models/v4/train_bertimbau_v4.py`, testado) para o pool
COMPLETO da v6 (91.080 linhas). Defaults v6:

  --split-col full_iid --eval-col full_iid --eval-value test
  --dfr-weights cell --weight-clip 25 --epochs 2

O restante dos controles metodologicos e o mesmo do v4 (val_sel/val_calib,
Platt em val_calib com mascara is_balanced_group, pior-grupo, teste intocado).

Contrato e decisoes: `models/v6/PLANO_ADAPTACAO_V6.md` + `models/v4/spec_treino_v4.md`.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_scheduler)

TEXT_COL = "text_no_url"
RUN_ID_TEMPLATE = "{split_col}_ml{max_length}_f{freeze}_{dfr}_seed{seed}"

# ======================================================================
# === VENDOR: models/evaluate.py @ 08d1acb0b38627c0251f282f8bc68f315c4ecf90a970bcd3f6d0e9f5a8211557 ===
# === EXTENSAO v6 (CODE_REVIEW V4): per_group/worst_group_f1/report aceitam
# === `threshold` opcional (default 0.5). Com os defaults o comportamento e
# === identico ao original; --check-vendor confirma maxdiff 0.
# ======================================================================

# ---------------------------------------------------------------- calibracao

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
              col: str = "group", min_n: int = 30,
              threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for name, idx in df.groupby(col).indices.items():
        if len(idx) < min_n:
            continue
        yi = y[idx]
        m = core_metrics(yi, p[idx], threshold)
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
                   only_groups: list | None = None,
                   threshold: float = 0.5) -> float:
    """Metrica de manchete: pior macro-F1 entre grupos ESTATISTICAMENTE confiaveis.

    Ignora grupos degenerados (uma classe so) e grupos pequenos demais, que
    produziriam um minimo dominado por ruido amostral. O `threshold` decide a
    classe predita em todas as metricas (V4).
    """
    d, yy, pp = df, y, p
    if only_groups is not None:
        m = df[col].isin(only_groups).to_numpy()
        if m.sum() == 0:
            return float("nan")
        d, yy, pp = df[m].reset_index(drop=True), y[m], p[m]
    tab = per_group(d, yy, pp, col=col, min_n=min_n, threshold=threshold)
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

    tab = per_group(df, y, p, threshold=threshold)
    if not tab.empty:
        print("\nPOR GRUPO (ordenado do pior para o melhor):")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        frac = tab[~tab["confiavel"]]
        if not frac.empty:
            print(f"  (grupos com confiavel=False sao pequenos demais — "
                  f"n<{MIN_GROUP_N} ou minoria<{MIN_MINORITY_N} — e ficam fora da manchete)")
        wg = worst_group_f1(df, y, p, threshold=threshold)
        print(f"\n>>> PIOR GRUPO macro-F1 = {wg:.4f}   (metrica de manchete, "
              f"so grupos confiaveis)")
        glob["worst_group_macro_f1"] = wg

        # Restrito aos grupos sem confundimento origem->rotulo: e onde
        # "robustez de grupo" e uma pergunta bem-posta.
        if "is_balanced_group" in df.columns and bool(df["is_balanced_group"].any()):
            m = df["is_balanced_group"].to_numpy()
            wgb = worst_group_f1(df[m].reset_index(drop=True), y[m], p[m],
                                 threshold=threshold)
            print(f">>> PIOR GRUPO (so grupos balanceados) = {wgb:.4f}")
            glob["worst_group_balanced"] = wgb

            # O score e calibrado sob prior balanceado, entao o ECE medido sobre
            # a base inteira (71,5% fake, prior artefatual) penaliza uma escolha
            # deliberada. Este e o ECE no regime em que o score foi definido.
            bm = core_metrics(y[m], p[m], threshold)
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
            hm = core_metrics(y[hard], p[hard], threshold)
            print(f"\nCASOS LIMITROFES  n={hm['n']}  acc={hm['acc']:.4f}  "
                  f"score medio={p[hard].mean():.3f}")
            glob["hard_acc"] = hm["acc"]
        if len(pure) >= 20:
            pm = core_metrics(y[pure], p[pure], threshold)
            print(f"FALSO PURO        n={pm['n']}  acc={pm['acc']:.4f}  "
                  f"score medio={p[pure].mean():.3f}")
            glob["pure_acc"] = pm["acc"]

    if "is_ptpt" in df.columns:
        pt = np.flatnonzero(df["is_ptpt"].to_numpy())
        br = np.flatnonzero(~df["is_ptpt"].to_numpy())
        if len(pt) >= 30 and len(br) >= 30:
            a, b = (core_metrics(y[pt], p[pt], threshold),
                    core_metrics(y[br], p[br], threshold))
            print(f"\nDIALETO  PT-PT: n={a['n']} macro-F1={a['macro_f1']:.4f} | "
                  f"resto: n={b['n']} macro-F1={b['macro_f1']:.4f}")
            glob["ptpt_macro_f1"] = a["macro_f1"]

    return glob


# ======================================================================
# === END VENDOR: models/evaluate.py ===
# ======================================================================

# ======================================================================
# === VENDOR: models/data.py::group_balanced_weights @ cd7bb66399124faa0a4b3e7bd6261a6ced9c9f0f37c44d8b0c8f5b627a32b9e4 ===
# ======================================================================

def group_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """Peso por amostra que iguala o peso TOTAL de cada celula (grupo x rotulo).

    Base do DFR (Kirichenko et al., 2023): retreinar a ultima camada com os
    grupos equilibrados remove a dependencia de atalhos de proveniencia sem
    tocar no encoder.

    Usa ponderacao em vez de reamostragem porque os grupos aqui diferem em duas
    ordens de grandeza (Fake.br tem 2.506 exemplos por celula, COVID19.BR_raw
    tem 41). Igualar por downsampling destruiria a base — na pratica sobravam
    ~156 exemplos. A ponderacao e equivalente em esperanca e usa tudo.
    """
    cells = df.groupby(["group", "target"]).indices
    w = np.ones(len(df), dtype=np.float64)
    if not cells:
        return w
    for idx in cells.values():
        w[idx] = 1.0 / len(idx)
    return w * (len(df) / w.sum())  # normaliza para media 1


# ======================================================================
# === END VENDOR: models/data.py ===
# ======================================================================


# ------------------------------------------------------------------ utils
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    return o


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def atomic_copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------- batching / modelo
def length_grouped_batches(lengths, batch_size, generator, mega=50):
    """Ordena por comprimento dentro de megabatches para minimizar padding,
    mantendo aleatoriedade entre epocas. Copia verbatim de models/encoder.py:83."""
    idx = torch.randperm(len(lengths), generator=generator).tolist()
    span = batch_size * mega
    out = []
    for i in range(0, len(idx), span):
        chunk = sorted(idx[i:i + span], key=lambda j: lengths[j])
        out += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    perm = torch.randperm(len(out), generator=generator).tolist()
    return [out[i] for i in perm]


class LengthGroupedBatchSampler(BatchSampler):
    def __init__(self, lengths, batch_size: int, seed: int, mega: int = 50):
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.mega = int(mega)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(length_grouped_batches(self.lengths, self.batch_size, g,
                                           self.mega))

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


class TextDS(Dataset):
    """Guarda tokens SEM padding — o collator padda por batch."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        item["idx"] = i
        return item

    def lengths(self):
        return [len(x) for x in self.enc["input_ids"]]


# Collators em nivel de modulo (picklaveis para DataLoader com workers > 0
# no Windows/spawn; CODE_REVIEW V3). Nao usam closure de `main`.
class TrainCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, feats):
        idx = [int(f.pop("idx")) for f in feats]
        batch = self.collator(feats)
        batch["idx"] = torch.tensor(idx, dtype=torch.long)
        return batch


class EvalCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, feats):
        for f in feats:
            f.pop("idx", None)
        return self.collator(feats)


def infer(model, ds, collate, batch_size: int, device, use_amp: bool,
          amp_dtype) -> np.ndarray:
    model.eval()
    logits = []
    order = np.argsort(ds.lengths(), kind="stable")
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            sel = order[s:s + batch_size]
            batch = collate([ds[i] for i in sel])
            batch.pop("labels", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp \
                else contextlib.nullcontext()
            with ctx:
                out = model(**batch).logits
            logits.append(out.float().cpu().numpy())
    res = np.zeros((len(ds), 2), dtype=np.float32)
    if len(logits):
        res[order] = np.concatenate(logits)
    return res


def sigmoid_z(logits: np.ndarray) -> np.ndarray:
    z = _decision(logits)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def text_key(s: str) -> str:
    """Normalizacao de texto do prepare_v6 (copiada de dedup.normalize_text).

    Usada no controle de vazamento treino x teste: duplicatas exatas
    normalizadas nunca devem aparecer dos dois lados.
    """
    import re
    import unicodedata
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def mask_entities_text(text: str) -> str:
    import re
    masks = [
        (r"\b(lula|bolsonaro|dilma|temer|doria|ciro|haddad|moraes)\b", "[POLITICO]"),
        (r"\b(cloroquina|ivermectina|vacina|coronavac|pfizer|astrazeneca)\b", "[SAUDE]"),
        (r"\b(stf|tse|minist[eé]rio p[uú]blico)\b", "[INSTITUICAO]"),
    ]
    if not isinstance(text, str):
        return ""
    for pat, tag in masks:
        text = re.sub(pat, tag, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------- token stats
def _dist(lens: np.ndarray) -> dict:
    lens = np.asarray(lens, dtype=np.int64)
    if len(lens) == 0:
        return {"n": 0}
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


def _cap_cost(lens: np.ndarray, cap: int, batch_size: int = 32, mega: int = 50,
              seed: int = 42) -> dict:
    clipped = np.minimum(np.asarray(lens, dtype=np.int64), cap)
    g = torch.Generator().manual_seed(seed)
    batches = length_grouped_batches(clipped.tolist(), batch_size, g, mega)
    paid, at_cap = 0, 0
    for b in batches:
        m = max(clipped[j] for j in b)
        paid += len(b) * int(m)
        if m >= cap:
            at_cap += 1
    n = int(len(clipped))
    return {
        "cap": int(cap), "batch_size": batch_size, "mega": mega,
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


def compute_token_stats(tokenizer, pool_df, info_df, train_df, include_pool: bool,
                        split_col: str) -> dict:
    t0 = time.perf_counter()
    lens_info = tokenize_lengths(tokenizer, info_df[TEXT_COL])
    out = {"tokenizer": tokenizer.name_or_path, "add_special_tokens": True,
           "truncation": False, "distributions": {}, "cost": {}}
    out["distributions"]["informativos"] = _dist(lens_info)
    if include_pool:
        out["distributions"]["pool"] = _dist(tokenize_lengths(tokenizer, pool_df[TEXT_COL]))
    info_rids = set(info_df["rid"].tolist())
    if set(train_df["rid"].tolist()).issubset(info_rids):
        pos = {r: i for i, r in enumerate(info_df["rid"].tolist())}
        lens_train = np.array([lens_info[pos[r]] for r in train_df["rid"]],
                              dtype=np.int64)
    else:
        lens_train = tokenize_lengths(tokenizer, train_df[TEXT_COL])
    out["distributions"]["train"] = _dist(lens_train)
    out["distributions"]["train"]["split_col"] = split_col
    for cap in (128, 192, 256):
        out["cost"][str(cap)] = _cap_cost(lens_train, cap)
    out["cap_tokens_pagos_por_epoca"] = {
        str(c): out["cost"][str(c)]["tokens_pagos_por_epoca"] for c in (128, 192, 256)}
    base = out["cost"]["192"]["tokens_pagos_por_epoca"]
    out["razao_256_192"] = round(
        out["cost"]["256"]["tokens_pagos_por_epoca"] / base, 4) if base else None
    out["assumptions"] = {
        "batching": "length_grouped_batches(batch=32, mega=50, seed=42), comprimentos truncados no cap",
        "tokens_pagos": "sum(len(batch) * max(len_no_batch))",
    }
    out["tokenize_seconds_total"] = round(time.perf_counter() - t0, 2)
    return out


# ------------------------------------------------------------------ losses
def build_sample_weights(train_df: pd.DataFrame, args, device) -> tuple:
    n = len(train_df)
    y = train_df["target"].to_numpy().astype(int)
    w = np.ones(n, dtype=np.float64)
    info = {}
    if args.class_weights:
        counts = np.bincount(y, minlength=2).astype(float)
        cw = counts.sum() / (2.0 * counts)
        w *= cw[y]
        info["class_weights"] = {"true": float(cw[0]), "fake": float(cw[1])}
        print(f"[E5] class_weights ativos: true={cw[0]:.3f} fake={cw[1]:.3f} "
              f"(degrada a calibracao; ablate/medido apenas)")
    if args.dfr_weights == "cell":
        wd = group_balanced_weights(train_df)
        eff_prior = float(wd[y == 1].sum() / wd.sum())
        before = {"min": float(wd.min()), "max": float(wd.max()),
                  "mean": float(wd.mean()), "prior_efetivo_fake": eff_prior}
        wd = np.minimum(wd, args.weight_clip)
        wd = wd * (len(wd) / wd.sum())
        after = {"min": float(wd.min()), "max": float(wd.max()),
                 "mean": float(wd.mean()),
                 "prior_efetivo_fake": float(wd[y == 1].sum() / wd.sum())}
        print(f"[E5] DFR cell: antes clip min={before['min']:.4f} "
              f"max={before['max']:.2f} prior_fake_efetivo={before['prior_efetivo_fake']:.4f} | "
              f"depois clip({args.weight_clip}): min={after['min']:.4f} "
              f"max={after['max']:.2f} prior_fake_efetivo={after['prior_efetivo_fake']:.4f}")
        w *= wd
        info["dfr_weights"] = {"before_clip": before, "after_clip": after}
    if args.class_weights or args.dfr_weights == "cell":
        if w.sum() <= 0:
            raise ValueError("pesos por amostra degenerados")
        w = w * (len(w) / w.sum())
        info["sample_weight"] = {"min": float(w.min()), "max": float(w.max()),
                                 "mean": float(w.mean())}
        return torch.tensor(w, dtype=torch.float32, device=device), info
    return None, info


# ----------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--prepare", action="store_true",
                    help="delega para models.v6.prepare_v6 e sai (local)")
    ap.add_argument("--sanitized", default="data/FakenewsBR_sanitized_v6.csv")
    ap.add_argument("--labels", default="data/FakenewsBR_v6_labels.csv")
    ap.add_argument("--provenance", default="data/FakenewsBR_v6_provenance.csv")
    ap.add_argument("--out-dir", default="models/v6/processed",
                    help="out-dir do --prepare (nao confundir com --out)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--split-col", default="full_iid")
    ap.add_argument("--eval-col", default="full_iid")
    ap.add_argument("--eval-value", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--drive-out", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--min-delta", type=float, default=0.005)
    ap.add_argument("--best-metric", choices=["worst_group", "macro_f1"],
                    default="worst_group")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--freeze-layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dfr-weights", choices=["off", "cell"], default="cell")
    ap.add_argument("--weight-clip", type=float, default=25.0)
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--mask-entities", action="store_true")
    ap.add_argument("--drop-tiers", default="")
    ap.add_argument("--amp", choices=["auto", "off", "fp16", "bf16"], default="auto")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--calib-col", default="val_calib")
    ap.add_argument("--threshold-primary", type=float, default=0.5)
    ap.add_argument("--save-every-epoch", choices=["both", "last", "best", "none"],
                    default="both")
    ap.add_argument("--predict-all", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--token-stats", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=1024)
    ap.add_argument("--ablate", choices=["none", "fullpool", "informative", "ood",
                                         "dfr", "maxlen256",
                                         "freeze4", "freeze0", "classweights", "llmoff"],
                    default="none")
    ap.add_argument("--probe-max-length", type=int, nargs="?", const=256,
                    default=None,
                    help="probe opt-in de VRAM; sem a flag nao roda")
    ap.add_argument("--check-vendor", action="store_true")
    return ap


ABLATE_PRESETS = {
    # default v6 ja e full_iid + DFR cell clip 25 + 2 epocas; `fullpool` fica
    # como alias vazio para compatibilidade com os comandos v4.
    "none": {},
    "fullpool": {},
    # abalacao de referencia: informativos (bal_iid) sem pesos, como o A0 v4
    "informative": {"split_col": "bal_iid", "dfr_weights": "off",
                    "eval_col": "bal_iid"},
    # OOD WhatsApp: teste = canal whatsapp
    "ood": {"split_col": "ood_wa", "eval_col": "ood_wa"},
    "dfr": {"dfr_weights": "cell", "weight_clip": 25.0},
    "maxlen256": {"max_length": 256},
    "freeze4": {"freeze_layers": 4},
    "freeze0": {"freeze_layers": 0, "batch_size": 16},
    "classweights": {"class_weights": True, "dfr_weights": "off"},
    "llmoff": {"dfr_weights": "cell", "drop_tiers": "llm_local,corroborated"},
}


def apply_ablate(args, argv) -> argparse.Namespace:
    explicit = set()
    for tok in argv:
        t = tok.split("=")[0]
        if t.startswith("--"):
            explicit.add(t[2:].replace("-", "_"))
    preset = ABLATE_PRESETS.get(args.ablate, {})
    for key, val in preset.items():
        if key not in explicit:
            setattr(args, key, val)
    return args


# ------------------------------------------------------------- check-vendor
def _max_diff(a, b) -> float:
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return float("inf")
        if a.dtype.kind in "fi" and b.dtype.kind in "fi":
            av, bv = a.astype(float), b.astype(float)
            both_nan = np.isnan(av) & np.isnan(bv)
            dd = np.abs(np.where(both_nan, 0.0, av - bv))
            return float(np.nanmax(dd)) if dd.size else 0.0
        return 0.0 if np.array_equal(a, b) else float("inf")
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max([_max_diff(a[k], b[k]) for k in a], default=0.0)
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            return float("inf")
        d = 0.0
        for c in a.columns:
            if a[c].dtype.kind in "fi" and b[c].dtype.kind in "fi":
                av, bv = a[c].to_numpy(float), b[c].to_numpy(float)
                both_nan = np.isnan(av) & np.isnan(bv)
                dd = np.abs(np.where(both_nan, 0.0, av - bv))
                d = max(d, float(np.nanmax(dd)) if dd.size else 0.0)
            else:
                if not a[c].astype(str).equals(b[c].astype(str)):
                    return float("inf")
        return d
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max([_max_diff(x, y) for x, y in zip(a, b)], default=0.0)
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return 0.0
        return float(abs(float(a) - float(b)))
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return float(abs(int(a) - int(b)))
    if isinstance(a, bool) != isinstance(b, bool):
        return float("inf")
    if a == b:
        return 0.0
    try:
        return float(abs(float(a) - float(b)))
    except Exception:
        return float("inf")


def _vendored_eval_sha() -> str:
    return "08d1acb0b38627c0251f282f8bc68f315c4ecf90a970bcd3f6d0e9f5a8211557"


def _vendored_data_sha() -> str:
    return "cd7bb66399124faa0a4b3e7bd6261a6ced9c9f0f37c44d8b0c8f5b627a32b9e4"


def run_check_vendor() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    try:
        from models import evaluate as EV
        import importlib
        data_mod = importlib.import_module("models.data")
    except Exception as e:
        print(f"[check-vendor] vendor check e local: nao consegui importar "
              f"models.evaluate/models.data ({e})")
        return 1

    src_eval = root / "models" / "evaluate.py"
    src_data = root / "models" / "data.py"
    want_eval, want_data = _vendored_eval_sha(), _vendored_data_sha()
    got_eval = sha256_file(src_eval) if src_eval.exists() else "ausente"
    got_data = sha256_file(src_data) if src_data.exists() else "ausente"
    print(f"[check-vendor] sha256 evaluate.py embutido={want_eval}")
    print(f"[check-vendor] sha256 evaluate.py em disco ={got_eval}"
          f"{'  OK' if got_eval == want_eval else '  <<< DIVERGE'}")
    print(f"[check-vendor] sha256 data.py     embutido={want_data}")
    print(f"[check-vendor] sha256 data.py     em disco ={got_data}"
          f"{'  OK' if got_data == want_data else '  <<< DIVERGE'}")

    rng = np.random.default_rng(42)
    n = 600
    y = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n - n // 2, dtype=int)])
    z = rng.normal(size=n) * 1.7 + 1.4 * (y - 0.5) * 2.0
    logits64 = np.column_stack([np.zeros(n), z]).astype(np.float64)
    logits32 = logits64.astype(np.float32)
    df = pd.DataFrame({
        "group": rng.choice(["a", "b", "c", "d"], n),
        "channel": rng.choice(["portal", "whatsapp", "covid"], n),
        "publisher": rng.choice(["p1", "p2", "p3"], n),
        "era": rng.choice([">=2023", "2018-2022", "<=2017"], n),
        "rating_class": rng.choice(["hard", "false_pure", "true_rating", "other"], n),
        "is_ptpt": rng.random(n) < 0.3,
        "is_balanced_group": rng.random(n) < 0.8,
        "label_tier": rng.choice(["v1", "checker", "llm_local"], n),
        "target": y,
    })

    failures = []
    for tag, lg, tol in (("float64", logits64, 0.0), ("float32", logits32, 1e-9)):
        p_mine = apply_platt(lg, 1.0, 0.0)
        p_orig = EV.apply_platt(lg, 1.0, 0.0)
        d = _max_diff(p_mine, p_orig)
        ok = d <= tol
        if not ok:
            failures.append(f"apply_platt {tag} diff={d}")
        print(f"[check-vendor] apply_platt {tag}: maxdiff={d} "
              f"({'OK' if ok else 'FALHOU tol=%s' % tol})")
        a_mine, b_mine = fit_platt(lg, y)
        a_orig, b_orig = EV.fit_platt(lg, y)
        d = max(abs(a_mine - a_orig), abs(b_mine - b_orig))
        ok = d <= (0.0 if tag == "float64" else 1e-9)
        if not ok:
            failures.append(f"fit_platt {tag} diff={d}")
        print(f"[check-vendor] fit_platt {tag}: a={a_mine:.10f} b={b_mine:.10f} "
              f"maxdiff={d} ({'OK' if ok else 'FALHOU'})")
        m_mine = core_metrics(y, p_mine, 0.5)
        m_orig = EV.core_metrics(y, p_orig, 0.5)
        d = _max_diff(m_mine, m_orig)
        ok = d <= (0.0 if tag == "float64" else 1e-9)
        if not ok:
            failures.append(f"core_metrics {tag} diff={d}")
        print(f"[check-vendor] core_metrics {tag}: maxdiff={d} "
              f"({'OK' if ok else 'FALHOU'})")
        e_mine = expected_calibration_error(p_mine, y)
        e_orig = EV.expected_calibration_error(p_orig, y)
        d = abs(e_mine - e_orig)
        ok = d <= (0.0 if tag == "float64" else 1e-9)
        if not ok:
            failures.append(f"ece {tag} diff={d}")
        print(f"[check-vendor] ece {tag}: maxdiff={d} ({'OK' if ok else 'FALHOU'})")

    p = apply_platt(logits64, 1.0, 0.0)
    w_mine = worst_group_f1(df, y, p)
    w_orig = EV.worst_group_f1(df, y, p)
    ok = (np.isnan(w_mine) and np.isnan(w_orig)) or w_mine == w_orig
    if not ok:
        failures.append(f"worst_group_f1 mine={w_mine} orig={w_orig}")
    print(f"[check-vendor] worst_group_f1: mine={w_mine} orig={w_orig} "
          f"({'OK' if ok else 'FALHOU'})")

    g_mine = per_group(df, y, p)
    g_orig = EV.per_group(df, y, p)
    d = _max_diff(g_mine, g_orig)
    if d > 1e-9:
        failures.append(f"per_group diff={d}")
    print(f"[check-vendor] per_group: maxdiff={d} ({'OK' if d <= 1e-9 else 'FALHOU'})")

    r_mine = reliability_table(p, y)
    r_orig = EV.reliability_table(p, y)
    d = _max_diff(r_mine, r_orig)
    if d > 1e-9:
        failures.append(f"reliability_table diff={d}")
    print(f"[check-vendor] reliability_table: maxdiff={d} "
          f"({'OK' if d <= 1e-9 else 'FALHOU'})")

    cap_mine, cap_orig = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(cap_mine):
        rep_mine = report(df, y, p, "sintetico", 0.5)
    with contextlib.redirect_stdout(cap_orig):
        rep_orig = EV.report(df, y, p, "sintetico", 0.5)
    d = _max_diff(rep_mine, rep_orig)
    if d > 1e-9:
        failures.append(f"report diff={d}")
    same_print = cap_mine.getvalue() == cap_orig.getvalue()
    if not same_print:
        failures.append("report stdout difere")
    print(f"[check-vendor] report: maxdiff={d} ({'OK' if d <= 1e-9 else 'FALHOU'})")
    print(f"[check-vendor] report stdout identico: {same_print}")

    wb_mine = group_balanced_weights(df)
    wb_orig = data_mod.group_balanced_weights(df)
    d = _max_diff(wb_mine, wb_orig)
    if d != 0.0:
        failures.append(f"group_balanced_weights diff={d}")
    print(f"[check-vendor] group_balanced_weights: maxdiff={d} "
          f"({'OK' if d == 0.0 else 'FALHOU'})")

    if got_eval != want_eval:
        failures.append("sha256 evaluate.py divergente")
    if got_data != want_data:
        failures.append("sha256 data.py divergente")
    if failures:
        print(f"[check-vendor] RESULTADO: FALHOU ({len(failures)})")
        for f in failures:
            print("  -", f)
        return 1
    print("[check-vendor] RESULTADO: PASSOU (diferencas 0 em float64 / <=1e-9 em float32)")
    return 0


# ------------------------------------------------------------------ prepare
def run_prepare(args) -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    try:
        from models.v6 import prepare_v6
    except Exception as e:
        print("[prepare] nao consegui importar models.v6.prepare_v6. No Colab, "
              "gere o parquet localmente e envie v6_pool.parquet/v6_splits.parquet. "
              f"({e})")
        return 2
    argv = ["--sanitized", args.sanitized, "--labels", args.labels,
            "--provenance", args.provenance, "--out-dir", args.out_dir,
            "--seed", str(args.seed)]
    if args.token_stats:
        argv.append("--token-stats")
    return int(prepare_v6.main(argv))


# --------------------------------------------------------------- smoke utils
def strat_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    strat = df["label"].astype(str)
    _, sub = train_test_split(df, test_size=n, random_state=seed, stratify=strat)
    return sub.reset_index(drop=True)


def threshold_grid(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.951, 0.01):
        f = f1_score(y, (p >= thr).astype(int), average="macro")
        if f > best_f1:
            best_f1, best_thr = float(f), float(thr)
    return best_thr, best_f1


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    args = apply_ablate(args, argv)

    if args.check_vendor:
        return run_check_vendor()
    if args.prepare:
        return run_prepare(args)

    if args.data is None or args.splits is None:
        print("[ERRO] --data e --splits sao obrigatorios (fora de --check-vendor/--prepare)")
        return 2

    script_sha = sha256_file(Path(__file__).resolve())
    t_start = time.perf_counter()
    timing = {"tokenize_s": 0.0, "train_s": 0.0, "eval_s": 0.0, "total_s": 0.0}

    # ---------------------------------------------------------------- E0
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("[ERRO] --device cuda pedido mas CUDA indisponivel")
            return 2
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    capability = list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None
    use_amp = False
    amp_dtype = None
    scaler_enabled = False
    if device.type == "cuda" and args.amp != "off":
        if args.amp == "fp16":
            amp_dtype = torch.float16
        elif args.amp == "bf16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        use_amp = True
        scaler_enabled = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled) \
        if device.type == "cuda" else None
    set_seed(args.seed)
    import transformers
    print(f"[E0] device={device} amp={args.amp} dtype={amp_dtype} "
          f"scaler={scaler_enabled} torch={torch.__version__} "
          f"transformers={transformers.__version__} cuda={torch.version.cuda}")
    if device.type == "cuda":
        print(f"[E0] gpu={torch.cuda.get_device_name(0)} capability={capability}")
    print(f"[E0] seed={args.seed} cudnn.deterministic=True benchmark=False")

    # run id / out
    if args.run_id is None:
        dfr = "on" if args.dfr_weights != "off" else "off"
        args.run_id = RUN_ID_TEMPLATE.format(
            split_col=args.split_col, max_length=args.max_length,
            freeze=args.freeze_layers, dfr=dfr, seed=args.seed)
        if args.ablate != "none":
            args.run_id += f"_ablate-{args.ablate}"
        if args.smoke:
            args.run_id += "_smoke"
    if args.out is None:
        args.out = str(Path("models/v6/artifacts") / args.run_id)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- E2
    print(f"[E2] data={args.data} splits={args.splits} split_col={args.split_col}")
    if not Path(args.data).exists() or not Path(args.splits).exists():
        print("[ERRO] parquets nao encontrados")
        return 2
    pool = pd.read_parquet(args.data)
    splits = pd.read_parquet(args.splits)
    pool["rid"] = pool["rid"].astype("int64")
    splits["rid"] = splits["rid"].astype("int64")
    if "target" not in pool.columns:
        if "label" not in pool.columns:
            print("[ERRO] parquet sem coluna 'label'/'target'")
            return 2
        pool["target"] = (pool["label"].astype(str) == "fake").astype(int)

    def resolve_split_col(name):
        if name in splits.columns:
            return name
        cand = f"split_{name}"
        if cand in splits.columns:
            return cand
        return None

    split_col_full = resolve_split_col(args.split_col)
    eval_col_full = resolve_split_col(args.eval_col)
    for name, full in ((args.split_col, split_col_full), (args.eval_col, eval_col_full)):
        if full is None:
            print(f"[ERRO] coluna de split '{name}' nao existe em "
                  f"{args.splits}: {list(splits.columns)}")
            return 2
    missing = set(splits["rid"]) - set(pool["rid"])
    if missing:
        print(f"[ERRO] {len(missing)} rids dos splits nao existem no pool")
        return 2
    keep_cols = ["rid"] + sorted({split_col_full, eval_col_full})
    pool = pool.merge(splits[keep_cols], on="rid", how="left", validate="one_to_one")
    pool = pool.rename(columns={split_col_full: args.split_col})
    if args.eval_col != args.split_col:
        if eval_col_full in pool.columns:
            pool = pool.rename(columns={eval_col_full: args.eval_col})
        else:
            pool[args.eval_col] = pool[args.split_col]

    # defensivo: U+FFFD nunca treina/avalia (prepare marcou has_ufffd).
    # V7: registra quantas linhas sairam de cada conjunto (n_test_bruto vs n_test).
    n_pool_bruto = int(len(pool))
    ufffd_removidas: dict = {}
    if "has_ufffd" in pool.columns:
        u = pool["has_ufffd"].to_numpy().astype(bool)
        if u.any():
            masks = {
                "train": (pool[args.split_col] == "train").to_numpy(),
                "val_sel": (pool[args.split_col] == "val_sel").to_numpy(),
                "val_calib": (pool[args.split_col] == "val_calib").to_numpy(),
                f"eval:{args.eval_col}={args.eval_value}":
                    (pool[args.eval_col] == args.eval_value).to_numpy(),
            }
            ufffd_removidas = {k: int((u & m).sum()) for k, m in masks.items()}
            print(f"[E2] removendo {int(u.sum())} linhas has_ufffd (defensivo): "
                  f"{ufffd_removidas}")
            pool = pool[~u].reset_index(drop=True)

    def take(col, val):
        return pool[pool[col] == val].reset_index(drop=True)

    train_df = take(args.split_col, "train")
    val_sel = take(args.split_col, "val_sel")
    val_calib = take(args.split_col, "val_calib")
    eval_df = take(args.eval_col, args.eval_value)
    n_eval_bruto = len(eval_df) + ufffd_removidas.get(
        f"eval:{args.eval_col}={args.eval_value}", 0)

    if args.drop_tiers:
        tiers = {t.strip() for t in args.drop_tiers.split(",") if t.strip()}
        before = len(train_df)
        train_df = train_df[~train_df["label_tier"].isin(tiers)].reset_index(drop=True)
        print(f"[E2] drop_tiers={sorted(tiers)}: {before} -> {len(train_df)} no treino")

    for d in (train_df, val_sel, val_calib, eval_df):
        if len(d) == 0:
            print("[ERRO] conjunto vazio apos selecao; verifique split/colunas")
            return 2

    y_tr = train_df["target"].to_numpy().astype(int)
    print("[E2] contagens (n, fake%):")
    for name, d in (("train", train_df), ("val_sel", val_sel),
                    ("val_calib", val_calib),
                    (f"eval:{args.eval_col}={args.eval_value}", eval_df)):
        pct = (d["target"].mean() * 100) if "target" in d.columns else float("nan")
        print(f"      {name:<32} n={len(d):>6} fake={pct:5.2f}%")

    # controle de vazamento: nenhum rid nem texto normalizado compartilhado
    # entre treino/val e o teste de avaliacao (o prepare ja forca duplicatas
    # exatas ao mesmo lado; aqui e a verificacao defensiva do trainer).
    inter = set(train_df["rid"]) & set(eval_df["rid"])
    print(f"[E2] interseccao rid treino x teste ({args.eval_col}={args.eval_value}): "
          f"{len(inter)}")
    if inter:
        print("[ERRO] rid compartilhado entre treino e teste; o split de treino "
              "contem linhas do conjunto de avaliacao (use --eval-col igual a "
              "--split-col; no v6 o full_iid tem teste proprio)")
        return 2
    learn_keys = set(train_df[TEXT_COL].map(text_key))
    learn_keys |= set(val_sel[TEXT_COL].map(text_key))
    learn_keys |= set(val_calib[TEXT_COL].map(text_key))
    test_keys = set(eval_df[TEXT_COL].map(text_key))
    n_leak = len(learn_keys & test_keys)
    print(f"[E2] duplicatas texto-normalizado treino/val x teste: {n_leak}")
    if n_leak:
        print("[ERRO] texto normalizado compartilhado entre treino/val e teste; "
              "a divisao do prepare_v6 deve manter duplicatas no mesmo lado")
        return 2

    # copia de avaliacao: is_ptpt_rule -> is_ptpt
    for d in (val_sel, val_calib, eval_df):
        if "is_ptpt_rule" in d.columns:
            d.rename(columns={"is_ptpt_rule": "is_ptpt"}, inplace=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    token_stats = None
    if args.token_stats or args.smoke:
        print("[E1] token stats (tokenizer em cache local/sessao)")
        info_df = pool[pool["is_balanced_group"]].reset_index(drop=True)
        token_stats = compute_token_stats(tok, pool, info_df, train_df,
                                          include_pool=not args.smoke,
                                          split_col=args.split_col)
        d = token_stats["distributions"]["informativos"]
        print(f"[E1] informativos: media={d['mean']} p95={d['p95']} "
              f">192={d['pct_gt_192']}% | razao_256_192={token_stats['razao_256_192']}")
        save_json(token_stats, out_dir / "token_stats.json")
        if args.token_stats and not args.smoke:
            print(f"[E1] token_stats.json gravado em {out_dir}; "
                  "saindo (remova --token-stats para treinar)")
            return 0

    # smoke: subconjuntos estratificados + overrides
    if args.smoke:
        print(f"[E8] SMOKE n={args.smoke_n} (subconjunto estratificado por rotulo)")
        train_df = strat_subset(train_df, args.smoke_n, args.seed)
        val_sel = strat_subset(val_sel, args.smoke_n, args.seed)
        val_calib = strat_subset(val_calib, args.smoke_n, args.seed)
        eval_df = strat_subset(eval_df, args.smoke_n, args.seed)
        y_tr = train_df["target"].to_numpy().astype(int)
        args.max_length = min(128, args.max_length)
        args.batch_size = min(8, args.batch_size)
        args.eval_batch_size = 32
        args.epochs = 1
        args.predict_all = False
        print(f"[E8] overrides: max_length={args.max_length} "
              f"batch_size={args.batch_size} eval_batch_size={args.eval_batch_size} "
              "epochs=1 predict_all=False")

    # ---------------------------------------------------------------- E3
    t_tok = time.perf_counter()
    text_fn = mask_entities_text if args.mask_entities else (lambda x: x)
    ds_tr = TextDS((text_fn(t) for t in train_df[TEXT_COL]), y_tr, tok, args.max_length)
    ds_vs = TextDS((text_fn(t) for t in val_sel[TEXT_COL]), val_sel["target"], tok, args.max_length)
    ds_vc = TextDS((text_fn(t) for t in val_calib[TEXT_COL]), val_calib["target"], tok, args.max_length)
    ds_ev = TextDS((text_fn(t) for t in eval_df[TEXT_COL]), eval_df["target"], tok, args.max_length)
    collate = DataCollatorWithPadding(tok)
    train_collate = TrainCollator(collate)
    eval_collate = EvalCollator(collate)

    sampler = LengthGroupedBatchSampler(ds_tr.lengths(), args.batch_size, args.seed)
    loader_kwargs = dict(batch_sampler=sampler, collate_fn=train_collate,
                         num_workers=args.workers, pin_memory=(device.type == "cuda"))
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(ds_tr, **loader_kwargs)
    timing["tokenize_s"] = time.perf_counter() - t_tok
    print(f"[E3] tokenizacao em {timing['tokenize_s']:.1f}s; "
          f"{len(ds_tr)} treino / {len(ds_vs)} val_sel / {len(ds_vc)} val_calib / "
          f"{len(ds_ev)} eval; workers={args.workers} "
          f"pin_memory={loader_kwargs['pin_memory']}")
    lens_tr = ds_tr.lengths()
    print(f"[E3] tokens treino: media={np.mean(lens_tr):.1f} "
          f"p95={np.percentile(lens_tr, 95):.0f} max={np.max(lens_tr)}")

    # ---------------------------------------------------------------- E4
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def freeze_layers_now(m, n_layers):
        if not n_layers:
            return
        base = m.base_model
        for p in base.embeddings.parameters():
            p.requires_grad = False
        for layer in base.encoder.layer[:n_layers]:
            for p in layer.parameters():
                p.requires_grad = False

    freeze_layers_now(model, args.freeze_layers)
    live = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[E4] congeladas embeddings + {args.freeze_layers} camadas "
          f"({live/1e6:.1f}M treinaveis)")
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]

    def build_optimizer(m):
        decay, no_decay = [], []
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if any(k in name for k in ("bias", "LayerNorm.weight"))
             else decay).append(p)
        return AdamW([{"params": decay, "weight_decay": args.weight_decay},
                      {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    opt = build_optimizer(model)
    effective = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(ds_tr) / effective)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_frac * total_steps)
    sched = get_scheduler("linear", opt, num_warmup_steps=warmup_steps,
                          num_training_steps=total_steps)
    print(f"[E4] steps/epoca={steps_per_epoch} total={total_steps} "
          f"warmup={warmup_steps} efetivo={effective}")

    sample_w, loss_info = build_sample_weights(train_df, args, device)
    ce_none = nn.CrossEntropyLoss(reduction="none")
    ce_mean = nn.CrossEntropyLoss()

    run_config = {
        "run_id": args.run_id,
        "created_utc": utcnow(),
        "script_sha256": script_sha,
        "data_sha256": sha256_file(Path(args.data)),
        "splits_sha256": sha256_file(Path(args.splits)),
        "split_col": args.split_col,
        "eval_col": args.eval_col,
        "eval_value": args.eval_value,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "freeze_layers": args.freeze_layers,
        "model": args.model,
        "seed": args.seed,
        "dfr_weights": args.dfr_weights,
        "class_weights": bool(args.class_weights),
        "weight_clip": args.weight_clip,
        "drop_tiers": args.drop_tiers,
        "mask_entities": bool(args.mask_entities),
        "smoke": bool(args.smoke),
        "args": vars(args),
    }

    # probe de VRAM (E9) -- opt-in, antes de treinar (V9: em CPU sai sem treinar)
    if args.probe_max_length is not None:
        if device.type != "cuda":
            print("[E9] --probe-max-length requer CUDA; em CPU nao ha VRAM a "
                  "medir. Saindo sem treinar (codigo 2); remova a flag para "
                  "treinar na CPU.")
            return 2
        else:
            torch.cuda.reset_peak_memory_stats()
            probe_ids = torch.randint(0, 30000, (args.batch_size, args.probe_max_length),
                                      device=device)
            probe_mask = torch.ones_like(probe_ids)
            probe_lbl = torch.randint(0, 2, (args.batch_size,), device=device)
            model.train()
            ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp \
                else contextlib.nullcontext()
            with ctx:
                probe_logits = model(input_ids=probe_ids, attention_mask=probe_mask).logits
            probe_loss = ce_mean(probe_logits, probe_lbl)
            if scaler_enabled:
                scaler.scale(probe_loss).backward()
            else:
                probe_loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[E9] probe {args.batch_size}x{args.probe_max_length}: pico={peak:.2f} GB "
                  f"(<=13.5? {'SIM' if peak <= 13.5 else 'NAO'})")
            if peak > 13.5:
                print("[ERRO] probe excedeu 13.5 GB; nao rodar esse cap nesse batch")
                return 2
            model.zero_grad(set_to_none=True)
            return 0

    # ---------------------------------------------------------------- E7
    start_epoch = 0
    start_global_step = 0
    history: list = []
    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    if args.resume:
        out_last = out_dir / "last"
        drive_base = Path(args.drive_out) if args.drive_out else None
        drive_last = (drive_base / "last") if drive_base else None

        def _last_state_key(d: Path):
            sp = d / "trainer_state.json"
            if sp.exists():
                try:
                    st = json.loads(sp.read_text(encoding="utf-8"))
                    return (int(st.get("epoch", 0)), int(st.get("global_step", 0)))
                except Exception as e:
                    print(f"[E7] aviso: {sp} ilegivel ({e}); sem epoch/global_step")
            return (-1, -1)

        chosen = None
        if out_last.exists() and drive_last is not None and drive_last.exists():
            ko, kd = _last_state_key(out_last), _last_state_key(drive_last)
            if kd > ko:
                print(f"[E7] last/ do Drive mais recente (drive={kd}, out={ko}); "
                      "preferindo o Drive")
                chosen = drive_last
            else:
                chosen = out_last
                print(f"[E7] last/ mais recente em --out (out={ko}, drive={kd}); "
                      "preferindo o local")
        elif out_last.exists():
            chosen = out_last
        elif drive_last is not None and drive_last.exists():
            chosen = drive_last
            print(f"[E7] last/ ausente em --out; restaurando de {drive_last}")
        if chosen is None:
            print("[E7] aviso: last/ nao existe no --out nem no --drive-out; "
                  "comecando do zero")
        else:
            if chosen != out_last:
                atomic_copy_tree(chosen, out_last)
                print(f"[E7] copiado {chosen} -> {out_last}")
                if drive_base is not None:
                    rc = drive_base / "run_config.json"
                    if rc.exists() and not (out_dir / "run_config.json").exists():
                        atomic_copy_file(rc, out_dir / "run_config.json")
                    best_drive = drive_base / "best"
                    if best_drive.exists() and not (out_dir / "best").exists():
                        atomic_copy_tree(best_drive, out_dir / "best")
            last_dir = out_last
            cfg_path = out_dir / "run_config.json"
            if not cfg_path.exists() and drive_base is not None \
                    and (drive_base / "run_config.json").exists():
                cfg_path = drive_base / "run_config.json"
            compat = ["data_sha256", "splits_sha256", "split_col", "eval_col",
                      "eval_value", "max_length", "batch_size", "epochs",
                      "freeze_layers", "model", "seed", "dfr_weights",
                      "weight_clip", "class_weights", "drop_tiers",
                      "mask_entities", "smoke"]
            diffs = []
            if cfg_path.exists():
                old = json.loads(cfg_path.read_text(encoding="utf-8"))
                for k in compat:
                    if old.get(k) != run_config.get(k):
                        diffs.append(f"{k}: {old.get(k)!r} -> {run_config.get(k)!r}")
            else:
                diffs = ["run_config.json ausente"]
            if diffs and not args.force_resume:
                print("[E7] ERRO: run_config incompativel com --resume:")
                for d in diffs:
                    print("      -", d)
                print("      use --force-resume para continuar mesmo assim")
                return 2
            if diffs:
                print("[E7] aviso: --force-resume ignorando diferencas de config:")
                for d in diffs:
                    print("      -", d)
            model = AutoModelForSequenceClassification.from_pretrained(last_dir, num_labels=2)
            freeze_layers_now(model, args.freeze_layers)
            model.to(device)
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = build_optimizer(model)
            opt_path = last_dir / "optimizer.pt"
            if opt_path.exists():
                opt.load_state_dict(torch.load(opt_path, map_location="cpu",
                                               weights_only=False))
            sched_path = last_dir / "scheduler.pt"
            if sched_path.exists():
                sched.load_state_dict(torch.load(sched_path, map_location="cpu",
                                                 weights_only=False))
            scaler_path = last_dir / "scaler.pt"
            if scaler is not None and scaler_path.exists():
                scaler.load_state_dict(torch.load(scaler_path, map_location="cpu",
                                                  weights_only=False))
            state_path = last_dir / "trainer_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                start_epoch = int(state.get("epoch", 0))
                start_global_step = int(state.get("global_step", 0))
                history = state.get("history", [])
                best_score = float(state.get("best_score", -float("inf")))
                best_epoch = int(state.get("best_epoch", 0))
                bad_epochs = int(state.get("bad_epochs", 0))
            rng_path = last_dir / "rng.pt"
            if rng_path.exists():
                rng = torch.load(rng_path, map_location="cpu", weights_only=False)
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"])
                if device.type == "cuda" and rng.get("cuda"):
                    torch.cuda.set_rng_state_all(rng["cuda"])
            print(f"[E7] retomando de {last_dir}: epoca {start_epoch}/{args.epochs} "
                  f"(global_step={start_global_step}, best={best_score:.4f} @ ep {best_epoch})")
    save_json(run_config, out_dir / "run_config.json")

    def save_checkpoint(ckpt_dir: Path, with_opt: bool) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)
        if with_opt:
            torch.save(opt.state_dict(), ckpt_dir / "optimizer.pt")
            torch.save(sched.state_dict(), ckpt_dir / "scheduler.pt")
            torch.save(scaler.state_dict() if scaler is not None else {},
                       ckpt_dir / "scaler.pt")
            rng = {"python": random.getstate(), "numpy": np.random.get_state(),
                   "torch": torch.get_rng_state()}
            if device.type == "cuda":
                rng["cuda"] = torch.cuda.get_rng_state_all()
            torch.save(rng, ckpt_dir / "rng.pt")
            state = {"epoch": epoch + 1, "global_step": global_step,
                     "best_score": best_score, "best_epoch": best_epoch,
                     "bad_epochs": bad_epochs, "history": history}
            save_json(state, ckpt_dir / "trainer_state.json")

    def mirror_to_drive() -> None:
        if not args.drive_out or args.smoke:
            return
        drive = Path(args.drive_out)
        for sub in ("last", "best"):
            atomic_copy_tree(out_dir / sub, drive / sub)
        for f in ("history.json", "run_config.json"):
            if (out_dir / f).exists():
                atomic_copy_file(out_dir / f, drive / f)

    # ---------------------------------------------------------------- E5
    global_step = start_global_step
    for epoch in range(start_epoch, args.epochs):
        sampler.epoch = epoch
        model.train()
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loss_sum, tokens_paid, samples = 0.0, 0, 0
        accum = 0
        n_batches = len(train_loader)
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            bidx = batch.pop("idx")
            tokens_paid += int(batch["input_ids"].numel())
            samples += int(labels.numel())
            ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp \
                else contextlib.nullcontext()
            with ctx:
                logits = model(**batch).logits
            if device.type == "cuda" and step == 1:
                peak0 = torch.cuda.max_memory_allocated() / 1e9
                if peak0 > 13.5:
                    print(f"[E9] ERRO: VRAM de pico {peak0:.2f} GB > 13.5 GB no "
                          "primeiro forward. Escada de OOM: (2) batch 16 x cap + "
                          "--grad-accum 2; (3) batch 8 x cap + --grad-accum 4; "
                          "(4) gradient_checkpointing por ultimo.")
                    return 2
            ce = ce_none(logits, labels)
            raw = (ce * sample_w[bidx]).mean() if sample_w is not None else ce.mean()
            loss = raw / args.grad_accum
            if scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum += 1
            if accum == args.grad_accum or step == n_batches:
                if scaler_enabled:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, args.clip)
                if scaler_enabled:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                accum = 0
            loss_sum += float(raw.item())

        train_s = time.perf_counter() - t0
        timing["train_s"] += train_s
        # inferencia de validacao (val_sel)
        t_val = time.perf_counter()
        l_vs = infer(model, ds_vs, eval_collate, args.eval_batch_size, device,
                     use_amp, amp_dtype)
        y_vs = val_sel["target"].to_numpy().astype(int)
        p_vs = sigmoid_z(l_vs)
        m_vs = core_metrics(y_vs, p_vs)
        wg_vs = worst_group_f1(val_sel, y_vs, p_vs)
        val_loss = float(ce_mean(torch.tensor(l_vs), torch.tensor(y_vs)).item())
        val_s = time.perf_counter() - t_val
        score_stop = wg_vs if not np.isnan(wg_vs) else m_vs["macro_f1"]
        score_best = m_vs["macro_f1"] if args.best_metric == "macro_f1" else score_stop
        improved = score_best > best_score + args.min_delta
        if improved:
            best_score, best_epoch, bad_epochs = score_best, epoch + 1, 0
        else:
            bad_epochs += 1
        scaled_scale = float(scaler.get_scale()) if scaler is not None else 0.0
        vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        rec = {
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(n_batches, 1), 5),
            "val_loss": round(val_loss, 5),
            "val_macro_f1": round(m_vs["macro_f1"], 5),
            "val_f1_fake": round(m_vs["f1_fake"], 5),
            "val_worst_group": None if np.isnan(wg_vs) else round(float(wg_vs), 5),
            "val_ece": round(m_vs["ece"], 5),
            "tokens_per_s": round(tokens_paid / max(train_s, 1e-9), 1),
            "samples_per_s": round(samples / max(train_s, 1e-9), 2),
            "lr_last": round(float(sched.get_last_lr()[0]), 9),
            "scaler_scale": scaled_scale,
            "vram_peak_gb": round(vram, 3),
            "seconds": round(train_s + val_s, 1),
            "best_so_far": bool(improved),
        }
        history.append(rec)
        print(f"[E5] epoca {epoch+1}/{args.epochs}: train_loss={rec['train_loss']:.4f} "
              f"val_loss={rec['val_loss']:.4f} macro_f1={rec['val_macro_f1']:.4f} "
              f"worst_group={rec['val_worst_group']} ece={rec['val_ece']:.4f} "
              f"tokens/s={rec['tokens_per_s']:.0f} "
              f"({rec['seconds']:.0f}s, vram={rec['vram_peak_gb']:.2f}GB)"
              + ("  [melhor]" if improved else ""))

        if args.save_every_epoch in ("both", "last"):
            save_checkpoint(out_dir / "last", with_opt=True)
        if improved and args.save_every_epoch in ("both", "best"):
            save_checkpoint(out_dir / "best", with_opt=False)
        save_json(history, out_dir / "history.json")
        save_json(run_config, out_dir / "run_config.json")
        mirror_to_drive()
        if bad_epochs >= args.patience:
            print(f"[E5] early stop: sem melhora > {args.min_delta} por "
                  f"{bad_epochs} epoca(s)")
            break

    # ---------------------------------------------------------------- E6
    t_eval = time.perf_counter()
    best_dir = out_dir / "best"
    eval_model_dir = best_dir if (best_dir / "config.json").exists() else out_dir / "last"
    if not (eval_model_dir / "config.json").exists():
        print("[ERRO] nenhum checkpoint para avaliar (best/ ou last/ ausentes)")
        return 2
    model = AutoModelForSequenceClassification.from_pretrained(eval_model_dir, num_labels=2)
    model.to(device)
    print(f"[E6] avaliando checkpoint {eval_model_dir}")

    l_vs = infer(model, ds_vs, eval_collate, args.eval_batch_size, device, use_amp, amp_dtype)
    l_vc = infer(model, ds_vc, eval_collate, args.eval_batch_size, device, use_amp, amp_dtype)
    l_ev = infer(model, ds_ev, eval_collate, args.eval_batch_size, device, use_amp, amp_dtype)
    y_vs = val_sel["target"].to_numpy().astype(int)
    y_vc = val_calib["target"].to_numpy().astype(int)
    y_ev = eval_df["target"].to_numpy().astype(int)

    # Platt em val_calib, restrito a is_balanced_group quando valido
    cal_mask = np.ones(len(val_calib), dtype=bool)
    cal_note = "val_calib inteira"
    if "is_balanced_group" in val_calib.columns:
        bal = val_calib["is_balanced_group"].to_numpy().astype(bool)
        if bal.sum() >= 50 and len(np.unique(y_vc[bal])) == 2:
            cal_mask = bal
            cal_note = "is_balanced_group"
        else:
            print("[E6] aviso: mascara is_balanced_group indisponivel/pequena; "
                  "calibrando na val_calib inteira")
    a_platt, b_platt = fit_platt(l_vc[cal_mask], y_vc[cal_mask])
    p_cal_vc = apply_platt(l_vc, a_platt, b_platt)
    ece_before = expected_calibration_error(apply_platt(l_vc, 1.0, 0.0), y_vc)
    ece_after = expected_calibration_error(p_cal_vc, y_vc)
    thr_opt, f1_at_thr = threshold_grid(y_vc[cal_mask], p_cal_vc[cal_mask])
    calib_prior = float(y_vc[cal_mask].mean())
    print(f"[E6] Platt em {cal_note} (n={int(cal_mask.sum())}, "
          f"prior_fake={calib_prior:.4f}): a={a_platt:.4f} b={b_platt:+.4f}")
    print(f"[E6] ECE val_calib antes={ece_before:.4f} depois={ece_after:.4f}; "
          f"limiar otimo (grid .05-.95) = {thr_opt:.2f} (macro-F1={f1_at_thr:.4f})")

    p_cal_vs = apply_platt(l_vs, a_platt, b_platt)
    p_cal_ev = apply_platt(l_ev, a_platt, b_platt)
    eval_kind = "ood" if args.eval_col == "ood_wa" else "test"
    test_block = report(eval_df, y_ev, p_cal_ev,
                        f"TESTE[{eval_kind}] | limiar primario {args.threshold_primary}",
                        threshold=args.threshold_primary)
    # V4: o bloco secundario aplica o limiar otimo em TODAS as metricas
    # (inclusive worst_group/per_group). Se o limiar coincidir com o primario,
    # nao ha segundo bloco (evita bloco duplicado identico).
    if abs(thr_opt - args.threshold_primary) > 1e-9:
        test_thr_block = report(eval_df, y_ev, p_cal_ev,
                                f"TESTE[{eval_kind}] | limiar val_opt {thr_opt:.2f}",
                                threshold=thr_opt)
    else:
        test_thr_block = None
        print(f"[E6] limiar val_opt == primario ({thr_opt:.2f}); "
              "test_at_val_threshold omitido")
    ood_block = test_block if eval_kind == "ood" else None

    # predicoes do pool (diagnostico) -- smoke desliga
    y_pool, p_pool = None, None
    if args.predict_all:
        t_pool = time.perf_counter()
        ds_pool = TextDS((text_fn(t) for t in pool[TEXT_COL]), np.zeros(len(pool), dtype=int),
                         tok, args.max_length)
        l_pool = infer(model, ds_pool, eval_collate, args.eval_batch_size, device,
                       use_amp, amp_dtype)
        y_pool = pool["target"].to_numpy().astype(int) if "target" in pool.columns \
            else np.zeros(len(pool), dtype=int)
        p_pool = apply_platt(l_pool, a_platt, b_platt)
        print(f"[E6] predict-all: pool n={len(pool)} em {time.perf_counter()-t_pool:.1f}s")

    # ---------------------------------------------------------------- E6 saidas
    train_rids = set(train_df["rid"].tolist())
    pred_frames = []

    def pred_frame(df, logits, split_name):
        z = _decision(logits)
        frame = pd.DataFrame({
            "rid": df["rid"].to_numpy(),
            "split": split_name,
            "y_true": df["target"].to_numpy().astype(int),
            "logit0": logits[:, 0].astype(float),
            "logit1": logits[:, 1].astype(float),
            "z": z.astype(float),
            "p_raw": 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700))),
            "p_cal": apply_platt(logits, a_platt, b_platt),
        })
        frame["pred_05"] = (frame["p_cal"] >= args.threshold_primary).astype(int)
        frame["pred_val_thr"] = (frame["p_cal"] >= thr_opt).astype(int)
        for col in ("group", "channel", "publisher", "era", "lang_variant",
                    "rating_class", "label_tier", "has_ufffd", "mojibake_flag"):
            frame[col] = df[col].to_numpy() if col in df.columns else None
        frame["is_ptpt_rule"] = df["is_ptpt"].to_numpy() if "is_ptpt" in df.columns else None
        frame["in_training_pool"] = frame["rid"].isin(train_rids)
        return frame

    pred_frames.append(pred_frame(val_sel, l_vs, "val_sel"))
    pred_frames.append(pred_frame(val_calib, l_vc, "val_calib"))
    pred_frames.append(pred_frame(eval_df, l_ev, eval_kind))
    if args.predict_all:
        covered = set(pd.concat(pred_frames)["rid"].tolist())
        pool_diag_mask = ~pool["rid"].isin(covered).to_numpy()
        sub = pool[pool_diag_mask].reset_index(drop=True)
        if len(sub):
            pred_frames.append(pred_frame(sub, l_pool[pool_diag_mask], "pool_diag"))
    predictions = pd.concat(pred_frames, ignore_index=True)
    pred_cols = ["rid", "split", "y_true", "logit0", "logit1", "z", "p_raw", "p_cal",
                 "pred_05", "pred_val_thr", "group", "channel", "publisher", "era",
                 "lang_variant", "is_ptpt_rule", "rating_class", "label_tier",
                 "has_ufffd", "mojibake_flag", "in_training_pool"]
    predictions = predictions[pred_cols]
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    cut_frames = []
    for cut in ("group", "channel", "publisher", "era", "label_tier"):
        tab = per_group(eval_df, y_ev, p_cal_ev, col=cut,
                        threshold=args.threshold_primary)
        if not tab.empty:
            tab = tab.rename(columns={cut: "chave"})
            tab.insert(0, "cut", cut)
            cut_frames.append(tab)
    per_group_df = pd.concat(cut_frames, ignore_index=True) if cut_frames \
        else pd.DataFrame(columns=["cut", "chave"])
    per_group_df.to_csv(out_dir / "per_group.csv", index=False)
    reliability_table(p_cal_ev, y_ev).to_csv(out_dir / "reliability.csv", index=False)

    # V14: prior do treino e prior efetivo da loss (DFR pos-clip) no calibration
    train_prior_fake = float(y_tr.mean())
    eff_prior = loss_info.get("dfr_weights", {}).get("after_clip", {}) \
        .get("prior_efetivo_fake")
    effective_prior_fake = float(eff_prior) if eff_prior is not None \
        else train_prior_fake
    calibration = {
        "platt_a": float(a_platt), "platt_b": float(b_platt),
        "max_length": args.max_length, "model": args.model,
        "mask_entities": bool(args.mask_entities),
        "threshold": float(args.threshold_primary),
        "threshold_val_opt": float(thr_opt),
        "calib_on": args.calib_col, "calib_n": int(cal_mask.sum()),
        "calib_mask": cal_note,
        "calib_prior_fake": calib_prior,
        "train_prior_fake": train_prior_fake,
        "effective_prior_fake": effective_prior_fake,
        "freeze_layers": args.freeze_layers, "seed": args.seed,
        "split_col": args.split_col, "dfr_weights": args.dfr_weights,
        "weight_clip": args.weight_clip, "class_weights": bool(args.class_weights),
        "script_sha256": script_sha, "created_utc": utcnow(),
    }
    save_json(calibration, out_dir / "calibration.json")
    if (out_dir / "best").exists():
        save_json(calibration, out_dir / "best" / "calibration.json")

    cuts = {}
    for cut in ("channel", "publisher", "era", "label_tier"):
        tab = per_group(eval_df, y_ev, p_cal_ev, col=cut,
                        threshold=args.threshold_primary)
        cuts[cut] = jsonable(tab.rename(columns={cut: "chave"})
                             .to_dict(orient="records")) if not tab.empty else []
    if "is_ptpt" in eval_df.columns:
        cuts["lang_variant_rule"] = {
            "n_ptpt": int(eval_df["is_ptpt"].sum()),
            "macro_f1_ptpt": test_block.get("ptpt_macro_f1", None)}
    else:
        cuts["lang_variant_rule"] = {}
    cuts["lang_variant_prov"] = jsonable(
        eval_df.groupby("lang_variant")["target"].agg(["size", "mean"]).reset_index()
        .rename(columns={"size": "n", "mean": "fake_pct"}).to_dict(orient="records")) \
        if "lang_variant" in eval_df.columns else []

    const_diag = []
    if args.predict_all and y_pool is not None:
        for name, idx in pool.groupby("group").indices.items():
            yi, pi = y_pool[idx], p_pool[idx]
            m = core_metrics(yi, pi, args.threshold_primary)
            const_diag.append({
                "group": name, "n": int(len(idx)),
                "fake_pct": round(float(yi.mean() * 100), 2),
                "in_training_pool": bool(pool["rid"].iloc[idx].isin(train_rids).any()),
                "p_cal_mean": round(float(pi.mean()), 4),
                "acc": m["acc"],
                "macro_f1": None if np.isnan(m["macro_f1"]) else m["macro_f1"],
                "f1_fake": m["f1_fake"], "ece": m["ece"],
            })

    timing["eval_s"] = time.perf_counter() - t_eval
    timing["total_s"] = time.perf_counter() - t_start
    metrics = {
        "run_id": args.run_id,
        "created_utc": utcnow(),
        "config": {
            "args": jsonable(vars(args)),
            "ablate": args.ablate,
            "loss_info": jsonable(loss_info),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "script_sha256": script_sha,
        },
        "data": {
            "data_sha256": run_config["data_sha256"],
            "splits_sha256": run_config["splits_sha256"],
            "split_col": args.split_col,
            "eval_col": args.eval_col,
            "eval_value": args.eval_value,
            "n_train": len(train_df), "n_val_sel": len(val_sel),
            "n_val_calib": len(val_calib), "n_test": len(eval_df),
            "n_pool_bruto": n_pool_bruto,
            "n_test_bruto": n_eval_bruto,
            "n_ufffd_removidas": ufffd_removidas,
            "train_fake_pct": round(float(y_tr.mean() * 100), 2),
            "smoke": bool(args.smoke),
        },
        "history": history,
        "best_epoch": best_epoch,
        "calibration": {"platt_a": a_platt, "platt_b": b_platt,
                        "calib_n": int(cal_mask.sum()),
                        "calib_prior_fake": calib_prior,
                        "train_prior_fake": train_prior_fake,
                        "effective_prior_fake": effective_prior_fake,
                        "ece_before": ece_before, "ece_after": ece_after,
                        "threshold": args.threshold_primary,
                        "threshold_val_opt": thr_opt},
        "test": test_block,
        "test_at_val_threshold": test_thr_block,
        "ood": ood_block,
        "cuts": cuts,
        "constant_groups_diagnostic": const_diag,
        "timing": timing,
        "hardware": {"device": str(device),
                     "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                     "capability": capability,
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "cuda": torch.version.cuda},
    }
    save_json(metrics, out_dir / "metrics.json")
    save_json(history, out_dir / "history.json")

    if args.drive_out and not args.smoke:
        drive = Path(args.drive_out)
        for f in ("metrics.json", "calibration.json", "predictions.csv",
                  "per_group.csv", "reliability.csv", "history.json",
                  "run_config.json", "token_stats.json"):
            if (out_dir / f).exists():
                atomic_copy_file(out_dir / f, drive / f)
        for sub in ("best", "last"):
            atomic_copy_tree(out_dir / sub, drive / sub)
        print(f"[E6] artefatos copiados para {drive}")

    # ---------------------------------------------------------------- E8
    if args.smoke:
        tr_rec = history[-1] if history else {}
        tok_s = tr_rec.get("tokens_per_s", 0.0)
        samples_s = tr_rec.get("samples_per_s", 0.0)
        vram = tr_rec.get("vram_peak_gb", 0.0)
        print("\n[E8] === SMOKE: medicao ===")
        print(f"  throughput treino: {samples_s:.2f} amostras/s | {tok_s:.0f} tokens/s pagos")
        print(f"  VRAM pico: {vram:.2f} GB (CPU = 0)")
        print(f"  tempos: tokenizacao={timing['tokenize_s']:.1f}s "
              f"treino={timing['train_s']:.1f}s avaliacao_final={timing['eval_s']:.1f}s "
              f"total={timing['total_s']:.1f}s")
        default_tokens = None
        if token_stats:
            default_tokens = token_stats["cap_tokens_pagos_por_epoca"].get("192")
        print("  projecao T4 (HIPOTESE: speedup T4/CPU em [10, 30]x, NAO medido):")
        if tok_s > 0 and default_tokens:
            for sp in (10, 30):
                eta = default_tokens / (tok_s * sp)
                print(f"    speedup {sp:>2}x -> ~{eta/60:.1f} min/epoca "
                      f"(default {args.split_col} 192: "
                      f"{default_tokens/1e6:.2f}M tokens pagos/epoca; "
                      f"{args.epochs} epocas ~{args.epochs*eta/60:.0f} min)")
        else:
            print("    sem tokens/s medido; projecao indisponivel")
        required = [
            out_dir / "best" / "model.safetensors",
            out_dir / "best" / "config.json",
            out_dir / "best" / "calibration.json",
            out_dir / "last" / "model.safetensors",
            out_dir / "last" / "optimizer.pt",
            out_dir / "last" / "scheduler.pt",
            out_dir / "last" / "scaler.pt",
            out_dir / "last" / "rng.pt",
            out_dir / "last" / "trainer_state.json",
            out_dir / "history.json",
            out_dir / "run_config.json",
            out_dir / "metrics.json",
            out_dir / "calibration.json",
            out_dir / "predictions.csv",
            out_dir / "per_group.csv",
            out_dir / "reliability.csv",
            out_dir / "token_stats.json",
        ]
        missing = [str(p.relative_to(out_dir)) for p in required if not p.exists()]
        if missing:
            print(f"[E8] ERRO: artefatos do contrato ausentes: {missing}")
            return 1
        print(f"[E8] contrato completo: {len(required)} artefatos presentes em {out_dir}")
        print("\n[E8] metricas do smoke (numeros produzidos nesta execucao):")
        print(f"  val_sel worst_group={tr_rec.get('val_worst_group')} "
              f"macro_f1={tr_rec.get('val_macro_f1')} ece={tr_rec.get('val_ece')}")
        print(f"  test acc={test_block.get('acc')} macro_f1={test_block.get('macro_f1')} "
              f"worst_group={test_block.get('worst_group_macro_f1')} "
              f"ece={test_block.get('ece')}")

    print(f"\n[FIM] run {args.run_id}: history={len(history)} epoca(s) "
          f"best_epoch={best_epoch} total={timing['total_s']:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except torch.cuda.OutOfMemoryError:
        print("[E9] CUDA OOM. Escada do PLANO 5.5 (aplicar em ordem): "
              "(2) batch 16 x cap + --grad-accum 2; (3) batch 8 x cap + "
              "--grad-accum 4; (4) model.gradient_checkpointing_enable() por ultimo.")
        sys.exit(2)





