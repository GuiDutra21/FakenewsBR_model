"""autoresearch/experiment.py — UM experimento: 1 config -> budget fixo de steps
-> avaliacao congelada -> resultado.

Portado do espirito de karpathy/autoresearch (https://github.com/karpathy/autoresearch):
budget fixo por experimento, UMA metrica de manchete, tudo medido e logado.
Diferenca: aqui o budget e em steps de otimizacao (deterministico em CPU) e a
metrica e a media de macro-F1 nos grupos confiaveis do subset de avaliacao.

Uso standalone:
  python models/v6/autoresearch/experiment.py --config '{"lr": 3e-5}' --steps 100
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_scheduler)

ROOT = Path(__file__).resolve().parents[3]
ADIR = Path(__file__).resolve().parent
DATA = ROOT / "models/v6/processed/v6_pool.parquet"
SPLITS = ROOT / "models/v6/processed/v6_splits.parquet"
BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
SPLIT_COL = "split_full_iid"
CACHE = ADIR / "data"

# Carrega o trainer v6 como modulo (fonte unica das funcoes comprovadas).
_spec = importlib.util.spec_from_file_location(
    "train_v6", ROOT / "models/v6/train_bertimbau_v6.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

DEFAULTS: dict = {
    "max_length": 192,
    "batch_size": 32,
    "lr": 2e-5,
    "warmup_frac": 0.10,
    "weight_decay": 0.01,
    "clip": 1.0,
    "freeze_layers": 6,
    "dfr_weights": "cell",
    "weight_clip": 25.0,
    "class_weights": False,
    "mask_entities": False,
    "drop_tiers": "",
    "data_filter": "full",
    "steps": 100,
    "seed": 42,
    "train_n": 5000,
    "eval_n": 3000,
    "eval_batch_size": 64,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_base() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    sp = pd.read_parquet(SPLITS)
    df = df.merge(sp[["rid", SPLIT_COL]], on="rid", how="left")
    df = df[df[SPLIT_COL].isin(["train", "val_sel"])]
    df = df[~df["has_ufffd"].astype(bool)].reset_index(drop=True)
    df["target"] = (df["label"] == "fake").astype(int)
    return df


def strat_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    key = df["group"].astype(str) + "|" + df["label"].astype(str)
    vc = key.value_counts()
    rare = key.map(vc) < 2
    if rare.any():
        key = key.where(~rare, df["label"].astype(str))
        vc = key.value_counts()
        rare = key.map(vc) < 2
        if rare.any():
            key = key.where(~rare, "all")
    try:
        sub, _ = train_test_split(df, train_size=n, random_state=seed,
                                  stratify=key)
    except ValueError:
        sub = df.sample(n=n, random_state=seed)
    return sub.reset_index(drop=True)


def frozen_subsets(n_train: int, n_eval: int, seed: int) -> tuple:
    CACHE.mkdir(parents=True, exist_ok=True)
    tr_path = CACHE / f"train_{n_train}_{seed}.parquet"
    ev_path = CACHE / f"eval_{n_eval}_{seed}.parquet"
    if tr_path.exists() and ev_path.exists():
        return pd.read_parquet(tr_path), pd.read_parquet(ev_path)
    df = load_base()
    tr = strat_sample(df[df[SPLIT_COL] == "train"], n_train, seed)
    ev = strat_sample(df[df[SPLIT_COL] == "val_sel"], n_eval, seed)
    tr.to_parquet(tr_path, index=False)
    ev.to_parquet(ev_path, index=False)
    return tr, ev


def apply_data_filter(train: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = train
    if cfg.get("data_filter") == "informative":
        out = out[out["is_balanced_group"].astype(bool)]
    tiers = [t.strip() for t in str(cfg.get("drop_tiers", "")).split(",") if t.strip()]
    if tiers:
        out = out[~out["label_tier"].isin(tiers)]
    return out.reset_index(drop=True)


def evaluate(model, ds, collate, eval_bs: int, eval_df: pd.DataFrame) -> dict:
    logits = T.infer(model, ds, collate, eval_bs, torch.device("cpu"),
                     use_amp=False, amp_dtype=None)
    p = T.sigmoid_z(logits)
    y = eval_df["target"].to_numpy().astype(int)
    glob = T.core_metrics(y, p)
    tab = T.per_group(eval_df, y, p, col="group", min_n=40)
    if tab.empty:
        return {"global": glob, "group_mean_macro_f1": float("nan"),
                "group_worst_macro_f1": float("nan"), "n_groups": 0,
                "n_reliable": 0}
    rel = tab[(tab["n"] >= 40) & (tab["minoria_n"] >= 10)
              & tab["macro_f1"].notna()]
    mean_f1 = float(rel["macro_f1"].mean()) if len(rel) else float("nan")
    worst_f1 = float(rel["macro_f1"].min()) if len(rel) else float("nan")
    return {
        "global": glob,
        "group_mean_macro_f1": mean_f1,
        "group_worst_macro_f1": worst_f1,
        "n_groups": int(len(tab)),
        "n_reliable": int(len(rel)),
        "groups": tab.to_dict("records"),
    }


def run_one(cfg_in: dict, verbose: bool = True) -> dict:
    cfg = {**DEFAULTS, **cfg_in}
    t0 = time.time()
    T.set_seed(int(cfg["seed"]))
    torch.set_num_threads(8)
    device = torch.device("cpu")
    train_all, eval_df = frozen_subsets(int(cfg["train_n"]), int(cfg["eval_n"]),
                                        int(cfg["seed"]))
    train = apply_data_filter(train_all, cfg)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tr_texts = train["text_no_url"].astype(str).tolist()
    ev_texts = eval_df["text_no_url"].astype(str).tolist()
    if cfg["mask_entities"]:
        tr_texts = [T.mask_entities_text(t) for t in tr_texts]
        ev_texts = [T.mask_entities_text(t) for t in ev_texts]

    ml = int(cfg["max_length"])
    bs = int(cfg["batch_size"])
    ds_tr = T.TextDS(tr_texts, train["target"].to_numpy(), tok, ml)
    ds_ev = T.TextDS(ev_texts, eval_df["target"].to_numpy(), tok, ml)
    collate = DataCollatorWithPadding(tok)
    tc, ec = T.TrainCollator(collate), T.EvalCollator(collate)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2)
    freeze = int(cfg["freeze_layers"])
    if freeze > 0:
        base = model.base_model
        for p in base.embeddings.parameters():
            p.requires_grad = False
        for layer in base.encoder.layer[:freeze]:
            for p in layer.parameters():
                p.requires_grad = False

    w, _ = T.build_sample_weights(train, SimpleNamespace(**cfg), device)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(k in n for k in ("bias", "LayerNorm.weight"))
         else decay).append(p)
    opt = AdamW([{"params": decay, "weight_decay": float(cfg["weight_decay"])},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=float(cfg["lr"]))
    steps = int(cfg["steps"])
    sched = get_scheduler("linear", opt,
                          num_warmup_steps=int(float(cfg["warmup_frac"]) * steps),
                          num_training_steps=steps)

    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    batch_lists = T.length_grouped_batches(ds_tr.lengths(), bs, gen)
    model.train()
    paid_tokens = 0
    loss_acc = 0.0
    step = 0
    epoch = 0
    while step < steps:
        for bidx in batch_lists:
            if step >= steps:
                break
            batch = tc([ds_tr[i] for i in bidx])
            yb = batch.pop("labels")
            maxlen = int(batch["input_ids"].shape[1])
            logits = model(**batch).logits
            if w is not None:
                wb = w[batch.pop("idx")]
                loss = (F.cross_entropy(logits, yb, reduction="none") * wb).mean()
            else:
                batch.pop("idx", None)
                loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(cfg["clip"]))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            loss_acc += float(loss.item())
            paid_tokens += int(batch["input_ids"].numel())
            step += 1
        epoch += 1
        if step < steps:
            gen = torch.Generator().manual_seed(int(cfg["seed"]) + epoch)
            batch_lists = T.length_grouped_batches(ds_tr.lengths(), bs, gen)

    train_s = time.time() - t0
    metrics = evaluate(model, ds_ev, ec, int(cfg["eval_batch_size"]), eval_df)
    total_s = time.time() - t0
    res = {
        "config": cfg,
        "steps_done": step,
        "n_train": int(len(train)),
        "n_eval": int(len(eval_df)),
        "train_loss": round(loss_acc / max(step, 1), 5),
        "paid_tokens": paid_tokens,
        "tokens_per_s": round(paid_tokens / max(train_s, 1e-9), 1),
        "train_s": round(train_s, 1),
        "total_s": round(total_s, 1),
        **metrics,
    }
    if verbose:
        g = metrics["global"]
        log(f"steps={step} train_loss={res['train_loss']:.4f} "
            f"acc={g['acc']:.4f} macroF1={g['macro_f1']:.4f} "
            f"grpMean={metrics['group_mean_macro_f1']:.4f} "
            f"grpWorst={metrics['group_worst_macro_f1']:.4f} "
            f"ECE={g['ece']:.4f} ({total_s:.0f}s, {res['tokens_per_s']:.0f} tok/s)")
    del model
    gc.collect()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="{}", help="JSON com overrides")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--train-n", type=int, default=None)
    ap.add_argument("--eval-n", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.loads(a.config)
    if a.steps is not None:
        cfg["steps"] = a.steps
    if a.train_n is not None:
        cfg["train_n"] = a.train_n
    if a.eval_n is not None:
        cfg["eval_n"] = a.eval_n
    res = run_one(cfg)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                          default=float), encoding="utf-8")
        log(f"resultado -> {a.out}")


if __name__ == "__main__":
    main()
