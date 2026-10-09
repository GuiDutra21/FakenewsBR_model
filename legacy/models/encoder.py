"""
Fine-tuning do BERTimbau (substitui bert_finetune.py).

Correcoes em relacao ao script original, todas medidas nesta maquina:

  1. padding dinamico por batch + agrupamento por comprimento.
     O original fazia `tokenizer(textos, padding=True, max_length=256)` na lista
     INTEIRA, paddando tudo para 256. A mediana real e 33 tokens => 3,20x de
     compute desperdicado. Era a causa de 1 epoca levar 545 min.
  2. `text_no_url` no lugar de `text_clean`. O original alimentava um modelo
     *cased* com texto minusculizado e sem acento.
  3. removido `torch.device("vulkan")`: o backend Vulkan do PyTorch e
     inference-only e nunca teve kernels de backward. O ramo nunca executava.
  4. split estratificado por (grupo x rotulo), warmup de 10%, lr 2e-5,
     weight decay sem bias/LayerNorm, clip de gradiente, melhor checkpoint por
     macro-F1 de validacao.
  5. SEM pesos de classe por padrao: o objetivo do projeto e um score
     CALIBRADO, e ponderar a loss distorce justamente a probabilidade que
     queremos calibrar. Desbalanceamento e tratado no limiar e nas metricas.

Custo estimado (27,6k amostras de treino, batch 16):
    cap 192 + padding dinamico ................ ~148 min/epoca
    cap 192 + dinamico + 6 camadas congeladas .. ~103 min/epoca
    (original: 545 min/epoca)
"""
from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_scheduler)

from models import data as D
from models import evaluate as E

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

_MASKS = [
    (r"\b(lula|bolsonaro|dilma|temer|doria|ciro|haddad|moraes)\b", "[POLITICO]"),
    (r"\b(cloroquina|ivermectina|vacina|coronavac|pfizer|astrazeneca)\b", "[SAUDE]"),
    (r"\b(stf|tse|minist[eé]rio p[uú]blico)\b", "[INSTITUICAO]"),
]


def mask_entities(text: str) -> str:
    """Mascara entidades polarizadas. Fica atras de flag de proposito: a pergunta
    'mascarar ou nao' ficou em aberto no plano e merece ser MEDIDA, nao assumida."""
    if not isinstance(text, str):
        return ""
    for pat, tag in _MASKS:
        text = re.sub(pat, tag, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


class TextDS(Dataset):
    """Guarda tokens SEM padding — o collator padda por batch."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item

    def lengths(self):
        return [len(x) for x in self.enc["input_ids"]]


def length_grouped_batches(lengths, batch_size, generator, mega=50):
    """Ordena por comprimento dentro de megabatches para minimizar padding,
    mantendo aleatoriedade entre epocas."""
    idx = torch.randperm(len(lengths), generator=generator).tolist()
    span = batch_size * mega
    out = []
    for i in range(0, len(idx), span):
        chunk = sorted(idx[i:i + span], key=lambda j: lengths[j])
        out += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    perm = torch.randperm(len(out), generator=generator).tolist()
    return [out[i] for i in perm]


@torch.no_grad()
def infer(model, ds, collate, batch_size, device):
    model.eval()
    logits = []
    order = np.argsort(ds.lengths(), kind="stable")
    for s in range(0, len(order), batch_size):
        sel = order[s:s + batch_size]
        batch = collate([ds[i] for i in sel])
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits.append(model(**batch).logits.cpu().numpy())
    out = np.zeros((len(ds), 2), dtype=np.float32)
    out[order] = np.concatenate(logits)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model", default=MODEL_NAME,
                    help="encoder base. Para atacar o vies dialetal PT-PT, "
                         "'xlm-roberta-base' era a recomendacao do transformer_comparison.md")
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--freeze-layers", type=int, default=6,
                    help="congela embeddings + N camadas inferiores (0 = fine-tune completo)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask-entities", action="store_true")
    ap.add_argument("--class-weights", action="store_true",
                    help="ativa loss ponderada (piora a calibracao — so para comparar)")
    ap.add_argument("--balanced-groups-only", action="store_true",
                    help="treina so nos grupos sem confundimento origem->rotulo")
    ap.add_argument("--out", default="models/artifacts/bertimbau_finetuned")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cpu")  # nao ha caminho de GPU para backward nesta maquina
    print(f"device={device} threads={torch.get_num_threads()} torch={torch.__version__}")

    df = D.load(a.csv or D.DEFAULT_CSV)
    if a.balanced_groups_only:
        df = df[df["is_balanced_group"]].reset_index(drop=True)
        print(f"restrito a grupos balanceados: n={len(df)}")
    split = D.iid_split(df, seed=a.seed)
    print(split.describe())

    col = D.TEXT_COL
    get = (lambda d: d[col].map(mask_entities).tolist()) if a.mask_entities \
        else (lambda d: d[col].astype(str).tolist())
    if a.mask_entities:
        print("mascaramento de entidades ATIVO")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
    print(f"encoder: {a.model}")

    if a.freeze_layers:
        # `base_model` em vez de `.bert` para funcionar tambem com XLM-R/DeBERTa.
        base = model.base_model
        for p in base.embeddings.parameters():
            p.requires_grad = False
        for layer in base.encoder.layer[:a.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        live = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"congeladas embeddings + {a.freeze_layers} camadas ({live/1e6:.1f}M treinaveis)")
    model.to(device)

    ds_tr = TextDS(get(split.train), split.train["target"], tok, a.max_length)
    ds_va = TextDS(get(split.val), split.val["target"], tok, a.max_length)
    ds_te = TextDS(get(split.test), split.test["target"], tok, a.max_length)
    collate = DataCollatorWithPadding(tok, return_tensors="pt")

    tr_lens = ds_tr.lengths()
    print(f"tokens no treino: media={np.mean(tr_lens):.0f} "
          f"p50={np.percentile(tr_lens,50):.0f} p95={np.percentile(tr_lens,95):.0f} "
          f"(padding fixo em {a.max_length} desperdicaria "
          f"{a.max_length/np.mean(tr_lens):.2f}x)")

    if a.class_weights:
        counts = np.bincount(split.train["target"].to_numpy(), minlength=2)
        w = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float, device=device)
        print(f"AVISO: loss ponderada ativa (true={w[0]:.3f} fake={w[1]:.3f}) — "
              f"isso degrada a calibracao do score")
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(k in n for k in ("bias", "LayerNorm.weight")) else decay).append(p)
    opt = AdamW([{"params": decay, "weight_decay": 0.01},
                 {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)

    gen = torch.Generator().manual_seed(a.seed)
    steps = ((len(ds_tr) + a.batch_size - 1) // a.batch_size) * a.epochs
    sched = get_scheduler("linear", opt, num_warmup_steps=int(0.1 * steps),
                          num_training_steps=steps)

    best_f1, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        t0 = time.perf_counter()
        run = 0.0
        batches = length_grouped_batches(tr_lens, a.batch_size, gen)
        loop = tqdm(batches, desc=f"epoca {ep+1}/{a.epochs}")
        for step, bidx in enumerate(loop, 1):
            batch = collate([ds_tr[i] for i in bidx])
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            loop.set_postfix(loss=run / step)

        yva = split.val["target"].to_numpy()
        lva = infer(model, ds_va, collate, a.batch_size * 2, device)
        pva = E.apply_platt(lva, 1.0, 0.0)  # sem calibrar: so para ranquear epocas
        m = E.core_metrics(yva, pva)
        wg = E.worst_group_f1(split.val, yva, pva)
        print(f"epoca {ep+1} ({(time.perf_counter()-t0)/60:.1f} min): "
              f"val macro-F1={m['macro_f1']:.4f} pior-grupo={wg:.4f} ECE={m['ece']:.4f}")
        score = wg if not np.isnan(wg) else m["macro_f1"]
        if score > best_f1:
            best_f1 = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  -> melhor checkpoint (pior-grupo={score:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibra na validacao, avalia no teste. Nunca o contrario.
    yva = split.val["target"].to_numpy()
    lva = infer(model, ds_va, collate, a.batch_size * 2, device)
    pa, pb = E.fit_platt(lva, yva)
    yte = split.test["target"].to_numpy()
    lte = infer(model, ds_te, collate, a.batch_size * 2, device)
    pte = E.apply_platt(lte, pa, pb)
    print(f"\nPlatt calibrado: a={pa:.4f} b={pb:+.4f}  "
          f"ECE val {E.expected_calibration_error(E.apply_platt(lva,1.0,0.0), yva):.4f}"
          f" -> {E.expected_calibration_error(E.apply_platt(lva,pa,pb), yva):.4f}")
    E.report(split.test, yte, pte, "TESTE | BERTimbau fine-tuned (calibrado)")

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "calibration.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({"platt_a": float(pa), "platt_b": float(pb),
                   "max_length": a.max_length, "model": a.model,
                   "mask_entities": a.mask_entities}, f, indent=2)
    print(f"\nmodelo salvo em {a.out}")
    print("proximo passo: python -m models.embed export --model", a.out,
          "--out models/artifacts/bertimbau_ft.onnx")


if __name__ == "__main__":
    main()
