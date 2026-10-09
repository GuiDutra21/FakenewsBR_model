"""Reexecuta a cabeca de classificacao do BERTimbau fine-tuned sobre val + teste.

Esse e o modelo que produziu "acc 0,8655 / F1(fake) 0,9102 / PT-PT 0,6248" no
`5_finetune.log`. Ele NAO e a cabeca ERM nem a DFR de `score.py` (essas rodam
sobre os embeddings mean-pooled de `embeddings_ft.npy`): e a camada
`BertForSequenceClassification.classifier` treinada junto com o encoder, com
Platt ajustado na validacao inteira.

Salva as probabilidades por `rid` para a analise por fatia em `analyze.py`.
A primeira coisa que o script faz depois de inferir e conferir que reproduz a
acuracia do log — se nao reproduzir, o split mudou e nada a jusante vale.

Uso:
    PYTHONPATH=legacy python -m investigation.reverse_analysis.predict_ft
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

from models import data as D
from models import evaluate as E
from models.encoder import TextDS, infer

ART = Path("legacy/models/artifacts/bertimbau_finetuned")
OUT = Path("legacy/models/artifacts/preds_ft_classifier.csv")
LOGGED_TEST_ACC = 0.8655


def main():
    torch.set_num_threads(8)
    cal = json.loads((ART / "calibration.json").read_text(encoding="utf-8"))
    if cal.get("mask_entities"):
        raise SystemExit("modelo treinado com mascaramento; este script nao o aplica")

    df = D.load(D.DEFAULT_CSV)
    split = D.iid_split(df, seed=42)
    print(split.describe(), flush=True)

    tok = AutoTokenizer.from_pretrained(ART)
    model = AutoModelForSequenceClassification.from_pretrained(ART).eval()
    collate = DataCollatorWithPadding(tok, return_tensors="pt")

    frames = []
    for name, d in (("test", split.test), ("val", split.val)):
        ds = TextDS(d[D.TEXT_COL].astype(str).tolist(), d["target"], tok,
                    cal["max_length"])
        t0 = time.perf_counter()
        logits = infer(model, ds, collate, 32, torch.device("cpu"))
        p = E.apply_platt(logits, cal["platt_a"], cal["platt_b"])
        acc = float(((p >= 0.5).astype(int) == d["target"].to_numpy()).mean())
        print(f"{name}: n={len(d)} acc={acc:.4f} "
              f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
        if name == "test" and abs(acc - LOGGED_TEST_ACC) > 0.002:
            raise SystemExit(f"acc de teste {acc:.4f} nao reproduz o log "
                             f"({LOGGED_TEST_ACC}); o split divergiu")
        frames.append(pd.DataFrame({
            "rid": d["rid"].to_numpy(), "split": name,
            "logit_true": logits[:, 0], "logit_fake": logits[:, 1], "p_fake": p,
        }))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
    print(f"salvo em {OUT}")


if __name__ == "__main__":
    main()
