"""
Extracao de embeddings do BERTimbau na Vega 8 via ONNX Runtime + DirectML.

Medido nesta maquina (Ryzen 7 5825U / Radeon Vega 8):
    seq=64  batch=16 -> CPU 18,8 am/s | DirectML 51,2 am/s  (2,73x)
    seq=128 batch=16 -> CPU  8,7 am/s | DirectML 20,5 am/s  (2,35x)
    seq=128 batch=64 -> CPU  8,7 am/s | DirectML 15,4 am/s  (1,78x)

Batch grande PIORA o ganho: a iGPU divide o barramento DDR4 com a CPU e satura
banda antes de FLOPs. Por isso o default e batch 16.

Uso:
    python -m models.embed export --out models/artifacts/bertimbau.onnx
    python -m models.embed run    --onnx models/artifacts/bertimbau.onnx
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

BUCKETS = (32, 64, 96, 128, 192, 256)
BASE_MODEL = "neuralmind/bert-base-portuguese-cased"


def export(model_name_or_dir: str, out_path: str, opset: int = 17) -> str:
    """Exporta o encoder (sem cabeca de classificacao) para ONNX."""
    import torch
    from transformers import AutoModel

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    model = AutoModel.from_pretrained(model_name_or_dir).eval()

    class Encoder(torch.nn.Module):
        """Envolve o BERT para expor apenas last_hidden_state — o pooler do BERT
        e treinado em NSP e nao ajuda em sondagem linear."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state

    ids = torch.randint(0, 29000, (1, 64))
    mask = torch.ones_like(ids)
    torch.onnx.export(
        Encoder(model), (ids, mask), out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"},
                      "attention_mask": {0: "b", 1: "s"},
                      "last_hidden_state": {0: "b", 1: "s"}},
        opset_version=opset, dynamo=False,
    )
    mb = os.path.getsize(out_path) / 1e6
    print(f"ONNX exportado: {out_path} ({mb:.1f} MB)")
    return out_path


def _session(onnx_path: str, provider: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # silencia avisos de particionamento por no
    if provider == "CPUExecutionProvider":
        so.intra_op_num_threads = 8
    avail = ort.get_available_providers()
    if provider not in avail:
        raise RuntimeError(f"provider {provider} indisponivel; ha {avail}")
    sess = ort.InferenceSession(onnx_path, so, providers=[provider])
    got = sess.get_providers()[0]
    if got != provider:
        print(f"AVISO: pedimos {provider}, ORT usou {got}")
    return sess


def _bucket(n: int, max_length: int) -> int:
    for b in BUCKETS:
        if b >= n:
            return min(b, max_length)
    return max_length


def extract(texts, onnx_path: str, tokenizer_name: str = BASE_MODEL,
            provider: str = "DmlExecutionProvider", batch_size: int = 16,
            max_length: int = 256, pooling: str = "mean",
            verbose: bool = True) -> np.ndarray:
    """Retorna matriz (N, hidden) de embeddings, na ordem original de `texts`.

    Ordena por comprimento antes de lotear (menos padding e menos variacao de
    shape, que e cara no DirectML) e restaura a ordem no fim.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    texts = [str(t) for t in texts]
    enc = tok(texts, truncation=True, max_length=max_length)["input_ids"]
    lens = np.array([len(x) for x in enc])
    order = np.argsort(lens, kind="stable")

    sess = _session(onnx_path, provider)
    out = None
    t0 = time.perf_counter()
    done = 0

    for start in range(0, len(order), batch_size):
        sel = order[start:start + batch_size]
        seqs = [enc[i] for i in sel]
        L = _bucket(max(len(s) for s in seqs), max_length)
        ids = np.zeros((len(seqs), L), dtype=np.int64)
        mask = np.zeros((len(seqs), L), dtype=np.int64)
        for r, s in enumerate(seqs):
            s = s[:L]
            ids[r, :len(s)] = s
            mask[r, :len(s)] = 1

        hidden = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        if pooling == "cls":
            vec = hidden[:, 0, :]
        else:  # media mascarada — padrao para sondagem linear
            m = mask[:, :, None].astype(np.float32)
            vec = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-6, None)

        if out is None:
            out = np.zeros((len(texts), vec.shape[1]), dtype=np.float32)
        out[sel] = vec.astype(np.float32)

        done += len(sel)
        if verbose and (start // batch_size) % 50 == 0:
            el = time.perf_counter() - t0
            rate = done / max(el, 1e-6)
            eta = (len(order) - done) / max(rate, 1e-6) / 60
            print(f"  {done}/{len(order)}  {rate:.1f} am/s  ETA {eta:.1f} min",
                  flush=True)

    if verbose:
        el = time.perf_counter() - t0
        print(f"  concluido: {len(texts)} textos em {el/60:.1f} min "
              f"({len(texts)/el:.1f} am/s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export")
    e.add_argument("--model", default=BASE_MODEL)
    e.add_argument("--out", default="models/artifacts/bertimbau.onnx")

    r = sub.add_parser("run")
    r.add_argument("--onnx", default="models/artifacts/bertimbau.onnx")
    r.add_argument("--tokenizer", default=BASE_MODEL)
    r.add_argument("--csv", default=None)
    r.add_argument("--out", default="models/artifacts/embeddings.npy")
    r.add_argument("--provider", default="DmlExecutionProvider",
                   choices=["DmlExecutionProvider", "CPUExecutionProvider"])
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--max-length", type=int, default=256)
    r.add_argument("--pooling", default="mean", choices=["mean", "cls"])

    a = ap.parse_args()
    if a.cmd == "export":
        export(a.model, a.out)
        return

    from models import data as D
    df = D.load(a.csv or D.DEFAULT_CSV)
    print(f"extraindo {len(df)} embeddings via {a.provider} "
          f"(batch={a.batch_size}, max_len={a.max_length})")
    emb = extract(df[D.TEXT_COL].tolist(), a.onnx, a.tokenizer,
                  provider=a.provider, batch_size=a.batch_size,
                  max_length=a.max_length, pooling=a.pooling)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, emb)
    # `rid` ancora os embeddings as linhas; score.py valida o alinhamento.
    np.save(a.out.replace(".npy", "_rid.npy"), df["rid"].to_numpy())
    print(f"salvo: {a.out}  shape={emb.shape}")


if __name__ == "__main__":
    main()
