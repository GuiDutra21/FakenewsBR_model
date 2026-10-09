#!/usr/bin/env python
"""Fila sequencial dos testes de hiperparametros do train_bertimbau_v6.py.

Uso (a partir da raiz do repositorio, d:/residencia):
    python FakenewsBR_model/organizado/train/experimentos/run_experiments.py --fase 1
    python FakenewsBR_model/organizado/train/experimentos/run_experiments.py --so T1_lr3e-5_s43 T1_lr3e-5_s44

- Roda um teste por vez (uma GPU so; em paralelo faltaria memoria).
- Pula testes que ja tem metrics.json (da para interromper e retomar a fila).
- Se um teste falhar por falta de memoria, repete com --batch-size 16
  --grad-accum 2 (mesmo batch efetivo de 32).
- Um teste com erro nao para a fila; o status vai para status.json.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # d:/residencia
EXP_DIR = Path(__file__).resolve().parent           # .../train/experimentos
TRAIN = EXP_DIR.parent / "train_bertimbau_v6.py"
DATA_DIR = EXP_DIR.parents[1] / "prepare_data" / "processed"

BASE_ARGS = [
    "--data", str(DATA_DIR / "v6_pool.parquet"),
    "--splits", str(DATA_DIR / "v6_splits.parquet"),
    "--best-metric", "macro_f1",
    "--no-predict-all",
    # teto de 15 epocas (default do script) -> early stopping decide. Warmup fixo
    # em ~400 passos = o mesmo das 2 epocas antigas (10% de 3.986); com
    # --warmup-frac 0.10 sobre 15 epocas viraria 1,5 epoca de warmup.
    "--warmup-steps", "400",
]

# nome -> flags extras (a seed e acrescentada por seed)
TESTES = {
    "BASE": [],
    "T1_lr3e-5": ["--lr", "3e-5"],
    "T2_lr5e-5": ["--lr", "5e-5"],
    "T3_patience2": ["--patience", "2"],
    "T4_freeze3": ["--freeze-layers", "3"],
    "T5_freeze0": ["--freeze-layers", "0"],
    "T6_maxlen256": ["--max-length", "256"],
    "T7_clip10": ["--weight-clip", "10"],
    "T8_droptiers": ["--drop-tiers", "llm_local,corroborated"],
    # T5 parou na epoca 1 com lr 2e-5; encoder inteiro costuma pedir lr menor
    "T9_freeze0_lr1e-5": ["--freeze-layers", "0", "--lr", "1e-5"],
}

FASES = {
    "1": [("BASE", s) for s in (42, 43, 44)],
    "2": [(t, 42) for t in TESTES if t != "BASE"],
}

OOM_FALLBACK = ["--batch-size", "16", "--grad-accum", "2"]


def nome_run(teste: str, seed: int) -> str:
    return f"{teste}_s{seed}"


def parse_nome(nome: str) -> tuple[str, int]:
    teste, s = nome.rsplit("_s", 1)
    return teste, int(s)


def manter_acordado() -> None:
    """Impede a suspensao do Windows enquanto a fila roda (nao cobre fechar a tampa)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def rodar(teste: str, seed: int, status: dict) -> None:
    nome = nome_run(teste, seed)
    out = EXP_DIR / "runs" / nome
    if (out / "metrics.json").exists():
        print(f"[fila] {nome}: ja concluido, pulando", flush=True)
        status[nome] = status.get(nome) or {"estado": "ok"}
        return
    log_dir = EXP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    extra = TESTES[teste] + ["--seed", str(seed), "--run-id", nome, "--out", str(out)]
    tentativas = [extra]
    if "--batch-size" not in extra:
        tentativas.append(extra + OOM_FALLBACK)
    for i, flags in enumerate(tentativas):
        log = log_dir / (f"{nome}.log" if i == 0 else f"{nome}_bs16.log")
        cmd = [sys.executable, "-u", str(TRAIN)] + BASE_ARGS + flags
        print(f"[fila] {datetime.now():%H:%M:%S} iniciando {nome}"
              f"{' (retry bs16 x accum2)' if i else ''} -> {log.name}", flush=True)
        t0 = time.perf_counter()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        mins = (time.perf_counter() - t0) / 60
        texto = log.read_text(encoding="utf-8", errors="replace").lower()
        oom = "out of memory" in texto or "cuda oom" in texto
        status[nome] = {"estado": "ok" if rc == 0 else ("oom" if oom else "erro"),
                        "rc": rc, "minutos": round(mins, 1), "flags": flags,
                        "log": log.name}
        print(f"[fila] {nome}: rc={rc} {status[nome]['estado']} em {mins:.1f} min", flush=True)
        save_status(status)
        if rc == 0 or not oom:
            return


def save_status(status: dict) -> None:
    p = EXP_DIR / "status.json"
    p.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fase", choices=sorted(FASES))
    g.add_argument("--so", nargs="+", help="nomes TESTE_sSEED, ex.: T1_lr3e-5_s43")
    args = ap.parse_args()
    fila = FASES[args.fase] if args.fase else [parse_nome(n) for n in args.so]
    for teste, _ in fila:
        if teste not in TESTES:
            print(f"[fila] teste desconhecido: {teste}")
            return 2
    sp = EXP_DIR / "status.json"
    status = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    manter_acordado()
    print(f"[fila] {len(fila)} treino(s): {[nome_run(t, s) for t, s in fila]}", flush=True)
    for teste, seed in fila:
        rodar(teste, seed, status)
    save_status(status)
    print("[fila] fim", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
