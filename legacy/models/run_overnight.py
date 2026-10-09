"""
Pipeline completo: Trilha A (encoder congelado) -> fine-tuning -> Trilha A sobre
o modelo ajustado -> relatorio.

Roda tudo em sequencia sem supervisao. Cada etapa loga em models/artifacts/logs/
e uma falha nao derruba as etapas seguintes que ainda fizerem sentido.

    python -m models.run_overnight
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time

ART = "models/artifacts"
LOGS = f"{ART}/logs"
STATE = f"{ART}/pipeline_state.json"


def _now() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def save_state(state: dict) -> None:
    os.makedirs(ART, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def step(state: dict, name: str, args: list[str], skip_if: str | None = None) -> bool:
    """Executa uma etapa. Devolve True se terminou bem (ou foi pulada)."""
    if skip_if and os.path.exists(skip_if):
        print(f"[{_now()}] PULANDO {name} (ja existe {skip_if})", flush=True)
        state[name] = {"status": "skipped", "artifact": skip_if}
        save_state(state)
        return True

    os.makedirs(LOGS, exist_ok=True)
    log_path = f"{LOGS}/{name}.log"
    print(f"[{_now()}] INICIANDO {name}", flush=True)
    print(f"           {' '.join(args)}", flush=True)
    print(f"           log -> {log_path}", flush=True)

    t0 = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, *args], stdout=log,
                              stderr=subprocess.STDOUT, env=env)
    mins = (time.perf_counter() - t0) / 60

    ok = proc.returncode == 0
    state[name] = {"status": "ok" if ok else "falhou", "returncode": proc.returncode,
                   "minutos": round(mins, 1), "log": log_path}
    save_state(state)
    print(f"[{_now()}] {'CONCLUIDO' if ok else 'FALHOU'} {name} "
          f"({mins:.1f} min, rc={proc.returncode})", flush=True)
    if not ok:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = f.read()[-1500:]
        print(f"--- fim do log de {name} ---\n{tail}\n---", flush=True)
    return ok


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    state: dict = {"iniciado_em": dt.datetime.now().isoformat(timespec="seconds")}
    save_state(state)
    print(f"=== pipeline iniciado {_now()} ===", flush=True)

    # ---- Trilha A: encoder congelado (BERTimbau pre-treinado) -----------
    step(state, "1_embed_export_base",
         ["-m", "models.embed", "export", "--out", f"{ART}/bertimbau.onnx"],
         skip_if=f"{ART}/bertimbau.onnx")

    step(state, "2_embed_run_base",
         ["-m", "models.embed", "run", "--onnx", f"{ART}/bertimbau.onnx",
          "--out", f"{ART}/embeddings.npy", "--provider", "DmlExecutionProvider",
          "--batch-size", "16", "--max-length", "256"],
         skip_if=f"{ART}/embeddings.npy")

    step(state, "3_score_frozen",
         ["-m", "models.score", "--embeddings", f"{ART}/embeddings.npy",
          "--ood", "whatsapp", "portal",
          "--out", f"{ART}/score_frozen.json"])

    # Ablação barata: quanto as features estilísticas contribuem de fato?
    step(state, "4_score_frozen_nostyle",
         ["-m", "models.score", "--embeddings", f"{ART}/embeddings.npy",
          "--no-style", "--ood", "whatsapp",
          "--out", f"{ART}/score_frozen_nostyle.json"])

    # ---- Trilha B: fine-tuning ------------------------------------------
    ft_dir = f"{ART}/bertimbau_finetuned"
    ft_ok = step(state, "5_finetune",
                 ["-m", "models.encoder", "--max-length", "192",
                  "--freeze-layers", "6", "--epochs", "2", "--batch-size", "16",
                  "--lr", "2e-5", "--out", ft_dir],
                 skip_if=f"{ft_dir}/config.json")

    # ---- Trilha A de novo, agora sobre o encoder ajustado ---------------
    if ft_ok:
        step(state, "6_embed_export_ft",
             ["-m", "models.embed", "export", "--model", ft_dir,
              "--out", f"{ART}/bertimbau_ft.onnx"],
             skip_if=f"{ART}/bertimbau_ft.onnx")

        step(state, "7_embed_run_ft",
             ["-m", "models.embed", "run", "--onnx", f"{ART}/bertimbau_ft.onnx",
              "--tokenizer", ft_dir, "--out", f"{ART}/embeddings_ft.npy",
              "--provider", "DmlExecutionProvider", "--batch-size", "16",
              "--max-length", "192"],
             skip_if=f"{ART}/embeddings_ft.npy")

        step(state, "8_score_finetuned",
             ["-m", "models.score", "--embeddings", f"{ART}/embeddings_ft.npy",
              "--ood", "whatsapp", "portal",
              "--out", f"{ART}/score_finetuned.json"])
    else:
        print(f"[{_now()}] fine-tuning falhou — pulando etapas 6-8", flush=True)

    step(state, "9_relatorio", ["-m", "models.make_report"])

    state["terminado_em"] = dt.datetime.now().isoformat(timespec="seconds")
    save_state(state)
    print(f"\n=== pipeline terminado {_now()} ===", flush=True)
    for k, v in state.items():
        if isinstance(v, dict):
            print(f"  {k:28s} {v['status']:8s} {v.get('minutos', '')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
