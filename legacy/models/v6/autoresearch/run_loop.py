"""autoresearch/run_loop.py — loop autonomo de experimentos.

Personalidade portada de karpathy/autoresearch: budget fixo por experimento,
UMA metrica de manchete, mantem/descarta, log de tudo, roda a noite inteira.
A 'modificacao de codigo' vira um espaco de configs de treino (JSON).

Fases:
  0  baseline (referencia) + repeats de seed (ruido)
  1  variacoes de UM eixo por vez (sweep)
  2  descida coordenada: combina os eixos vencedores sobre o melhor config
  3  verificacao: melhor config com outra seed e, se sobrar budget, 2x steps

Uso:
  python models/v6/autoresearch/run_loop.py --max-minutes 250
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ADIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ar_experiment", ADIR / "experiment.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

BASELINE = dict(E.DEFAULTS)
# baseline canonico do projeto (R0): 100 steps de budget fixo
BASELINE["steps"] = 100

PHASE1 = [
    ("baseline", {}),
    ("dfr_off", {"dfr_weights": "off"}),
    ("ml128", {"max_length": 128}),
    ("ml256", {"max_length": 256}),
    ("freeze0", {"freeze_layers": 0}),
    ("freeze4", {"freeze_layers": 4}),
    ("freeze8", {"freeze_layers": 8}),
    ("mask", {"mask_entities": True}),
    ("lr1e-5", {"lr": 1e-5}),
    ("lr3e-5", {"lr": 3e-5}),
    ("clip05", {"clip": 0.5}),
    ("warmup0", {"warmup_frac": 0.0}),
    ("wd0", {"weight_decay": 0.0}),
    ("droptiers", {"drop_tiers": "llm_local,corroborated"}),
    ("informative", {"data_filter": "informative"}),
    ("classweights", {"class_weights": True, "dfr_weights": "off"}),
    ("batch16", {"batch_size": 16}),
    ("seed43", {"seed": 43}),
]

# eixos da fase 2, na ordem em que serao combinados (magnitude decidida pelos dados)
PHASE2_AXES = [
    ("max_length", [128, 192, 256]),
    ("freeze_layers", [0, 4, 6, 8]),
    ("lr", [1e-5, 2e-5, 3e-5]),
    ("dfr_weights", ["cell", "off"]),
    ("mask_entities", [False, True]),
    ("clip", [0.5, 1.0]),
    ("warmup_frac", [0.0, 0.10]),
    ("weight_decay", [0.0, 0.01]),
    ("data_filter", ["full", "informative"]),
]

PRIMARY = "group_mean_macro_f1"
TSV = ADIR / "results.tsv"
PROGRESS = ADIR / "progress.md"
VERDICT = ADIR / "VERDICT.md"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id(n: int, label: str) -> str:
    return f"{n:03d}_{label}"


def safe(v, fmt="{:.4f}"):
    if v is None:
        return "-"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(fv):
        return "-"
    return fmt.format(fv)


def write_tsv(rows: list[dict]) -> None:
    cols = ["run_id", "phase", "label", "status", PRIMARY, "group_worst_macro_f1",
            "acc", "macro_f1", "f1_fake", "ece", "brier", "train_loss",
            "steps_done", "n_train", "n_eval", "total_s", "tokens_per_s",
            "seed", "config_json"]
    lines = ["\t".join(cols)]
    for r in rows:
        g = r.get("global") or {}
        cfg = r.get("config") or {}
        vals = [
            r["run_id"], r["phase"], r["label"], r.get("status", "ok"),
            safe(r.get(PRIMARY)), safe(r.get("group_worst_macro_f1")),
            safe(g.get("acc")), safe(g.get("macro_f1")), safe(g.get("f1_fake")),
            safe(g.get("ece")), safe(g.get("brier")), safe(r.get("train_loss")),
            str(r.get("steps_done", "")), str(r.get("n_train", "")),
            str(r.get("n_eval", "")), str(r.get("total_s", "")),
            str(r.get("tokens_per_s", "")), str(cfg.get("seed", "")),
            json.dumps(cfg, sort_keys=True),
        ]
        lines.append("\t".join(vals))
    TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_progress(rows: list[dict], t0: float, max_minutes: float,
                   best: dict | None, status: str) -> None:
    elapsed = (time.time() - t0) / 60
    lines = [
        "# autoresearch — progresso ao vivo",
        "",
        f"- atualizado: {now()}",
        f"- budget: {max_minutes:.0f} min | decorrido: {elapsed:.1f} min",
        f"- experimentos: {len(rows)} | status: {status}",
        "",
    ]
    if best:
        cfg = best.get("config", {})
        lines += [
            "## melhor ate agora",
            "",
            f"- run: `{best['run_id']}` ({best['label']})",
            f"- {PRIMARY}: **{safe(best.get(PRIMARY))}** | worst: "
            f"{safe(best.get('group_worst_macro_f1'))} | acc: "
            f"{safe((best.get('global') or {}).get('acc'))} | "
            f"macroF1: {safe((best.get('global') or {}).get('macro_f1'))} | "
            f"ECE: {safe((best.get('global') or {}).get('ece'))}",
            f"- eixos: ml={cfg.get('max_length')} freeze={cfg.get('freeze_layers')} "
            f"lr={cfg.get('lr')} dfr={cfg.get('dfr_weights')} "
            f"mask={cfg.get('mask_entities')} filtro={cfg.get('data_filter')} "
            f"clip={cfg.get('clip')} wd={cfg.get('weight_decay')} "
            f"warmup={cfg.get('warmup_frac')}",
            "",
        ]
    lines += [
        "## runs",
        "",
        f"| run | fase | label | {PRIMARY} | worst | acc | macroF1 | ECE | min | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        g = r.get("global") or {}
        lines.append(
            f"| {r['run_id']} | {r['phase']} | {r['label']} | "
            f"{safe(r.get(PRIMARY))} | {safe(r.get('group_worst_macro_f1'))} | "
            f"{safe(g.get('acc'))} | {safe(g.get('macro_f1'))} | "
            f"{safe(g.get('ece'))} | "
            f"{safe((r.get('total_s') or 0) / 60, '{:.1f}')} | "
            f"{r.get('status', 'ok')} |")
    PROGRESS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verdict(rows: list[dict], t0: float, status: str) -> None:
    ok = [r for r in rows if r.get("status") == "ok" and not np.isnan(
        float(r.get(PRIMARY) or float("nan")))]
    base = next((r for r in ok if r["label"] == "baseline"), None)
    ranked = sorted(ok, key=lambda r: -float(r[PRIMARY]))
    lines = [
        "# VERDICT — melhor estrategia (autoresearch v6)",
        "",
        f"- gerado: {now()} | status: {status}",
        f"- budget usado: {(time.time()-t0)/60:.1f} min | runs ok: {len(ok)}",
        f"- metrica de manchete: `{PRIMARY}` (media de macro-F1 nos grupos "
        "confiaveis do subset de avaliacao congelado, 3.000 linhas)",
        "",
        "## ranking",
        "",
        f"| # | run | {PRIMARY} | worst | acc | macroF1 | ECE | eixos |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, 1):
        g = r.get("global") or {}
        cfg = r.get("config", {})
        axes = (f"ml={cfg.get('max_length')} fz={cfg.get('freeze_layers')} "
                f"lr={cfg.get('lr')} dfr={cfg.get('dfr_weights')} "
                f"mask={int(bool(cfg.get('mask_entities')))} "
                f"filt={cfg.get('data_filter')} clip={cfg.get('clip')} "
                f"wd={cfg.get('weight_decay')} wu={cfg.get('warmup_frac')} "
                f"bt={cfg.get('batch_size')} seed={cfg.get('seed')}")
        lines.append(
            f"| {i} | {r['run_id']} {r['label']} | {safe(r.get(PRIMARY))} | "
            f"{safe(r.get('group_worst_macro_f1'))} | {safe(g.get('acc'))} | "
            f"{safe(g.get('macro_f1'))} | {safe(g.get('ece'))} | {axes} |")
    lines += ["", "## efeito de cada eixo (vs baseline)", ""]
    if base:
        b = float(base[PRIMARY])
        for r in ok:
            if r["label"] == "baseline":
                continue
            d = float(r[PRIMARY]) - b
            lines.append(f"- {r['label']}: {d:+.4f} ({safe(r[PRIMARY])})")
        lines += ["", f"baseline = {safe(b)} em {PRIMARY}", ""]
    if ranked:
        best = ranked[0]
        lines += [
            "## config vencedor (indicio, nao modelo final)",
            "",
            "```json",
            json.dumps(best["config"], indent=2, ensure_ascii=False, default=str),
            "```",
            "",
            "## proximo passo recomendado",
            "",
            "Rodar o trainer completo (`models/v6/train_bertimbau_v6.py`) com o "
            "config vencedor nos 63.753 exemplos de treino e validar no teste "
            "`full_iid` (o ranking aqui e um indicio de 100 steps / 5k amostras, "
            "sujeito a ruido; ver ressalvas no relatorio da manha).",
        ]
    VERDICT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=250.0)
    ap.add_argument("--max-runs", type=int, default=40)
    ap.add_argument("--est-min-per-run", type=float, default=12.0)
    a = ap.parse_args()

    t0 = time.time()
    rows: list[dict] = []
    best: dict | None = None
    n = 0
    status = "rodando"

    def remaining() -> float:
        return a.max_minutes - (time.time() - t0) / 60

    def maybe_run(phase: str, label: str, overrides: dict) -> dict:
        nonlocal n, best
        n += 1
        rid = run_id(n, label)
        cfg = {**BASELINE, **overrides}
        E.log(f"=== {rid} (fase {phase}) {json.dumps(overrides)} ===")
        try:
            res = E.run_one(cfg)
            res.update({"run_id": rid, "phase": phase, "label": label,
                        "status": "ok", "finished_utc": now()})
            if best is None or float(res[PRIMARY]) > float(best[PRIMARY]):
                best = res
                E.log(f"*** novo melhor: {rid} {PRIMARY}="
                      f"{safe(res[PRIMARY])} ***")
        except Exception as exc:  # noqa: BLE001
            E.log(f"!!! {rid} falhou: {exc}")
            res = {"run_id": rid, "phase": phase, "label": label,
                   "status": "erro", "config": cfg, "erro": str(exc)[:300],
                   "finished_utc": now()}
        rows.append(res)
        write_tsv(rows)
        write_progress(rows, t0, a.max_minutes, best, status)
        write_verdict(rows, t0, status)
        return res

    try:
        E.log(f"autoresearch: budget {a.max_minutes:.0f} min | {PHASE1} ---")
        for label, over in PHASE1:
            if n >= a.max_runs or remaining() < a.est_min_per_run:
                break
            maybe_run("1", label, over)

        # fase 2: descida coordenada a partir do melhor
        if best is not None and remaining() >= a.est_min_per_run:
            current = {k: v for k, v in best["config"].items()
                       if k in E.DEFAULTS}
            for axis, values in PHASE2_AXES:
                if n >= a.max_runs or remaining() < a.est_min_per_run:
                    break
                cur_val = current.get(axis)
                for v in values:
                    if v == cur_val:
                        continue
                    cand = {k: current[k] for k in E.DEFAULTS if k in current}
                    cand[axis] = v
                    res = maybe_run("2", f"{axis}={v}", cand)
                    if (res.get("status") == "ok"
                            and best is not None
                            and res["run_id"] == best["run_id"]):
                        pass
                # adota o melhor valor observado do eixo no config corrente
                axis_runs = [r for r in rows if r["phase"] == "2"
                             and r["label"].startswith(f"{axis}=")
                             and r["status"] == "ok"]
                if axis_runs and best is not None:
                    b = max(axis_runs, key=lambda r: float(r[PRIMARY]))
                    current = {k: v for k, v in b["config"].items()
                               if k in E.DEFAULTS}
                    E.log(f"[fase2] eixo {axis}: adotado {b['label']}")

        # fase 3: repeticao do melhor com outra seed (ruido) e budget 2x
        if best is not None and remaining() >= 2 * a.est_min_per_run:
            best_cfg = {k: v for k, v in best["config"].items()
                        if k in E.DEFAULTS}
            maybe_run("3", "best_seed43", {**best_cfg, "seed": 43})
        if best is not None and remaining() >= 2.5 * a.est_min_per_run:
            best_cfg = {k: v for k, v in best["config"].items()
                        if k in E.DEFAULTS}
            maybe_run("3", "best_steps200", {**best_cfg, "steps": 200})
    finally:
        status = "concluido" if remaining() > 0 else "budget esgotado"
        write_tsv(rows)
        write_progress(rows, t0, a.max_minutes, best, status)
        write_verdict(rows, t0, status)
        E.log(f"[fim] {len(rows)} runs | status={status} | VERDICT em {VERDICT}")


if __name__ == "__main__":
    main()
