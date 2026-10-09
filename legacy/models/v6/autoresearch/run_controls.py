"""Controles pos-autoresearch: isolar o confundidor 'informative' x epocas.

Com steps fixos, o subset informativo (n=2.027) viu ~1,6 epocas enquanto o
full (n=5.000) viu ~0,6. Estes controles igualam ~1,0 epoca nos dois casos:
  full_1ep:        156 steps x 32 = 4.992 amostras
  informative_1ep:  63 steps x 32 = 2.016 amostras
  informative_mask_1ep: idem + mask_entities (eixo promissor na fase 1)
Resultados em controls.tsv (mesmo formato do results.tsv).
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

ADIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ar_experiment", ADIR / "experiment.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

CONTROLS = [
    ("full_1ep", {"steps": 156}),
    ("informative_1ep", {"steps": 63, "data_filter": "informative"}),
    ("informative_mask_1ep", {"steps": 63, "data_filter": "informative",
                              "mask_entities": True}),
    ("informative_ml256_1ep", {"steps": 63, "data_filter": "informative",
                               "max_length": 256}),
]

TSV = ADIR / "controls.tsv"


def main() -> None:
    rows = []
    for label, over in CONTROLS:
        cfg = {**E.DEFAULTS, **over}
        E.log(f"=== controle {label} {json.dumps(over)} ===")
        try:
            res = E.run_one(cfg)
            res.update({"run_id": label, "status": "ok"})
        except Exception as exc:  # noqa: BLE001
            E.log(f"!!! {label} falhou: {exc}")
            res = {"run_id": label, "status": "erro", "config": cfg,
                   "erro": str(exc)[:300]}
        rows.append(res)
        with TSV.open("w", encoding="utf-8") as f:
            f.write("run_id\tstatus\tgroup_mean_macro_f1\tgroup_worst_macro_f1\t"
                    "acc\tmacro_f1\tece\tn_train\tsteps\ttotal_s\tconfig\n")
            for r in rows:
                g = r.get("global") or {}
                f.write("\t".join(str(x) for x in (
                    r["run_id"], r["status"],
                    r.get("group_mean_macro_f1"), r.get("group_worst_macro_f1"),
                    g.get("acc"), g.get("macro_f1"), g.get("ece"),
                    r.get("n_train"), r.get("steps_done"), r.get("total_s"),
                    json.dumps(r.get("config", {}), sort_keys=True))) + "\n")
        time.sleep(2)


if __name__ == "__main__":
    main()
