#!/usr/bin/env python
"""Remove as copias extras de duplicatas exatas do pool v6.

Uso: python pipeline/prepare/dedup_exatas.py

Le processed/v6_pool.parquet + v6_splits.parquet e grava em processed_dedup/
os mesmos arquivos sem as copias extras. Duplicata exata = mesmo texto apos
`prepare.normalize_text` (a normalizacao que mediu as 473 copias extras).

- Fica UMA linha por texto: a de menor rid. Se as copias tem rotulos
  conflitantes, fica a de menor rid entre as do rotulo majoritario (empate:
  menor rid de todas).
- Quase-duplicatas (Jaccard >= 0,8) nao sao tocadas.
- O lado de cada linha restante no split e o mesmo do v6 (o prepare ja poe
  todas as copias de um texto no mesmo lado), entao treino/val/teste so perdem
  as copias.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import TEXT_COL, normalize_text, sha256_file  # noqa: E402

SRC = HERE / "processed"
DST = HERE / "processed_dedup"


def main() -> int:
    pool = pd.read_parquet(SRC / "v6_pool.parquet")
    splits = pd.read_parquet(SRC / "v6_splits.parquet")
    key = pool[TEXT_COL].map(normalize_text)

    d = pd.DataFrame({"rid": pool["rid"], "key": key, "label": pool["label"]})
    d["n_label"] = d.groupby(["key", "label"])["rid"].transform("size")
    d["max_label"] = d.groupby("key")["n_label"].transform("max")
    # rotulo majoritario primeiro; empate no rotulo -> menor rid
    d["majoritario"] = d["n_label"] == d["max_label"]
    d = d.sort_values(["key", "majoritario", "rid"], ascending=[True, False, True])
    keep = set(d.drop_duplicates("key", keep="first")["rid"])

    removed = pool[~pool["rid"].isin(keep)]
    pool_out = pool[pool["rid"].isin(keep)].reset_index(drop=True)
    splits_out = splits[splits["rid"].isin(keep)].reset_index(drop=True)

    DST.mkdir(exist_ok=True)
    pool_out.to_parquet(DST / "v6_pool.parquet", index=False, compression="snappy")
    splits_out.to_parquet(DST / "v6_splits.parquet", index=False, compression="snappy")

    rm_side = splits.set_index("rid").loc[removed["rid"], "split_full_iid"]
    stats = {
        "origem": str(SRC),
        "pool_antes": int(len(pool)),
        "pool_depois": int(len(pool_out)),
        "removidas": int(len(removed)),
        "removidas_por_rotulo": removed["label"].value_counts().to_dict(),
        "removidas_por_lado_full_iid": rm_side.value_counts().to_dict(),
        "removidas_rids": sorted(int(r) for r in removed["rid"]),
        "sha256": {
            "v6_pool.parquet": sha256_file(DST / "v6_pool.parquet"),
            "v6_splits.parquet": sha256_file(DST / "v6_splits.parquet"),
        },
    }
    (DST / "dedup_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"pool {stats['pool_antes']} -> {stats['pool_depois']} "
          f"(removidas {stats['removidas']})")
    print(f"por lado (full_iid): {stats['removidas_por_lado_full_iid']}")
    print(f"por rotulo: {stats['removidas_por_rotulo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
