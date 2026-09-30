"""
Carga, grupos e splits do FakenewsBR.

Modulo compartilhado por encoder.py, embed.py e score.py. Centraliza as decisoes
que a EDA mostrou serem criticas:

  - `text_no_url` como coluna de texto (preserva caixa e acentos; o modelo e *cased*)
  - definicao explicita de GRUPO (proveniencia), porque em 58% da base o
    `dataset_name` determina o rotulo e isso e o maior vazamento do projeto
  - features estilisticas em TAXA, nao em contagem bruta, para nao reintroduzir
    o vies de comprimento
  - splits IID (estratificados por grupo x rotulo) e OOD (leave-one-channel-out)
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_CSV = "data/FakenewsBR_sanitized.csv"  # relativo a raiz do repo
TEXT_COL = "text_no_url"

# Grupos onde o rotulo NAO e determinado pela origem (classe minoritaria >= 15%)
# E que tem massa suficiente (>= 200 linhas). Treinar a cabeca de decisao aqui e
# o que impede o modelo de aprender "estilo de alegacao de agencia => fake".
# `Fake.br_raw` (39 linhas) fica de fora: sob ponderacao por celula, 27 exemplos
# de treino receberiam o mesmo peso total que os 2.506 do `Fake.br`.
BALANCED_GROUPS = [
    "Fake.br", "FakeWhatsApp.BR_2018", "COVID19.BR", "COVID19.BR_raw", "LLM4BR_300",
]

# Grupos degenerados: uma unica classe. Servem so como avaliacao OOD.
DEGENERATE_GROUPS = ["fakes", "true", "MuMiN-PT", "MuMiN-PT_raw"]

CHANNEL_OF = {
    "FakeWhatsApp.BR_2018": "whatsapp",
    "Fake.br": "portal", "Fake.br_raw": "portal",
    "COVID19.BR": "covid", "COVID19.BR_raw": "covid",
    "LLM4BR_300": "llm",
    "fakes": "agency_claim",
    "true": "press_true",
    "MuMiN-PT": "social", "MuMiN-PT_raw": "social",
}


def channel_of(group: str) -> str:
    """Canal por grupo, com fallback por prefixo para a expansao v2.

    A v1 nao tem grupos `FC_*`/`NEWS_*`, entao este fallback nao altera
    nenhum resultado da v1. Ele existe para os datasets novos:
      FC_*_VIRAL -> social       (texto viral de WhatsApp/redes)
      FC_*       -> agency_claim (alegacao de checador)
      NEWS_*     -> press_true   (manchete de portal)
    """
    if group in CHANNEL_OF:
        return CHANNEL_OF[group]
    g = str(group)
    if g.startswith("FC_") and g.endswith("_VIRAL"):
        return "social"
    if g.startswith("FC_"):
        return "agency_claim"
    if g.startswith("EXT_"):
        return "external_claim"
    if g.startswith("NEWS_"):
        return "press_true"
    return "other"


# Nº mínimo de linhas e fração mínima da classe minoritária para um grupo novo
# poder treinar a cabeça DFR. A v1 usa a lista fixa `BALANCED_GROUPS`;
# a expansão v2 aceita grupos `FC_*`/`EXT_*` desde que sejam informativos.
NEW_GROUP_PREFIXES = ("FC_", "EXT_", "NEWS_")
MIN_BALANCED_N = 200
MIN_MINORITY_FRAC = 0.15

# Features estilisticas em taxa. `word_len` fica de fora por padrao: e o atalho
# mais contaminado da base (r = -0,13 com fake). Habilitar so para medir o efeito.
STYLE_RATE = ["excl_rate", "quest_rate", "ellip_rate", "uppercase_word_ratio"]
STYLE_LENGTH = ["log_word_len"]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


_RATING_FALSE = {
    "falso", "errado", "fake", "mentira", "montagem", "insustentavel",
    "sem registro", "satira",
}
_RATING_HARD = {
    "enganoso", "enganador", "distorcido", "sem contexto", "fora de contexto",
    "exagerado", "impreciso", "esticado", "nao_e_bem_assim", "nao e bem assim",
    "verdadeiro, mas", "verdadeiro mas", "subestimado", "superestimado",
}
_RATING_TRUE = {"verdadeiro", "certo", "praticamente certo"}


def _rating_class(raw) -> str:
    """Normaliza factcheck_rating (caixa e acento inconsistentes) em 3 classes."""
    if not isinstance(raw, str) or not raw.strip():
        return "none"
    k = _strip_accents(raw.strip().lower())
    if k in _RATING_HARD:
        return "hard"
    if k in _RATING_FALSE:
        return "false_pure"
    if k in _RATING_TRUE:
        return "true_rating"
    return "other"


def load(csv: str = DEFAULT_CSV, include_length: bool = False,
         provenance_csv: str | None = None,
         labels_csv: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv, low_memory=False)
    if labels_csv:
        # treina somente nas linhas com rotulo de treino (camadas v1/checker/
        # checker_match/llm_local de alta confianca/corroboracao); as linhas
        # por procedencia (NEWS_*) ficam de fora por terem train_label vazio
        lab = pd.read_csv(labels_csv, low_memory=False)
        cols = [c for c in ("rid", "train_label", "label_tier", "auto_label",
                            "confidence", "method") if c in lab.columns]
        df = df.merge(lab[cols], on="rid", how="left")
        df["label_orig"] = df["label"]
        df["label"] = df["train_label"]
    df = df[df["label"].isin(["fake", "true"])].copy()
    df = df.dropna(subset=[TEXT_COL])
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]

    df["target"] = (df["label"] == "fake").astype(int)
    df["group"] = df["dataset_name"].astype(str)
    df["channel"] = df["group"].map(channel_of)
    # grupos novos (`FC_*`/`NEWS_*`) so treinam a cabeça DFR se tiverem massa e
    # as duas classes; a lista fixa da v1 permanece exatamente como estava.
    new_prefix = df["group"].str.startswith(NEW_GROUP_PREFIXES)
    info = []
    for _, sub in df.groupby("group"):
        if len(sub) == 0:
            continue
        frac = sub["target"].mean()
        info.append((sub["group"].iloc[0], len(sub),
                     min(frac, 1 - frac) >= MIN_MINORITY_FRAC))
    informative = {g for g, n, ok in info
                   if n >= MIN_BALANCED_N and ok and g.startswith(NEW_GROUP_PREFIXES)}
    df["is_balanced_group"] = df["group"].isin(BALANCED_GROUPS) | (
        new_prefix & df["group"].isin(informative))

    urls = df["url_review"].fillna("") + " " + df["factcheck_url"].fillna("")
    df["is_ptpt"] = urls.str.contains(
        r"poligrafo|observador|eco\.sapo\.pt", case=False, regex=True)
    df["rating_class"] = df["factcheck_rating"].apply(_rating_class)

    if provenance_csv:
        try:
            prov = pd.read_csv(provenance_csv, low_memory=False)
            keep = [c for c in prov.columns if c != "rid"]
            df = df.merge(prov[["rid"] + keep], on="rid", how="left")
        except FileNotFoundError:
            pass

    w = df["word_len"].fillna(0).clip(lower=0)
    denom = w + 1.0
    df["excl_rate"] = df["num_exclamations"].fillna(0) / denom
    df["quest_rate"] = df["num_questions"].fillna(0) / denom
    df["ellip_rate"] = df["num_ellipsis"].fillna(0) / denom
    df["uppercase_word_ratio"] = df["uppercase_word_ratio"].fillna(0.0)
    df["log_word_len"] = np.log1p(w)

    df = df.reset_index(drop=True)
    df.attrs["style_cols"] = STYLE_RATE + (STYLE_LENGTH if include_length else [])
    return df


def style_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> np.ndarray:
    cols = cols if cols is not None else df.attrs.get("style_cols", STYLE_RATE)
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[cols].to_numpy(dtype=np.float32)


def _safe_strat(keys: pd.Series, fallback: pd.Series, min_count: int = 2) -> pd.Series:
    """Torna uma chave de estratificacao utilizavel pelo train_test_split.

    Celulas (grupo x rotulo) raras caem primeiro para o rotulo puro; o que ainda
    sobrar raro e absorvido pela classe majoritaria. Garante que toda classe
    tenha >= min_count membros, senao o sklearn recusa o split.
    """
    out = keys.astype(str).copy()
    vc = out.value_counts()
    rare = out.map(vc) < min_count
    if rare.any():
        out.loc[rare] = fallback.loc[rare].astype(str)
    vc = out.value_counts()
    rare = out.map(vc) < min_count
    if rare.any():
        out.loc[rare] = vc.idxmax()
    return out


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    name: str

    def describe(self) -> str:
        def f(d):
            return f"n={len(d)} fake={d['target'].mean()*100:.1f}%"
        return (f"[{self.name}] treino({f(self.train)}) "
                f"val({f(self.val)}) teste({f(self.test)})")


def iid_split(df: pd.DataFrame, seed: int = 42,
              val_frac: float = 0.15, test_frac: float = 0.15) -> Split:
    """Split estratificado por (grupo x rotulo), nao so por rotulo.

    Estratificar so por rotulo deixa a composicao de origem variar entre os
    splits, o que torna a metrica por grupo incomparavel.
    """
    label = df["label"].astype(str)
    strat = _safe_strat(df["group"] + "|" + label, label, min_count=10)
    tr, tmp = train_test_split(df, test_size=val_frac + test_frac,
                               random_state=seed, stratify=strat)
    rel = test_frac / (val_frac + test_frac)
    strat2 = _safe_strat(strat.loc[tmp.index], label.loc[tmp.index], min_count=2)
    va, te = train_test_split(tmp, test_size=rel, random_state=seed, stratify=strat2)
    return Split(tr.reset_index(drop=True), va.reset_index(drop=True),
                 te.reset_index(drop=True), "iid")


def ood_split(df: pd.DataFrame, held_out_channel: str, seed: int = 42,
              val_frac: float = 0.15) -> Split:
    """Leave-one-channel-out: treina em todos os canais menos um, testa nele.

    E o teste que responde a pergunta real do projeto: um modelo treinado em
    portal de noticias detecta corrente de WhatsApp?
    """
    te = df[df["channel"] == held_out_channel]
    rest = df[df["channel"] != held_out_channel]
    if len(te) == 0 or len(rest) == 0:
        raise ValueError(f"canal '{held_out_channel}' vazio ou cobre tudo")
    lab = rest["label"].astype(str)
    strat = _safe_strat(rest["group"] + "|" + lab, lab, min_count=4)
    tr, va = train_test_split(rest, test_size=val_frac, random_state=seed,
                              stratify=strat)
    return Split(tr.reset_index(drop=True), va.reset_index(drop=True),
                 te.reset_index(drop=True), f"ood:{held_out_channel}")


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


def group_balanced_indices(df: pd.DataFrame, seed: int = 42,
                           per_cell: int | None = None) -> np.ndarray:
    """Variante por reamostragem. Mantida para comparacao; prefira os pesos."""
    rng = np.random.default_rng(seed)
    cells = df.groupby(["group", "target"]).indices
    if not cells:
        return np.arange(len(df))
    if per_cell is None:
        per_cell = int(np.median([len(v) for v in cells.values()]))
    out = [rng.choice(idx, size=per_cell, replace=len(idx) < per_cell)
           for idx in cells.values()]
    merged = np.concatenate(out)
    rng.shuffle(merged)
    return merged
