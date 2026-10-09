"""Verificador seletivo de manchetes (NEWS_*) — sem chave de API.

Estado da arte (ver METODOS_CHECAGEM.md): extrair a alegação -> recuperar
evidência -> NLI/stance -> **abster-se quando a evidência é fraca**. Forçar
fake/true em tudo é o erro que a literatura isola (selective fact-checking).

Fontes de evidência sem chave, usadas aqui:
  1. corpus local ClaimReview (`feed.jsonl` + `gfc.jsonl`)  [match offline]
  2. Google News RSS (`news.google.com/rss/search`)          [retrieval web]
  3. GDELT DOC API                                           [retrieval web]
  4. checadores detectados nos resultados (Aos Fatos, Lupa, AFP, Boatos...)

Regras de rótulo (conservadoras):
  - checagem casada (título do resultado de checador com sobreposição alta):
      veredito parseado do título -> fake/true/hard (confiança alta)
  - senão, corroboração: >= K domínios reputáveis independentes descrevendo o
      mesmo evento -> true (confiança média)
  - senão -> unknown (abstenção)

Uso:
    python -m investigation.expansion.verify_news --limit 500
    python -m investigation.expansion.verify_news --limit 0   # todas (longo)
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from . import dedup as dd
from . import schema as sch

RAW = Path("investigation/expansion/raw")
PROC = Path("investigation/expansion/processed")
UA = {"User-Agent": "FakenewsBR-research/2.0 (+https://github.com/thiago-cg/FakenewsBR)"}

CHECKER_NAMES = {
    "aos fatos": "aosfatos", "aosfatos": "aosfatos",
    "lupa": "lupa", "agencia lupa": "lupa",
    "boatos.org": "boatos", "e-farsas": "efarsas", "e farsas": "efarsas",
    "afp checamos": "afp", "afp": "afp",
    "estadao verifica": "estadao", "estadão verifica": "estadao",
    "comprova": "comprova", "projeto comprova": "comprova",
    "uol confere": "uolconfere", "g1 fato ou fake": "g1", "fato ou fake": "g1",
    "poligrafo": "poligrafo", "polígrafo": "poligrafo",
    "observador": "observador", "verificamos": "generico",
    "checamos": "generico", "factcheck": "generico",
}

REPUTABLE = {
    "g1.globo.com", "globo.com", "folha.uol.com.br", "estadao.com.br",
    "uol.com.br", "cnnbrasil.com.br", "agenciabrasil.ebc.com.br",
    "bbc.com", "reuters.com", "apnews.com", "dw.com", "elpais.com",
    "poder360.com.br", "brasildefato.com.br", "oeco.org.br", "eco.sapo.pt",
    "nexojornal.com.br", "valor.globo.com", "oglobo.globo.com",
    "metropoles.com", "terra.com.br", "r7.com", "band.com.br",
    "sbtnews.com.br", "recordtv.r7.com", "correiobraziliense.com.br",
    "em.com.br", "zerohora.clicrbs.com.br", "gauchazh.clicrbs.com.br",
}

OPINION_MARKERS = re.compile(
    r"\b(an[aá]lise|opini[aã]o|coluna|editorial|artigo|ensaio|reflex[õo]es|"
    r"cr[ôo]nica|resenha|entrevista|debate|podcast|vídeo|fotos|ao vivo|"
    r"elei[çc][õo]es 20\d\d|hor[óo]scopo|previs[ãa]o do tempo)\b", re.I)
CLAIM_VERBS = re.compile(
    r"\b(e|é|foi|foram|ser[áa]|vai|v[ãa]o|tem|t[êe]m|havia|anunciou|"
    r"aprovou|sancionou|assinou|lan[çc]ou|confirmou|negou|disse|afirmou|"
    r"caiu|subiu|aumentou|diminuiu|mata|morre|morreu|pro[íi]be|libera|"
    r"autoriza|bloqueia|recusa|veta|diz|mostra|revela|denuncia|acusa)\b", re.I)


def is_claim_like(text: str) -> bool:
    t = (text or "").strip()
    n = len(t.split())
    if n < 5 or n > 40:
        return False
    if t.endswith("?"):
        return False
    if OPINION_MARKERS.search(t):
        return False
    if not CLAIM_VERBS.search(t):
        return False
    return True


def _domains_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------- Google News RSS
def gnews(query: str, session: requests.Session, timeout: int = 25) -> list[dict]:
    u = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(query)
         + "&hl=pt-BR&gl=BR&ceid=BR:pt-419")
    try:
        r = session.get(u, headers=UA, timeout=timeout)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        title = html.unescape(it.findtext("title") or "")
        link = it.findtext("link") or ""
        src_el = it.find("source")
        source = html.unescape(src_el.text) if src_el is not None else ""
        src_url = src_el.get("url", "") if src_el is not None else ""
        dom = _domains_of(src_url) or _domains_of(link)
        out.append({"title": title, "link": link, "source": source,
                    "domain": dom, "pub": it.findtext("pubDate") or ""})
    return out


def _factchecker(item: dict) -> str | None:
    blob = (item.get("source", "") + " " + item.get("domain", "") + " "
            + item.get("title", "")).lower()
    for name, key in CHECKER_NAMES.items():
        if name in blob:
            return key
    return None


_VERDICT_FALSE = re.compile(
    r"\be falso que\b|\be fake que\b|\bnao e verdade\b|\bnao procede\b|"
    r"\be mentira\b|\bdesmente\b|\bchecamos que e falso\b|\be falso\b")
_VERDICT_TRUE = re.compile(
    r"\be verdade que\b|\be verdadeiro que\b|\bconfirmado\b|\bprocede\b|"
    r"\bchecamos que e verdade\b|\be verdade\b")
_STOP = set("""a o e de da do das dos em no na nos nas um uma para por com que
se sobre entre apos ate como mais menos muito pouco ja nao sim ao aos as os
the of and to in for on with is are was were this that from by as at be or an
it its diz disse afirma afirmou segundo ainda tambem pode pode ser vai vao
foi sao era sera tem tinham ha havia""".split())


def _content_tokens(text: str) -> set[str]:
    """Tokens de conteudo, sem stopwords nem palavras de veredito."""
    toks = {w for w in dd.normalize_text(text).split() if len(w) >= 4}
    toks -= _STOP
    toks -= {"falso", "falsa", "fake", "verdade", "verdadeiro", "verdadeira",
             "mentira", "desmente", "desmentido", "checamos", "procede",
             "confirmado", "nao", "procede"}
    return toks


def _claim_overlap(a: str, b: str) -> tuple[float, int]:
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0, 0
    inter = len(ta & tb)
    return inter / min(len(ta), len(tb)), inter


def _verdict_from_title(title: str) -> str | None:
    t = dd.normalize_text(title)
    if _VERDICT_FALSE.search(t):
        return "fake"
    if _VERDICT_TRUE.search(t):
        return "true"
    return None


# ------------------------------------------------------------------- matching
def _overlap(a: str, b: str) -> float:
    ta = {w for w in dd.normalize_text(a).split() if len(w) >= 4}
    tb = {w for w in dd.normalize_text(b).split() if len(w) >= 4}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


@dataclass
class QueryCache:
    path: Path
    data: dict = field(default_factory=dict)
    _loaded: bool = False

    def load(self):
        if self.path.exists() and not self._loaded:
            for line in self.path.open("r", encoding="utf-8"):
                try:
                    r = json.loads(line)
                    self.data[r["q"]] = r["items"]
                except json.JSONDecodeError:
                    continue
        self._loaded = True

    def get(self, q: str) -> list[dict] | None:
        self.load()
        return self.data.get(q)

    def put(self, q: str, items: list[dict]):
        self.data[q] = items
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"q": q, "items": items}, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ veredito
_ORD = {"primeira": 1, "segunda": 2, "terceira": 3, "quarta": 4,
        "quinta": 5, "sexta": 6, "setima": 7, "oitava": 8, "nona": 9,
        "decima": 10, "primeiro": 1, "segundo": 2, "terceiro": 3,
        "quarto": 4, "quinto": 5, "sexto": 6, "setimo": 7, "oitavo": 8,
        "nono": 9, "decimo": 10}


def _numbers(text: str) -> set[int]:
    """Numeros citados (digitos e ordinais por extenso)."""
    t = dd.normalize_text(text)
    out = {int(x) for x in re.findall(r"\b\d+\b", t)}
    out |= {_ORD[w] for w in t.split() if w in _ORD}
    return out


def _same_numbers(a: str, b: str) -> bool:
    na, nb = _numbers(a), _numbers(b)
    if not na:
        return True
    return bool(na & nb)


def _parse_pub(pub: str):
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(pub)
    except Exception:  # noqa: BLE001
        return None


def _near_date(date_iso, pub: str, max_days: int = 45) -> bool:
    if not date_iso or not pub:
        return True
    try:
        ts = pd.Timestamp(date_iso)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
    except Exception:  # noqa: BLE001
        return True
    p = _parse_pub(pub)
    if p is None:
        return True
    if p.tzinfo is None:
        p = p.tz_localize("UTC")
    return abs((ts - p).days) <= max_days


def classify(text: str, own_domain: str, items: list[dict],
             min_corrob: int = 3, sims: list[float] | None = None,
             sim_fc: float = 0.80, sim_corr: float = 0.75,
             date_iso: str | None = None) -> dict:
    """Retorna dict com auto_label, confidence, method, evidence.

    Com `sims` (cosseno do encoder local por item): alta precisao exige
    similitude >= sim_fc, veredito explicito no titulo do checador e
    consistencia de numeros citados. Sem encoder, cai no match lexical
    conservador.
    """
    def sim(i: int, item: dict) -> tuple[float, int]:
        if sims is not None:
            return float(sims[i]), 0
        ov, inter = _claim_overlap(text, item.get("title", ""))
        return ov, inter

    best_fc = None
    for i, it in enumerate(items):
        fc = _factchecker(it)
        if not fc:
            continue
        s, inter = sim(i, it)
        ok = (s >= sim_fc) if sims is not None else (s >= 0.6 and inter >= 3)
        if not ok:
            continue
        if not _same_numbers(text, it.get("title", "")):
            continue
        v = _verdict_from_title(it.get("title", ""))
        if v:
            best_fc = {"checker": fc, "verdict": v, "item": it, "sim": s}
            break
    if best_fc:
        return {"auto_label": best_fc["verdict"], "confidence": "alta",
                "method": f"factcheck:{best_fc['checker']}",
                "evidence": best_fc["item"].get("link", ""),
                "evidence_title": best_fc["item"].get("title", "")[:200],
                "sim": best_fc["sim"]}

    doms = set()
    for i, it in enumerate(items):
        s, inter = sim(i, it)
        d = it.get("domain", "")
        ok = (s >= sim_corr) if sims is not None else (s >= 0.6 and inter >= 3)
        if not ok or d in doms:
            continue
        if d not in REPUTABLE or d == own_domain:
            continue
        if not _same_numbers(text, it.get("title", "")):
            continue
        if not _near_date(date_iso, it.get("pub", "")):
            continue
        doms.add(d)
    if len(doms) >= min_corrob:
        return {"auto_label": "true", "confidence": "media",
                "method": "corroboracao", "evidence": ";".join(sorted(doms)),
                "evidence_title": "", "sim": None}
    return {"auto_label": "unknown", "confidence": "-", "method": "abstencao",
            "evidence": "", "evidence_title": "", "sim": None}


# ---------------------------------------------------------------------- run
def load_news(v2: Path) -> pd.DataFrame:
    v1 = pd.read_csv("FakenewsBR_sanitized.csv", usecols=["rid"])
    df = pd.read_csv(v2, low_memory=False)
    new = df.iloc[len(v1):].copy()
    new = new[new["dataset_name"].astype(str).str.startswith("NEWS_")]
    new["own_domain"] = new["url_review"].fillna("").map(_domains_of)
    return new


def _embed_rows(fetch, onnx: str, provider: str):
    """Embeda titulos unicos e devolve lista de cossenos por linha."""
    import numpy as np

    from models.embed import extract

    texts: list[str] = []
    index: dict[str, int] = {}

    def add(t: str) -> int:
        k = dd.normalize_text(t)
        if not k:
            return -1
        if k not in index:
            index[k] = len(texts)
            texts.append(t)
        return index[k]

    head_idx, item_idx = [], []
    for r, items in fetch:
        head_idx.append(add(str(r.text)))
        item_idx.append([add(it.get("title", "")) for it in items])
    print(f"[verify] embedando {len(texts):,} textos unicos "
          f"(encoder local, {provider})", flush=True)
    emb = extract(texts, onnx, provider=provider, batch_size=32,
                  max_length=96, verbose=True)
    out = []
    for i in range(len(fetch)):
        hi = head_idx[i]
        if hi < 0:
            out.append([])
            continue
        hv = emb[hi]
        hn = float(np.linalg.norm(hv))
        sims = []
        for j in item_idx[i]:
            if j < 0:
                sims.append(0.0)
                continue
            v = emb[j]
            den = hn * float(np.linalg.norm(v))
            sims.append(float(hv @ v / den) if den else 0.0)
        out.append(sims)
    return out


def run(v2: Path, out: Path, limit: int, sample_seed: int | None,
        sleep_s: float, min_corrob: int, semantic: bool, onnx: str,
        provider: str, sim_fc: float, sim_corr: float) -> dict:
    news = load_news(v2)
    news = news[news["text"].astype(str).map(is_claim_like)]
    print(f"[verify] {len(news):,} manchetes claim-like de NEWS_*")
    if sample_seed is not None:
        news = news.sample(n=min(limit, len(news)), random_state=sample_seed)
    elif limit:
        news = news.head(limit)
    cache = QueryCache(out.with_suffix(".cache.jsonl"))
    sess = requests.Session()

    # fase 1: recuperacao de evidencia (Google News RSS)
    fetch = []
    t0 = time.time()
    for i, r in enumerate(news.itertuples(index=False)):
        q = str(r.text)[:120]
        items = cache.get(q)
        if items is None:
            items = gnews(q, sess)
            cache.put(q, items)
            time.sleep(sleep_s)
        fetch.append((r, items))
        if (i + 1) % 50 == 0:
            print(f"[verify] retrieval {i+1}/{len(news)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # fase 2: embeddings (opcional)
    sims_by_row: list[list[float] | None] = [None] * len(fetch)
    if semantic:
        try:
            sims_by_row = _embed_rows(fetch, onnx, provider)
        except Exception as e:  # noqa: BLE001
            print(f"[verify] encoder indisponivel ({e}); usando match lexical",
                  flush=True)
            sims_by_row = [None] * len(fetch)

    # fase 3: classificacao seletiva
    rows = []
    for (r, items), sims in zip(fetch, sims_by_row):
        res = classify(str(r.text), r.own_domain, items, min_corrob,
                       sims=sims, sim_fc=sim_fc, sim_corr=sim_corr,
                       date_iso=str(getattr(r, "date_iso", "") or ""))
        rows.append({"rid": int(r.rid), "text": str(r.text)[:200],
                     "publisher": r.dataset_name, **res,
                     "n_resultados": len(items)})
    got = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    got.to_csv(out, index=False)
    n = len(got)
    resumo = {
        "manchetes_claim_like": int(len(news)),
        "semantico": bool(semantic),
        "rotuladas_alta": int((got["confidence"] == "alta").sum()),
        "rotuladas_media": int((got["confidence"] == "media").sum()),
        "abstencao": int((got["auto_label"] == "unknown").sum()),
        "cobertura_pct": round(100 * (n - (got["auto_label"] == "unknown").sum()) / max(n, 1), 2),
        "distribuicao": got["auto_label"].value_counts().to_dict(),
        "por_metodo": got["method"].value_counts().to_dict(),
        "exemplos": got[got["auto_label"] != "unknown"].head(8)[
            ["text", "auto_label", "method", "evidence_title"]].to_dict("records"),
    }
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    out.with_suffix(".json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    return resumo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, default=Path("FakenewsBR_sanitized_v2.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("FakenewsBR_v2_news_verification.csv"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=1.2)
    ap.add_argument("--min-corroboracao", type=int, default=3)
    ap.add_argument("--no-semantic", action="store_true",
                    help="usa apenas match lexical (sem encoder local)")
    ap.add_argument("--onnx", default="legacy/models/artifacts/bertimbau_ft.onnx")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--sim-factcheck", type=float, default=0.80)
    ap.add_argument("--sim-corroboracao", type=float, default=0.75)
    a = ap.parse_args()
    run(a.v2, a.out, a.limit, a.sample_seed, a.sleep, a.min_corroboracao,
        not a.no_semantic, a.onnx, a.provider, a.sim_factcheck,
        a.sim_corroboracao)


if __name__ == "__main__":
    main()
