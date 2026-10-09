"""Monta RELATORIO.md a partir dos JSONs produzidos pelo pipeline."""
from __future__ import annotations

import datetime as dt
import json
import os

ART = "models/artifacts"
# Gera so a folha de dados. A analise curada vive em RELATORIO.md e nao deve ser
# sobrescrita por uma reexecucao do pipeline.
OUT = "RELATORIO_DADOS.md"

# Baseline linear ja estabelecido no projeto (walkthrough.md).
BASELINE = {"A (ingenua, TF-IDF)": (0.7811, 0.8413),
            "B (higienizada, TF-IDF)": (0.7670, 0.8292)}

RUNS = [
    ("score_frozen.json", "BERTimbau congelado + estilo"),
    ("score_frozen_nostyle.json", "BERTimbau congelado, sem estilo"),
    ("score_finetuned.json", "BERTimbau fine-tuned + estilo"),
]


def load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:.{nd}f}"
    return str(v)


def table(rows: list[dict]) -> str:
    cols = [("split", "split"), ("variant", "variante"),
            ("macro_f1_balanced", "macro-F1 (bal.)"),
            ("worst_group_balanced", "pior-grupo"),
            ("ece_balanced", "ECE (bal.)"),
            ("acc", "acc (total)"), ("macro_f1", "macro-F1 (total)"),
            ("pr_auc", "PR-AUC"), ("hard_acc", "limítrofes")]
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(fmt(r.get(k)) for k, _ in cols) + " |")
    return "\n".join([head, sep, *body])


def main() -> int:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    state = load(f"{ART}/pipeline_state.json") or {}

    parts: list[str] = []
    parts.append(f"# Relatório — Score de Confiança FakenewsBR\n")
    parts.append(f"Gerado automaticamente em {ts}.\n")

    parts.append("## Baseline a superar\n")
    parts.append("| modelo | acurácia | F1(fake) |\n|---|---|---|")
    for k, (a, f) in BASELINE.items():
        parts.append(f"| {k} | {a:.4f} | {f:.4f} |")
    parts.append("")
    parts.append("> A comparação honesta é contra a **Variante B** (0,7670), que teve os "
                 "atalhos removidos. A Variante A é o teto contaminado.\n")

    any_run = False
    for fname, title in RUNS:
        rows = load(f"{ART}/{fname}")
        if not rows:
            continue
        any_run = True
        parts.append(f"## {title}\n")
        parts.append(table(rows))
        parts.append("")

        iid = [r for r in rows if r.get("split") == "iid"]
        erm = next((r for r in iid if "ERM" in r.get("variant", "")), None)
        dfr = next((r for r in iid if "DFR" in r.get("variant", "")), None)
        if erm and dfr:
            parts.append(
                f"No split IID, a ingênua tem acurácia total {fmt(erm.get('acc'))} "
                f"contra {fmt(dfr.get('acc'))} da honesta — mas o pior-grupo "
                f"inverte: {fmt(erm.get('worst_group_balanced'))} contra "
                f"**{fmt(dfr.get('worst_group_balanced'))}**. A queda na acurácia "
                f"total é o modelo deixando de responder \"falso\" automaticamente "
                f"para tudo que tem cara de alegação de agência.\n")

        for ch in ("whatsapp", "portal"):
            o = [r for r in rows if r.get("split") == f"ood:{ch}"]
            e = next((r for r in o if "ERM" in r.get("variant", "")), None)
            d = next((r for r in o if "DFR" in r.get("variant", "")), None)
            if e and d:
                parts.append(
                    f"Fora de domínio em `{ch}`: ingênua macro-F1 {fmt(e.get('macro_f1'))} "
                    f"(ECE {fmt(e.get('ece'))}) contra honesta {fmt(d.get('macro_f1'))} "
                    f"(ECE {fmt(d.get('ece'))}).\n")

    if not any_run:
        parts.append("Nenhum resultado encontrado — o pipeline não chegou a produzir JSONs.\n")

    parts.append("## Execução\n")
    parts.append("| etapa | status | minutos |\n|---|---|---|")
    for k, v in state.items():
        if isinstance(v, dict):
            parts.append(f"| {k} | {v.get('status')} | {v.get('minutos', '—')} |")
    parts.append("")

    parts.append("## Como ler estes números\n")
    parts.append(
        "- **pior-grupo** é a métrica de manchete, restrita a grupos com massa "
        "estatística (n ≥ 100 e classe minoritária ≥ 20). Acurácia média é "
        "enganosa porque 58,4% da base tem o rótulo determinado pela origem.\n"
        "- As colunas **(bal.)** são medidas apenas nos grupos onde a origem não "
        "prediz o rótulo (`Fake.br`, `FakeWhatsApp.BR_2018`, `COVID19.BR`, "
        "`LLM4BR_300`). São as que valem: é o único regime em que a pergunta "
        "\"o modelo entende desinformação?\" está bem-posta.\n"
        "- As colunas **(total)** incluem os subsets degenerados. Servem para "
        "mostrar o contraste, não para julgar o modelo.\n"
        "- O score é calibrado sob **prior balanceado** (~47% fake), não sob os "
        "71,5% da base — esse número é artefato de compilação, não prevalência "
        "real. Consequência: `ECE (bal.)` é a medida válida de calibração; o ECE "
        "sobre a base inteira penaliza uma escolha deliberada. Para um fluxo de "
        "prevalência conhecida π, aplique correção de prior por fora.\n"
        "- Espera-se que **DFR tenha acurácia total menor e pior-grupo maior** "
        "que ERM. Isso é a troca aceita pelo projeto, não um defeito.\n"
        "- `limítrofes` é a acurácia nas meias-verdades (`Enganoso`, `Distorcido`, "
        "`Fora de contexto`).\n")

    parts.append("## Limitação conhecida\n")
    parts.append(
        "Os números **fora de domínio da linha fine-tuned não são OOD honestos**. "
        "O encoder é ajustado no split IID, que contém todos os canais — quando "
        "depois avaliamos com `whatsapp` ou `portal` \"retidos\", o encoder já viu "
        "aqueles textos com rótulo. Só a cabeça foi retida, não o encoder.\n\n"
        "Os OOD da linha **congelada são válidos** (o encoder nunca viu rótulo "
        "nenhum). Para OOD honesto com fine-tuning seria preciso reajustar o "
        "encoder uma vez por canal excluído: ~3,8 h por canal nesta máquina.\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"relatorio escrito em {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
