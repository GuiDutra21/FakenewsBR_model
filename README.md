# FakenewsBR — prestação de contas (dataset v6 + BERTimbau)

Dataset em português para **detecção de desinformação**, com rótulos em
camadas que separam vereditos de checadores de manchetes rotuladas só por
procedência — e o fine-tuning do **BERTimbau** sobre ele, com protocolo
anti-atalho (splits por cluster, DFR, calibração Platt, avaliação por grupo).

## Resultado do modelo final

**Modelo:** BERTimbau base (`neuralmind/bert-base-portuguese-cased`) com
ajuste fino sobre o pool v6 deduplicado, pesos DFR e calibração Platt.
Run `D1_dedup_s42`, com os pesos em
[`pipeline/experiments/runs/D1_dedup_s42/best/`](pipeline/experiments/runs/D1_dedup_s42/best/).

**Teste:** 13.661 textos nunca vistos no treino (73% fake), com limiar de 0,5.

| métrica | valor | leitura |
|---|---:|---|
| **Acurácia** | **82,5%** | acerta 4 de cada 5 textos |
| **Macro-F1** | **0,785** | média do F1 das duas classes (não é inflada pela maioria fake) |
| **Pior grupo (macro-F1)** | **0,617** | desempenho na fonte mais difícil (LIAR-BR) |
| ROC-AUC / PR-AUC | 0,896 / 0,959 | qualidade da ordenação, sem depender do limiar |
| ECE | 0,024 | a probabilidade é confiável (diz 80% → acerta cerca de 80%) |

**Por classe**

| classe | n | acertos da classe (recall) | precisão | F1 |
|---|---:|---:|---:|---:|
| fake | 10.015 | 85,8% | 89,9% | 0,878 |
| true | 3.646 | 73,6% | 65,3% | 0,692 |

**Comparação no mesmo teste**

| modelo | acurácia | macro-F1 | pior grupo | ECE |
|---|---:|---:|---:|---:|
| Maioria (sempre "fake") | 0,733 | 0,423 | 0,272 | 0,267 |
| TF-IDF + regressão logística | 0,803 | 0,740 | 0,507 | 0,047 |
| v1 antigo¹ | 0,805 | 0,694 | 0,301 | 0,063 |
| BERTimbau v6 R0 (antes do dedup) | 0,806 | 0,768 | 0,541 | 0,032 |
| **BERTimbau v6 D1_dedup (final)** | **0,825** | **0,785** | **0,617** | **0,024** |

¹ 30% do teste estava no treino do v1. Só na parte limpa, o v1 cai para 0,765 / 0,597.

**Pontos fracos**

- A classe **true** é a mais fraca: o modelo acerta 74% dos textos verdadeiros, contra 86% dos falsos.
- Em **pt-PT** (Polígrafo), o macro-F1 é 0,689, contra 0,806 em pt-BR.
- O pior grupo é o **LIAR-BR**, com 0,617: são alegações que exigem conhecimento externo.
- O D1_dedup foi rodado só com a seed 42. As seeds 43 e 44 ainda faltam.

O relatório completo está em
[`docs/relatorio_tecnico/RELATORIO_TECNICO.md`](docs/relatorio_tecnico/RELATORIO_TECNICO.md).
A avaliação reproduzível fica no notebook em [`notebook/`](notebook/), que recalcula
todas as métricas acima.

## Dataset em números

- **v6:** 297.672 linhas; pool de treino verificado com **91.080** (66.772 fake / 24.308 true).
- 36 grupos de origem, dos quais 11 têm rótulo constante (39.373 linhas).

## Mapa do repositório

```
README.md / LICENSE          este arquivo; código MIT
data/                        CSVs/JSONs (os grandes ficam NO RELEASE, ver §Dados)

pipeline/                    >>> LINHA ATUAL (modelo final D1_dedup_s42) <<<
  prepare/                   prepare.py (CSVs -> parquet + splits), dedup_exatas.py, data.py
    processed/               pool v6 + splits (gerado)
    processed_dedup/         pool v6 sem duplicatas exatas (gerado; usado pelo D1)
  train/                     train_bertimbau_v6.py (+ cópia de trabalho do notebook)
  experiments/               run_experiments.py (fila), analisar.py, status.json
    runs/<run>/              métricas/predições de cada run; pesos só em D1_dedup_s42/best/
    logs/                    logs de treino (não versionados)
notebook/                    pacote autocontido de avaliação (notebook + modelo/ dados/
                             resultados/ comparacao/); roda sozinho ou no Colab

docs/
  relatorio_tecnico/         RELATORIO_TECNICO .md/.pdf/.docx (entrega atual)
  DATASET_CARD.md, SOURCES_AND_LICENSES.md, ...
  v4/, v6/                   notas de build, planos e code reviews por versão
  RELATORIO*.md              prestação de contas anterior (pipeline legado)
investigation/               8 trilhas de análise + expansion/ (pipeline v2→v6 do dataset)
plots/                       figuras da EDA
graphify-out/                grafo do código gerado por ferramenta

legacy/                      gerações anteriores (referência; não é o fluxo atual)
  models/                    pipeline v1 (data/embed/encoder/score/evaluate) + artifacts
  models/v4/                 prepare/train/explore/colab da v4 + processed + métricas
  models/v6/                 run R0 (sem dedup), baseline TF-IDF, diagnose, colab
  models/v6/autoresearch/    23 runs + 4 controles + VERDICT/RELATORIO_MANHA/REVIEW_CRITICA
  models/v6/compare/         v1×v6, TF-IDF, predições e métricas
  scripts/                   sanitize_dataset.py (v1), eda_analysis.py
external/                    clones de referência (NÃO versionado, não redistribuir)
```

Código legado que importa `models` roda com `PYTHONPATH=legacy` a partir da raiz.

## Dados

Versões: **v1** 39.466 (base histórica AKCIT-FN) · **v2** 215.640
(checadores/portais/G1/ClaimReview) · **v3** 242.913 (+LIAR-BR, AVERITEC-BR) ·
**v4** 291.521 (+Polígrafo PT-PT) · **v6** **297.672** + rótulos em camadas
(detalhes em `docs/DATASET_CARD.md`).

Pool de treino v6 (`train_label`, de `prepare_stats.json`):

| item | valor |
|---|---:|
| linhas sanitizadas | 297.672 (tiers: provenance 206.592, checker 44.347, v1 39.466, checker_match 4.166, llm_local 3.092, corroborated 9) |
| pool com rótulo | **91.080** (fake 66.772 / true 24.308 → 73,31%) |
| grupos / constantes | 36 / **11 (39.373 linhas)** |
| grupos informativos (DFR) | **36.896** (51,98% fake): COVID19.BR, COVID19.BR_raw, EXT_AVERITECBR, EXT_LIARBR, FC_POLIGRAFO, Fake.br, FakeWhatsApp.BR_2018, LLM4BR_300 |
| splits (`full_iid`, cluster, seed 42) | treino 63.753 / val_sel 9.108 / val_calib 4.554 / teste 13.661 |
| near-dups cruzando splits (MinHash ≥0,8) | **0 pares** |
| OOD `ood_wa` (WhatsApp retido) | treino 71.986 / teste 6.379 |

Entradas com SHA256 registrado (`legacy/models/v6/processed/prepare_stats.json`):
`data/FakenewsBR_sanitized_v6.csv` `d3364da9…`,
`data/FakenewsBR_v6_labels.csv` `8a0c36ff…`,
`data/FakenewsBR_v6_provenance.csv` `5413be2c…`.

**Onde baixar (Release `modelo-v6-R0`)**: `FakenewsBR_sanitized_v6.csv` (pool),
`FakenewsBR_v6_labels.csv`, `FakenewsBR_v6_provenance.csv`,
`FakenewsBR_v6_public.csv` (PII mascarada) e o bruto
`FakenewsBR_factchecked.csv`. Os CSVs grandes **não** vão para o git
(quota do LFS gratuito: 1 GB) — copie para `data/` e rode o §Reprodução.

## Análise exploratória — achados

1. **Atalho de procedência**: 11 grupos constantes (ex.: `fakes` 20.347 linhas
   100% fake, `true` 2.710 100% true). Quem treina ERM aprende "origem",
   não semântica. Resposta: treino **DFR** (só grupos informativos, pesos
   por célula grupo×rótulo, clip 25) + métrica manchete = **pior-grupo**.
2. **Só o prior da fonte já "acerta" 0,8055** (`diagnose_source_leak`):
   prever o grupo pelo texto (acc 0,642) e aplicar o prior do grupo dá
   acc 0,8055 / macro 0,7474 — mas **pior-grupo 0,2717**. É o teto do
   trapaceiro e o piso honesto de comparação.
3. **Contaminação do v1**: 30,36% do teste v6 estava no treino do v1
   (4.147/13.661); no OOD, 69,98%. Qualquer número do v1 fora do split
   dele é inválido — por isso a comparação usa o teste v6 + recorte limpo.
4. **TF-IDF é um baseline forte**: LR + Platt chega a 0,8032/0,7404.
   O BERT só se paga se passar disso **no pior grupo**, não na média.

## Histórico de modelos

> Registro das versões anteriores. Os números abaixo são do **run original do
> R0 em CPU**, que é o do Release. O R0 foi re-treinado depois, e a comparação
> atual, no mesmo teste do modelo final, está em
> [Resultado do modelo final](#resultado-do-modelo-final).

### R0 — BERTimbau no pool v6 completo (run anterior ao dedup; pesos no Release `modelo-v6-R0`)

`neuralmind/bert-base-portuguese-cased` (12 camadas, 768, 100M),
ml192, freeze 6, lr 2e-5, warmup 0,1, batch 32, **2 épocas em CPU**
(24.667 s, `torch 2.14.0+cpu`, `transformers 5.17.0`), DFR cell/clip 25,
**Platt** (val_calib n=1.844; ECE 0,0495→0,0159), seleção por pior-grupo,
seed 42. Teste `full_iid` (n=13.661):

| métrica | valor |
|---|---:|
| acc | **0,8163** |
| macro-F1 / F1-fake | **0,7782 / 0,8701** |
| PR-AUC / ROC-AUC | 0,9563 / 0,8902 |
| Brier / ECE | 0,1198 / **0,0188** |
| **pior-grupo** (publisher `ext_liarbr`, n=1.049) | **0,5749** |
| canais confiáveis: portal / covid / whatsapp / external_claim / agency_claim | 0,9718 / 0,8008 / 0,6652 / 0,6234 / 0,6815 |
| pt-PT | 0,6992 |

Artefatos no repo: `legacy/models/v6/artifacts/full_iid_ml192_f6_on_seed42/`
(`metrics.json`, `per_group.csv`, `predictions.csv`, `calibration.json`,
logs). **Pesos (`best/`) só no Release.**

### v1 antigo × v6 (mesmo teste, `legacy/models/v6/compare/`)

| modelo × teste | acc | macro-F1 | pior-grupo | ECE |
|---|---:|---:|---:|---:|
| v1 no teste v6 (contaminado) | 0,8051 | 0,6943 | 0,3008 | 0,0625 |
| v1 no recorte limpo (n=9.514) | 0,7650 | 0,5974 | 0,3008 | 0,0959 |
| **R0 no teste v6** | **0,8163** | **0,7782** | **0,5749** | **0,0188** |
| v1 no OOD limpo (n=1.915) | 0,7608 | 0,7598 | 0,7598 | 0,0696 |

Leitura: no recorte limpo o v1 desaba no pior grupo (0,30); o R0 quase
dobra isso (0,57) e calibra 3–5× melhor. OOD do R0 (R1) ainda pendente.

### TF-IDF + LR + Platt (`legacy/models/v6/compare/baseline_tfidf/`)

Maioria: 0,7331/0,4230. LR calibrado@0,5: **0,8032 / 0,7404 / pior 0,5070 /
ECE 0,0468**. Régua honesta: o R0 supera o linear em +0,038 de macro-F1 e
+0,068 no pior grupo, com menos da metade do ECE.

### autoresearch — 23 runs + 4 controles + review crítica

Orçamento curto e fixo (5.000/3.000 linhas, 100 steps, seed 42;
métrica = média de macro-F1 nos grupos confiáveis; ruído de seed ±0,02).
Top: `freeze0+ml256+informative` 0,6244 ≈ `ml256+informative` 0,6226 ≈
`freeze4` 0,6220 ≈ `informative` 0,6172 (tudo dentro do ruído entre si);
`ml128` 0,5790; `mask` 0,5360 (pior-grupo inalterado: 0,3254);
`classweights` 0,5263; baseline 0,4437. Descartados: batch16, lr 1e-5.

A **review crítica** (`REVIEW_CRITICA.md`) achou 2 erros bloqueantes no
desenho: orçamento em *steps* (incomparável entre batch/ml diferentes) e
confundidor de épocas no eixo dados (+0,17 anunciados eram ~1,58 vs 0,64
época). Os **4 controles 1-época** fecharam o segundo:
full 0,5134 · informative 0,5563 · **informative+mask 0,6043**
(pior 0,5011) · informative+ml256 0,4857.
**Veredicto corrigido**: efeito real do pool curado ≈ +0,04 (não +0,17);
`freeze_layers=6` (empate, mais barato); `mask_entities` é o candidato a
confirmar no run completo; ml256 não se paga. Nada aqui substitui o run
full — indica a direção dele.

## Reprodução

Linha atual (modelo final `D1_dedup_s42`), a partir da raiz do repositório:

```bash
# 1) dados do Release -> data/
# 2) preparar (valida splits, near-dups, DFR; ~3 min em CPU) -> pipeline/prepare/processed/
python pipeline/prepare/prepare.py --force
# 3) remover duplicatas exatas do treino -> pipeline/prepare/processed_dedup/
python pipeline/prepare/dedup_exatas.py
# 4) fila de experimentos (BASE/T1..T9/D1_dedup) -> pipeline/experiments/runs/
python pipeline/experiments/run_experiments.py --so D1_dedup_s42
python pipeline/experiments/analisar.py
# 5) avaliar: abrir notebook/avaliacao_modelo_fakenewsbr_v6.ipynb
```

Run R0 legado (sem dedup) e comparações:

```bash
python legacy/models/v6/train_bertimbau_v6.py --data legacy/models/v6/processed/v6_pool.parquet \
  --splits legacy/models/v6/processed/v6_splits.parquet --split-col full_iid \
  --max-length 192 --freeze-layers 6 --epochs 2 --out legacy/models/v6/artifacts/R0
python legacy/models/v6/baseline_tfidf.py
python legacy/models/v6/diagnose_source_leak.py
python legacy/models/v6/compare_v1_vs_v6.py   # requer os pesos do v1
```

Notebook Colab (T4) do R0: `legacy/models/v6/colab_bertimbau_v6.ipynb` (21 células).
Pipeline legado (score/Trilha A/B): `docs/pipeline-legado.md`.

## O que foi podado e por quê

Para caber no GitHub (LFS gratuito = 1 GB), o commit leva **código, docs,
métricas, predições e parquets processados**; o Release leva os binários
(R0 `best/`, CSVs). Podado do disco e fora do git: 19 pastas `last/`
(pesos duplicados do `best/`), 76 estados de otimizador (`optimizer.pt`
etc.), 18 `best/` de smokes descartáveis, ONNX/embeddings do pipeline
legado e os pesos do v1 (superados; métricas da comparação preservadas).

Na reorganização de 09/10/2026 também saíram do disco (~25 GB): as 21 pastas
`last/` dos experimentos e do R0, os `best/` de todos os runs exceto
`D1_dedup_s42` (métricas e predições preservadas) e os zips `notebook.zip` /
`model.zip` (geráveis a partir das pastas).

**Optimizer states: não compensa subir.** Servem só para *retomar* um
treino interrompido; o R0 terminou (melhor época salva + calibrada) e
reproduzir do zero é determinístico (seed 42 + código + SHAs acima).
Cada um custa 329 MB e amarra às versões exatas das libs — peso morto.

## Limitações e próximos passos

- R0b (mesmo treino **sem** DFR, teste do valor do DFR): pausado a 1 época,
  retomável (`--resume`).
- R1 (OOD `ood_wa`) e o run completo `informative+mask`: pendentes.
- Pior grupo (0,57, `ext_liarbr`) e canais `whatsapp`/`external_claim`
  (~0,62–0,67) seguem como fronteira.
- Conteúdo de terceiros sob termos das fontes: ler
  `docs/SOURCES_AND_LICENSES.md` antes de redistribuir.
