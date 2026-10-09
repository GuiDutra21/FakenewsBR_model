# Relatorio da manha — autoresearch v6 (CPU)

- Período: 23:03 → 03:16 (252,5 min), 23 experimentos concluídos, 0 falhas.
- Protocolo: subset congelado 5.000 treino / 3.000 avaliação (estratificado por
  grupo×rótulo), orçamento fixo de 100 steps, uma métrica:
  `group_mean_macro_f1` (média de macro-F1 nos grupos confiáveis n>=40,
  minoria>=10 do subset de avaliação). Tudo em `results.tsv` / `VERDICT.md`.
- Ruído medido: baseline 0,4437 vs mesma config com seed 43 = 0,4658 →
  **±0,02**. Diferenças menores que isso não são conclusão.

## Ranking (top 8)

| # | run | grpMean | worst | acc | macroF1 | ECE | min |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 021 freeze_layers=0 (ml256+informative) | 0,6244 | 0,5042 | 0,5397 | 0,5321 | 0,0318 | 21,5 |
| 2 | 020 max_length=256 (informative) | 0,6226 | 0,4998 | 0,5500 | 0,5394 | 0,0212 | 16,5 |
| 3 | 022 freeze_layers=4 (ml256+informative) | 0,6220 | 0,5042 | 0,5507 | 0,5389 | 0,0346 | 18,1 |
| 4 | 015 informative (ml192) | 0,6172 | 0,4870 | 0,5290 | 0,5174 | 0,0389 | 13,6 |
| 5 | 023 freeze_layers=8 (ml256+informative) | 0,6159 | 0,4876 | 0,5417 | 0,5318 | 0,0118 | 14,9 |
| 6 | 019 max_length=128 (informative) | 0,5790 | 0,4354 | 0,5047 | 0,4984 | 0,0644 | 10,7 |
| 7 | 008 mask (full) | 0,5360 | 0,3254 | 0,7703 | 0,6340 | 0,1052 | 9,3 |
| 8 | 016 classweights (full) | 0,5263 | 0,4402 | 0,7290 | 0,6702 | 0,0589 | 9,1 |
| 20 | 001 baseline (full, DFR, ml192) | 0,4437 | 0,3277 | 0,7473 | 0,5415 | 0,0967 | 9,6 |
| 23 | 017 batch16 | 0,3719 | 0,3025 | 0,7400 | 0,4445 | 0,0872 | 6,0 |

## Achados por eixo

| eixo | efeito (vs baseline, mini-regime) | leitura |
|---|---|---|
| `informative` (só grupos sem confundimento) | **+0,17** | maior alavanca, **confundida com épocas** (ver abaixo) |
| `max_length=256` | +0,065 (full) / +0,005 (informative) | 128 dói (-0,04); 256 ≈ 192 dentro do ruído |
| `freeze 0/4/8` | ≤ +0,02 (single) | no combo ml256, freeze 0/4/6 ficam a ≤0,005; freeze 0 custa +30–60% tempo |
| `mask_entities` | +0,092 (full, 1 seed) | promissor; não testado em informative |
| `classweights` | +0,083 (full, 1 seed) | melhora grupo, pressiona calibração; DFR+Platt é preferível |
| `warmup 0` | +0,074 (full, 1 seed) | promissor |
| `lr=3e-5` | +0,069 (full, 1 seed) | promissor; lr=1e-5 é ruim (-0,062) |
| `lr=1e-5` | -0,062 | descartar |
| `batch=16` | -0,072 | descartar |
| `clip 0,5` / `wd 0` | ≈ 0 | sem efeito |

## O confundidor central

Com steps fixos, quem tem menos dados vê mais épocas:
- subset **full** = 5.000 linhas → 100 steps = 3.200 amostras = **0,64 época**;
- subset **informative** = 2.027 linhas → 100 steps = **1,58 épocas**.

Logo, o +0,17 do `informative` mistura "dado curado" com "treinou 2,5x mais
por dado". Controles em execução (`run_controls.py`, ~05:15–05:55) igualam
~1,0 época: `full_1ep` (156 steps), `informative_1ep` (63), `mask_1ep` (63),
`ml256_1ep` (63). O veredicto final usa esses números.

## Veredicto (indício, não modelo final)

1. **Curar o dado é a aposta principal**: treinar nos grupos informativos
   (36.896 linhas) é consistente com o plano original (A0) e com a análise de
   vazamento de procedência (43% do pool é rótulo-por-origem). Mesmo
   descontando o confundidor de épocas, é a única alavanca que produziu >0,1.
2. **max_length 192 ou 256** — 128 perde ~0,04; 256 é neutro no mini-regime e
   reduz truncamento de 18,8% para 13,0% (vale um run full de confirmação).
3. **freeze 6** (custo/benefício). freeze 0 só se sobrar orçamento de GPU/CPU.
4. **lr 2e-5, warmup 0,1, clip 1,0, wd 0,01** — sem evidência para mudar;
   lr 3e-5 merece um teste full.
5. **mask_entities** merece um run full de confirmação (+0,09 no mini, 1 seed).
6. **Descartar**: batch 16, lr 1e-5, wd 0, clip 0,5.
7. DFR + calibração Platt continuam sendo o protocolo de score; nada no sweep
   justifica abrir mão de ECE.

## O que NÃO ficou concluído

- `informative` × `full` no orçamento real (2 épocas completas) — os controles
  respondem no regime 1 época/5k, não no full. O teste decisivo é rodar o
  trainer completo em `informative` e comparar no mesmo teste `full_iid` com o
  R0 já medido (macro-F1 0,7782 / pior-grupo 0,5749 / ECE 0,0188).
- Fase 3 do loop (melhor config com seed 43 e steps 200) não rodou: budget
  acabou às 03:16.
- Diferenças ≤0,02 são ruído de seed/amostragem.
