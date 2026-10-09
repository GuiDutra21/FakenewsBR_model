# Comparacao BASE v1 x BASE v6 -- testes full_iid e OOD WhatsApp

Gerado por `models/v6/compare_v1_vs_v6.py` em 2026-09-13T00:46:49Z (CPU, 14.6 min).

Modelo avaliado: `models/artifacts/bertimbau_finetuned` (v1, hidden 768, 12 camadas), Platt a=0,6999 b=0,0331 (`calibration.json`), max_length 192, batch 32, limiar 0,50. Inferencia em CPU, logits fp32.

Coluna "novo v6" preenchida em 2026-09-13 com o run local `models/v6/artifacts/full_iid_ml192_f6_on_seed42` (base 100M, full_iid + DFR cell clip 25, 2 epocas, freeze 6, Platt em val_calib). O run novo usa DFR; o antigo nao. Para o confronto BASE x BASE sem DFR, rodar o run R0b (`--dfr-weights off`) — coluna permanece com a ressalva "com DFR".

## 1. Contaminacao do teste v6 pelo treino v1

A v1 foi reconstruida com `models.data.load(FakenewsBR_sanitized.csv)` + `iid_split(seed=42)`: n=39.466 (71,55% fake), treino 27.626 / val 5.920 / teste 5.920.

| conjunto | n testado | rids no TREINO v1 | % contaminado | n limpo | rids no val v1 (nao treinados) |
|---|---:|---:|---:|---:|---:|
| full_iid | 13.661 | 4.147 | 30,36% | 9.514 | 872 |
| ood_wa | 6.379 | 4.464 | 69,98% | 1.915 | 958 |

`contaminado` = rid presente no TREINO v1 (o antigo viu o texto com rotulo no fine-tuning). O corte `limpo` remove apenas esses; linhas que ficaram no val/teste v1 nao foram treinadas, mas participaram de selecao de checkpoint e calibracao do antigo -- limitacao residual registrada.

## 2. Metricas globais (threshold 0,5)

| metrica | antigo v1 no teste v1 (registrado) | antigo v1 no teste v6 | antigo v1 no teste v6 limpo | novo v6 no teste v6 (com DFR) |
|---|---:|---:|---:|---:|
| n | 5.920 | 13.661 | 9.514 | 13.661 |
| acuracia | 0,8655 | 0,8051 | 0,7650 | 0,8163 |
| macro-F1 | 0,8214 | 0,6943 | 0,5974 | 0,7782 |
| F1 (fake) | 0,9102 | 0,8784 | 0,8571 | 0,8701 |
| PR-AUC | 0,9615 | 0,9353 | 0,9144 | 0,9563 |
| ROC-AUC | 0,9162 | 0,8453 | 0,7838 | 0,8902 |
| Brier | 0,0962 | 0,1385 | 0,1679 | 0,1198 |
| ECE (bins 15) | 0,0105 | 0,0625 | 0,0959 | 0,0188 |
| pior-grupo macro-F1 | 0,7276 | 0,3008 | 0,3008 | 0,5749 |
| pior-grupo (balanceados) | 0,7276 | 0,3008 | 0,3008 | 0,5749 |
| casos limitrofes (acc) | 0,9653 | 0,9546 | 0,9476 | 0,8072 |
| PT-PT macro-F1 | 0,6248 | 0,6069 | 0,5814 | 0,6992 |

Registrado de: `models/artifacts/logs/5_finetune.log (TESTE | BERTimbau fine-tuned, calibrado)`. Teste v1 n=5.920; teste v6 full_iid n=13.661 pos-U+FFFD (4.147 linhas contaminadas, 30,36%), limpo n=9.514.

## 3. Por canal -- teste full_iid (macro-F1 via `models.evaluate.per_group`, min_n=30)

| canal | n (full) | acc (full) | macro-F1 (full) | n (limpo) | acc (limpo) | macro-F1 (limpo) | novo v6 acc | novo v6 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `external_claim` | 1488 | 0,5060 | 0,3534 | 1488 | 0,5060 | 0,3534 | 0,6257 | 0,6234 |
| `social` | 387 | 0,7390 | 0,4250 | 364 | 0,7308 | 0,4222 | 0,9225 | 0,5386 |
| `whatsapp` | 957 | 0,7638 | 0,7618 | 285 | 0,6982 | 0,6955 | 0,6667 | 0,6652 |
| `press_true` | 880 | 0,1250 | - | 595 | 0,0655 | - | 0,7875 | - |
| `agency_claim` | 8477 | 0,9105 | 0,6578 | 6340 | 0,8847 | 0,6545 | 0,8445 | 0,6815 |
| `covid` | 346 | 0,8295 | 0,8283 | 101 | 0,7723 | 0,7722 | 0,8035 | 0,8008 |
| `llm` | 45 | 0,9778 | 0,9754 | - | - | - | 0,9778 | 0,9761 |
| `portal` | 1081 | 0,9898 | 0,9896 | 332 | 0,9789 | 0,9786 | 0,9722 | 0,9718 |

## 4. OOD WhatsApp -- teste ood_wa (leave-one-channel-out)

| metrica | antigo v1 no ood_wa v6 | antigo v1 no ood_wa v6 limpo | novo v6 (pendente: run R1 `ood_wa`) |
|---|---:|---:|---:|
| n | 6.379 | 1.915 | - |
| acuracia | 0,8244 | 0,7608 | - |
| macro-F1 | 0,8240 | 0,7598 | - |
| F1 (fake) | 0,8322 | 0,7753 | - |
| PR-AUC | 0,8999 | 0,8415 | - |
| ROC-AUC | 0,9137 | 0,8585 | - |
| Brier | 0,1254 | 0,1662 | - |
| ECE (bins 15) | 0,0163 | 0,0696 | - |
| pior-grupo macro-F1 | 0,8240 | 0,7598 | - |

Nao ha coluna `registrado` para o OOD: os numeros OOD da v1 em `RELATORIO_DADOS.md`/`score_finetuned.json` vem do pipeline de score (regressao logistica sobre embeddings + calibracao sob prior balanceado), nao do encoder puro avaliado aqui -- comparacao direta seria enganosa.

## 5. Nota metodologica (diferencas entre os dois runs)

1. **Dados**: v1 = 39.466 linhas (`FakenewsBR_sanitized.csv`, 71,5% fake, sem dedup por cluster); v6 = 91.080 linhas (`v6_pool.parquet`, com dedup exata/quase-dup e rotulos estratificados por tier). O teste v6 nao e amostra do teste v1; a v6 e uma expansao com novas fontes (FC_*, NEWS_*).
2. **DFR on/off**: o antigo e BASE puro (encoder + Platt). O default do trainer v6 liga DFR de celula (`--dfr-weights cell --weight-clip 25`), que reequilibra os grupos e muda o prior efetivo (~0,60) e a calibracao. Para um confronto BASE x BASE, o run v6 deve ser `--dfr-weights off`; a coluna final do MD deve registrar qual foi usado.
3. **Splits**: v1 usa IID estratificado por (grupo x rotulo) em uma base sem controle de quase-duplicata; v6 usa `full_iid` (estratificado por grupo x rotulo com dedup/clusters) e `ood_wa` (canal WhatsApp inteiramente retido). O teste v1 e ~5.920 linhas; o full_iid v6 tem 13.661 (pos-U+FFFD).
4. **Contaminacao medida**: 30,36% do teste full_iid v6 estava no treino v1; o corte `limpo` existe para isolar essa memoria. A contaminacao e estrutural: o pool v6 reaproveita as linhas da v1.
5. **Metricas**: todas de `models/evaluate.py` (`core_metrics`, `worst_group_f1`, `per_group`, `expected_calibration_error`); ECE com 15 bins; pior-grupo restrito a grupos com n>=100 e minoria>=20. Platt e limiar 0,5 identicos aos do artefato antigo.
6. **U+FFFD**: removidas defensivamente do pool (4 linhas), como faz `train_bertimbau_v6.py`; o teste full_iid fica com 13.661 linhas.

## 7. Baseline TF-IDF + LogReg (referencia de teto linear)

Executado em 2026-09-13 por `models/v6/baseline_tfidf.py` (30 s no total; TF-IDF 1-2 gramas, min_df=2, 304.291 features; Platt em `val_calib` com mascara `is_balanced_group`, mesmo protocolo do FT). Artefatos em `models/v6/compare/baseline_tfidf/`.

| metrica (teste full_iid, limiar 0,5) | maioria (chuta fake) | TF-IDF+LR ERM calibrado | TF-IDF+LR balanced calibrado | FT v6 R0 (DFR) calibrado |
|---|---:|---:|---:|---:|
| acuracia | 0,7331 | 0,8032 | 0,8032 | **0,8163** |
| macro-F1 | 0,4230 | 0,7404 | 0,7414 | **0,7782** |
| F1 (fake) | 0,8460 | 0,8681 | 0,8678 | 0,8701 |
| PR-AUC | - | 0,9357 | 0,9376 | **0,9563** |
| ROC-AUC | - | 0,8489 | 0,8517 | **0,8902** |
| Brier | - | 0,1380 | 0,1371 | **0,1198** |
| ECE (bins 15) | - | 0,0468 | 0,0461 | **0,0188** |
| pior-grupo macro-F1 | - | 0,5070 | 0,5070 | **0,5749** |
| balanceados: acc / macro-F1 | - | 0,6841 / 0,6829 | 0,6846 / 0,6836 | **0,7216 / 0,7216** |
| casos limitrofes (acc) | - | 0,8582 | 0,8601 | 0,8072 |
| PT-PT macro-F1 | - | 0,6663 | 0,6664 | **0,6992** |

Leitura: a tarefa nao e trivial (maioria = 0,7331 de acuracia), mas um modelo linear de saco-de-palavras ja alcanca 0,74 de macro-F1 e 0,85 de ROC-AUC. Isso confirma que os 0,9892 de um DistilBERT num dataset pequeno de fake news vem de regime/vazamento, nao de capacidade do encoder. O ganho do BERTimbau v6 sobre o linear esta onde importa para o projeto: pior-grupo (+6,8 p.p.), ECE (0,047 -> 0,019), ROC-AUC (+3,9 p.p.) e no regime balanceado (+3,8 p.p. de macro-F1); em acuracia bruta o ganho e de 1,3 p.p. Ressalva: o linear e melhor nos casos limitrofes (0,858 vs 0,807) -- o encoder ainda nao domina meias-verdades.

## 8. Como preencher a coluna do v6 novo (Colab)

Depois do run no Colab, copiar de `runs/<run_id>/metrics.json` o bloco `test` (e `ood` para o OOD) para a ultima coluna das tabelas:

```text
R0 default (full_iid + DFR cell, 2 epocas, freeze 6, ml192):
  runs/full_iid_ml192_f6_on_seed42/metrics.json -> test.*
BASE puro equivalente ao antigo (sem DFR):
  python models/v6/train_bertimbau_v6.py --data ... --splits ... \
    --split-col full_iid --dfr-weights off --max-length 192 \
    --batch-size 32 --epochs 2 --freeze-layers 6 --seed 42
```

