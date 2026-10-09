# VERDICT — melhor estrategia (autoresearch v6)

- gerado: 2026-09-14T06:16:23Z | status: concluido
- budget usado: 252.5 min | runs ok: 23
- metrica de manchete: `group_mean_macro_f1` (media de macro-F1 nos grupos confiaveis do subset de avaliacao congelado, 3.000 linhas)

## ranking

| # | run | group_mean_macro_f1 | worst | acc | macroF1 | ECE | eixos |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 021_freeze_layers=0 freeze_layers=0 | 0.6244 | 0.5042 | 0.5397 | 0.5321 | 0.0318 | ml=256 fz=0 lr=2e-05 dfr=cell mask=0 filt=informative clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 2 | 020_max_length=256 max_length=256 | 0.6226 | 0.4998 | 0.5500 | 0.5394 | 0.0212 | ml=256 fz=6 lr=2e-05 dfr=cell mask=0 filt=informative clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 3 | 022_freeze_layers=4 freeze_layers=4 | 0.6220 | 0.5042 | 0.5507 | 0.5389 | 0.0346 | ml=256 fz=4 lr=2e-05 dfr=cell mask=0 filt=informative clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 4 | 015_informative informative | 0.6172 | 0.4870 | 0.5290 | 0.5174 | 0.0389 | ml=192 fz=6 lr=2e-05 dfr=cell mask=0 filt=informative clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 5 | 023_freeze_layers=8 freeze_layers=8 | 0.6159 | 0.4876 | 0.5417 | 0.5318 | 0.0118 | ml=256 fz=8 lr=2e-05 dfr=cell mask=0 filt=informative clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 6 | 019_max_length=128 max_length=128 | 0.5790 | 0.4354 | 0.5047 | 0.4984 | 0.0644 | ml=128 fz=6 lr=2e-05 dfr=cell mask=0 filt=informative clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 7 | 008_mask mask | 0.5360 | 0.3254 | 0.7703 | 0.6340 | 0.1052 | ml=192 fz=6 lr=2e-05 dfr=cell mask=1 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 8 | 016_classweights classweights | 0.5263 | 0.4402 | 0.7290 | 0.6702 | 0.0589 | ml=192 fz=6 lr=2e-05 dfr=off mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 9 | 012_warmup0 warmup0 | 0.5180 | 0.3492 | 0.7360 | 0.6442 | 0.0889 | ml=192 fz=6 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.0 bt=32 seed=42 |
| 10 | 010_lr3e-5 lr3e-5 | 0.5126 | 0.3277 | 0.7473 | 0.6354 | 0.0546 | ml=192 fz=6 lr=3e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 11 | 004_ml256 ml256 | 0.5089 | 0.3385 | 0.7573 | 0.6115 | 0.1219 | ml=256 fz=6 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 12 | 014_droptiers droptiers | 0.4759 | 0.3492 | 0.7517 | 0.5690 | 0.0221 | ml=192 fz=6 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 13 | 018_seed43 seed43 | 0.4658 | 0.3000 | 0.7323 | 0.6077 | 0.0928 | ml=192 fz=6 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=43 |
| 14 | 006_freeze4 freeze4 | 0.4638 | 0.3277 | 0.7473 | 0.5631 | 0.0908 | ml=192 fz=4 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 15 | 002_dfr_off dfr_off | 0.4622 | 0.3277 | 0.7587 | 0.5570 | 0.0262 | ml=192 fz=6 lr=2e-05 dfr=off mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 16 | 005_freeze0 freeze0 | 0.4607 | 0.3277 | 0.7430 | 0.5656 | 0.0983 | ml=192 fz=0 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 17 | 003_ml128 ml128 | 0.4541 | 0.3196 | 0.7273 | 0.5929 | 0.0814 | ml=128 fz=6 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 18 | 007_freeze8 freeze8 | 0.4501 | 0.3277 | 0.7453 | 0.5495 | 0.1203 | ml=192 fz=8 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 19 | 011_clip05 clip05 | 0.4488 | 0.3277 | 0.7453 | 0.5488 | 0.1026 | ml=192 fz=6 lr=2e-05 dfr=cell mask=0 filt=full clip=0.5 wd=0.01 wu=0.1 bt=32 seed=42 |
| 20 | 001_baseline baseline | 0.4437 | 0.3277 | 0.7473 | 0.5415 | 0.0967 | ml=192 fz=6 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 21 | 013_wd0 wd0 | 0.4405 | 0.3277 | 0.7460 | 0.5368 | 0.0950 | ml=192 fz=6 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.0 wu=0.1 bt=32 seed=42 |
| 22 | 009_lr1e-5 lr1e-5 | 0.3817 | 0.2966 | 0.7410 | 0.4551 | 0.1294 | ml=192 fz=6 lr=1e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=32 seed=42 |
| 23 | 017_batch16 batch16 | 0.3719 | 0.3025 | 0.7400 | 0.4445 | 0.0872 | ml=192 fz=6 lr=2e-05 dfr=cell mask=0 filt=full clip=1.0 wd=0.01 wu=0.1 bt=16 seed=42 |

## efeito de cada eixo (vs baseline)

- dfr_off: +0.0185 (0.4622)
- ml128: +0.0104 (0.4541)
- ml256: +0.0652 (0.5089)
- freeze0: +0.0170 (0.4607)
- freeze4: +0.0202 (0.4638)
- freeze8: +0.0064 (0.4501)
- mask: +0.0923 (0.5360)
- lr1e-5: -0.0619 (0.3817)
- lr3e-5: +0.0689 (0.5126)
- clip05: +0.0052 (0.4488)
- warmup0: +0.0744 (0.5180)
- wd0: -0.0032 (0.4405)
- droptiers: +0.0322 (0.4759)
- informative: +0.1735 (0.6172)
- classweights: +0.0826 (0.5263)
- batch16: -0.0718 (0.3719)
- seed43: +0.0221 (0.4658)
- max_length=128: +0.1353 (0.5790)
- max_length=256: +0.1789 (0.6226)
- freeze_layers=0: +0.1807 (0.6244)
- freeze_layers=4: +0.1783 (0.6220)
- freeze_layers=8: +0.1723 (0.6159)

baseline = 0.4437 em group_mean_macro_f1

## config vencedor (indicio, nao modelo final)

```json
{
  "max_length": 256,
  "batch_size": 32,
  "lr": 2e-05,
  "warmup_frac": 0.1,
  "weight_decay": 0.01,
  "clip": 1.0,
  "freeze_layers": 0,
  "dfr_weights": "cell",
  "weight_clip": 25.0,
  "class_weights": false,
  "mask_entities": false,
  "drop_tiers": "",
  "data_filter": "informative",
  "steps": 100,
  "seed": 42,
  "train_n": 5000,
  "eval_n": 3000,
  "eval_batch_size": 64
}
```

## proximo passo recomendado

Rodar o trainer completo (`models/v6/train_bertimbau_v6.py`) com o config vencedor nos 63.753 exemplos de treino e validar no teste `full_iid` (o ranking aqui e um indicio de 100 steps / 5k amostras, sujeito a ruido; ver ressalvas no relatorio da manha).
