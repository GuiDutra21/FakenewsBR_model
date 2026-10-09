# autoresearch — progresso ao vivo

- atualizado: 2026-09-14T06:16:23Z
- budget: 255 min | decorrido: 252.5 min
- experimentos: 23 | status: concluido

## melhor ate agora

- run: `021_freeze_layers=0` (freeze_layers=0)
- group_mean_macro_f1: **0.6244** | worst: 0.5042 | acc: 0.5397 | macroF1: 0.5321 | ECE: 0.0318
- eixos: ml=256 freeze=0 lr=2e-05 dfr=cell mask=False filtro=informative clip=1.0 wd=0.01 warmup=0.1

## runs

| run | fase | label | group_mean_macro_f1 | worst | acc | macroF1 | ECE | min | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 001_baseline | 1 | baseline | 0.4437 | 0.3277 | 0.7473 | 0.5415 | 0.0967 | 9.6 | ok |
| 002_dfr_off | 1 | dfr_off | 0.4622 | 0.3277 | 0.7587 | 0.5570 | 0.0262 | 9.4 | ok |
| 003_ml128 | 1 | ml128 | 0.4541 | 0.3196 | 0.7273 | 0.5929 | 0.0814 | 7.6 | ok |
| 004_ml256 | 1 | ml256 | 0.5089 | 0.3385 | 0.7573 | 0.6115 | 0.1219 | 10.8 | ok |
| 005_freeze0 | 1 | freeze0 | 0.4607 | 0.3277 | 0.7430 | 0.5656 | 0.0983 | 12.4 | ok |
| 006_freeze4 | 1 | freeze4 | 0.4638 | 0.3277 | 0.7473 | 0.5631 | 0.0908 | 10.1 | ok |
| 007_freeze8 | 1 | freeze8 | 0.4501 | 0.3277 | 0.7453 | 0.5495 | 0.1203 | 8.3 | ok |
| 008_mask | 1 | mask | 0.5360 | 0.3254 | 0.7703 | 0.6340 | 0.1052 | 9.3 | ok |
| 009_lr1e-5 | 1 | lr1e-5 | 0.3817 | 0.2966 | 0.7410 | 0.4551 | 0.1294 | 9.0 | ok |
| 010_lr3e-5 | 1 | lr3e-5 | 0.5126 | 0.3277 | 0.7473 | 0.6354 | 0.0546 | 9.0 | ok |
| 011_clip05 | 1 | clip05 | 0.4488 | 0.3277 | 0.7453 | 0.5488 | 0.1026 | 9.0 | ok |
| 012_warmup0 | 1 | warmup0 | 0.5180 | 0.3492 | 0.7360 | 0.6442 | 0.0889 | 9.1 | ok |
| 013_wd0 | 1 | wd0 | 0.4405 | 0.3277 | 0.7460 | 0.5368 | 0.0950 | 9.1 | ok |
| 014_droptiers | 1 | droptiers | 0.4759 | 0.3492 | 0.7517 | 0.5690 | 0.0221 | 9.3 | ok |
| 015_informative | 1 | informative | 0.6172 | 0.4870 | 0.5290 | 0.5174 | 0.0389 | 13.6 | ok |
| 016_classweights | 1 | classweights | 0.5263 | 0.4402 | 0.7290 | 0.6702 | 0.0589 | 9.1 | ok |
| 017_batch16 | 1 | batch16 | 0.3719 | 0.3025 | 0.7400 | 0.4445 | 0.0872 | 6.0 | ok |
| 018_seed43 | 1 | seed43 | 0.4658 | 0.3000 | 0.7323 | 0.6077 | 0.0928 | 9.7 | ok |
| 019_max_length=128 | 2 | max_length=128 | 0.5790 | 0.4354 | 0.5047 | 0.4984 | 0.0644 | 10.7 | ok |
| 020_max_length=256 | 2 | max_length=256 | 0.6226 | 0.4998 | 0.5500 | 0.5394 | 0.0212 | 16.5 | ok |
| 021_freeze_layers=0 | 2 | freeze_layers=0 | 0.6244 | 0.5042 | 0.5397 | 0.5321 | 0.0318 | 21.5 | ok |
| 022_freeze_layers=4 | 2 | freeze_layers=4 | 0.6220 | 0.5042 | 0.5507 | 0.5389 | 0.0346 | 18.1 | ok |
| 023_freeze_layers=8 | 2 | freeze_layers=8 | 0.6159 | 0.4876 | 0.5417 | 0.5318 | 0.0118 | 14.9 | ok |
