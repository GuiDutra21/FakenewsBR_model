# Revisão crítica (adversarial) — autoresearch v6

Alvo da revisão: os 23 experimentos da noite (`results.tsv`/`VERDICT.md`) e a
minha análise da manhã (`RELATORIO_MANHA.md`). Severidades: BLOQUEANTE > ALTO >
MÉDIO > BAIXO.

## C1 — BLOQUEANTE: a unidade de orçamento estava errada

Orçamento fixo em **steps** não é comparável quando o eixo muda custo/step:

| run | min/100 steps | amostras/step | leitura |
|---|---:|---:|---|
| baseline | 9,6 | 32 | referência |
| batch16 | 6,0 | 16 | viu **metade** das amostras (0,32 vs 0,64 época) |
| ml128 | 7,6 | 32 (mas 128 tokens) | -21% de tempo; -36% de tokens pagos |
| ml256 | 10,8 | 32 | +12% de tokens pagos |
| freeze0 | 12,4 | 32 | +29% de compute por step |

Consequências:
- `batch16` ficou ruim por **déficit de amostras**, não por batch; a decisão
  prática (manter batch32) se sustenta só por **eficiência por amostra**
  (6,0 min p/ 1.600 vs 9,6 p/ 3.200 — pior por amostra de qualquer jeito).
- `ml128` "ganhou" compute e "perdeu" tokens; a diferença vs ml192 (0,4541 vs
  0,4437 full; 0,5790 vs 0,6172 informative) mistura as duas coisas.
- Comparações limpas só entre runs de **mesmo custo**: mask, lr*, warmup, wd,
  clip, droptiers, classweights, seed. O resto pede refação com orçamento em
  **wall-clock ou tokens** (o padrão Karpathy é wall-clock; em CPU dá para
  usar tokens pagos, mais determinístico).

## C2 — BLOQUEANTE: confundidor de épocas no eixo dados

- `informative` = 2.027 linhas × 100 steps × 32 = **1,58 épocas**;
  `full` = 5.000 × 100 × 32 = **0,64 época**.
- O +0,17 de `informative` compara "1,58 épocas em dado curado" vs "0,64 época
  em dado bruto". Não prova dado melhor; prova, no máximo, que em orçamento
  curto o regime curado aprende mais por unidade de tempo — pergunta diferente.
- Em dado pequeno (2k), 1,58 épocas também significa mais **memorização** num
  subset IID com a avaliação — o ranking favorece justamente o que não mede
  generalização.
- Controles 1-época (`run_controls.py`) rodando para desfazer (atrasados por
  contenção da máquina; ver §estado).

## C3 — BLOQUEANTE para a pergunta real: não medimos OOD nem o headline

- A métrica usada (`group_mean_macro_f1`) **não é a manchete do projeto**
  (`worst_group_f1`). Exemplo: `mask` subiu +0,09 no mean e ficou **igual** no
  pior grupo (0,3254 vs 0,3277) — se a régua do projeto fosse aplicada, o
  efeito seria ~zero. `classweights` subiu no mean E no pior (0,4402).
- Não medimos OOD (o `ood_wa` existe e responde a pergunta central do README).
  O sweep mede IID num val do mesmo split: otimiza o que sabemos que já é
  bom, não o que queremos provar.
- Sem Platt: ECE não é comparável entre estratégias de prior distinto
  (informative ~52%, classweights ~50%, full+DFR ~60%). Macro-F1 em threshold
  0,5 favorece prior balanceado — parte do ganho de `informative`/`classweights`
  pode ser só isso. PR-AUC seria imune.

## C4 — ALTO: estatística fraca

- 1 seed por config; ruído estimado com **um par** (seed42 0,4437 vs seed43
  0,4658 → Δ0,0221). Com ~20 comparações, vários "achados" serão espúrios.
- Efeitos +0,069 / +0,074 / +0,083 estão a ~3× o ruído — devia ter chamado de
  "candidatos", não "promissores".
- A fase 2 (descida coordenada) encadeou sobre diferenças dentro do ruído:
  freeze0 0,6244 vs freeze4 0,6220 vs fz6 0,6226 (≤0,005). O **"config
  vencedor" do `VERDICT.md` (freeze_layers=0) é indefensável**; o defensável é
  `freeze_layers=6` (empate estatístico, -30-60% de custo, já validado no R0).
- `RELATORIO_MANHA.md` aponta o confundidor, mas o veredicto dele ainda
  recomenda o vencedor com fz=0 sem marcar o empate. Minha falha de redação.

## C5 — MÉDIO: regime de dados pequenos não transfere

Subset 5k/2k e 100 steps recompensam o que **memoriza rápido** (freeze0, LR
alto, warmup curto). No orçamento real (63.753 × 2 épocas) o gradiente de
utilidade pode inverter. O ranking indica direção, não magnitude.

## C6 — BAIXO: detalhes de leitura

- `droptiers` (+0,032) e `seed43` (+0,022) caem no limite do ruído.
- O VERDICT.md auto-gerado lista "efeitos vs baseline" para runs da fase 2
  que já carregam `informative` embutido (deltas ~+0,18) — lido errado
  induz a atribuir tudo ao eixo da vez. Corrigido no RELATORIO, não no VERDICT.
- ml256 no regime informative é +0,005 (ruído) mas custa +21% de tempo; o
  argumento a favor (menos truncamento) é de projeto, não do sweep.

## O que o sweep suporta com segurança

1. **batch32 > batch16** (mesmo descontando o confundidor, batch16 é pior por
   amostra: 3,75 vs 3,0 s/amostra).
2. **lr 1e-5 é muito lento**; 2e-5 e 3e-5 são viáveis.
3. **max_length 128 não ajuda**; 192 é o chão razoável, 256 é neutro.
4. **informative é o único efeito >5× ruído** — mas ainda confundido com
   épocas; é hipótese forte, não conclusão.
5. Freeze no intervalo 0–8 muda pouco no mini-regime.

## O que fazer diferente (lista curta)

1. Orçamento em **tokens** (determinístico) ou wall-clock, não steps.
2. 3 seeds por config e decisão só com |Δ| > 3× ruído medido por config.
3. Métrica primária = **worst-group**; secundária = macro-F1 e PR-AUC.
4. Um holdout OOD pequeno (ex.: um canal retido) para qualquer veredicto de
   estratégia.
5. Platt no ECE antes de comparar estratégias de prior distinto.
6. Fase 2 só sobre eixos com efeito > ruído; e sempre registrar n_train e
   épocas efetivas (estava logado, eu não usei na primeira leitura).

## Estado dos controles (fechando C2)

`controls.tsv` (primeiros 2 de 4; atrasados por contenção com a máquina do
usuário — `full_1ep` levou 9.668 s em vez de ~1.000):

| controle | épocas | grpMean | grpWorst | ECE | leitura |
|---|---:|---:|---:|---:|---|
| baseline full @100 steps | 0,64 | 0,4437 | 0,3277 | 0,097 | ponto do sweep |
| **full_1ep** | **1,0** | **0,5134** | 0,3361 | 0,071 | +0,70/época extra |
| **informative_1ep** | **1,0** | **0,5563** | **0,3858** | 0,138 | +0,61/época extra |
| informative @100 steps | 1,58 | 0,6172 | 0,4870 | 0,039 | ponto do sweep |

Resposta a C2: com ~1,0 época nos dois regimes, `informative` segue na frente
(**0,5563 vs 0,5134**, Δ=+0,043 ≈ 2× ruído). Ou seja: **o efeito é real mas
~4× menor que o +0,17 do sweep** — a maior parte era duração de treino. Segue
a maior e única alavanca óbvia, agora com ordem de grandeza correta; a
confirmação definitiva só existe no run full (63,7k vs 25,8k, mesmo teste).
Pendentes: `informative_mask_1ep` e `informative_ml256_1ep` (rodando).

ECE de informative_1ep (0,138) é ruim no 1-época não calibrado — antes de
comparar estratégias de prior distinto, Platt é obrigatório (reforça C3).
