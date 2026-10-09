# program.md — autoresearch FakenewsBR (v6, CPU)

Este diretório implementa a personalidade de
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) no nosso
problema: **budget fixo por experimento, UMA métrica de manchete,
mantém/descarta, log de tudo, roda a noite inteira**.

## Regras do jogo

1. Cada experimento roda **100 steps de otimização** (budget fixo) no mesmo
   subset de treino congelado (5.000 linhas, estratificado por grupo×rótulo) e
   é avaliado no mesmo subset de validação congelado (3.000 linhas).
2. A métrica é **`group_mean_macro_f1`**: média do macro-F1 nos grupos
   confiáveis (n≥40 e minoria≥10) do subset de avaliação. Desempate:
   `group_worst_macro_f1`, depois ECE. Números globais (acc/macro-F1/ECE) só
   dão contexto.
3. Um eixo por vez na fase 1; na fase 2 os vencedores são combinados
   (descida coordenada); no fim, repetição do melhor com outra seed e budget 2x.
4. Nada é apagado: cada run vira linha no `results.tsv`, resumo no
   `progress.md` e o ranking no `VERDICT.md` (reescrito a cada run).
5. O resultado é um **indício de estratégia**, não um modelo treinado. O modelo
   final ainda precisa de run completo no trainer + validação no teste.
6. Nada de arquitetura: o espaço de busca é o que o trainer já aceita (dados,
   pesos, comprimento, freeze, LR, schedule, máscaras) — o mesmo que você usaria
   no run final.

## Arquivos

```
experiment.py   roda UM experimento (config JSON -> resultado)
run_loop.py     fila autonoma + fases + log/veredicto
results.tsv     uma linha por experimento (fonte da verdade)
progress.md     painel ao vivo (atualizado a cada run)
VERDICT.md      ranking + config vencedor (reescrito a cada run)
program.md      este arquivo (edite para mudar a organizacao da pesquisa)
```

## Como retomar

```powershell
$env:OMP_NUM_THREADS='8'; $env:MKL_NUM_THREADS='8'
python models/v6/autoresearch/run_loop.py --max-minutes 255 --max-runs 40
```

Um único experimento:

```powershell
python models/v6/autoresearch/experiment.py --config '{"max_length":256,"freeze_layers":4}'
```

## Caveat honesto

100 steps e 5k amostras detectam efeitos grandes; diferenças < ~0,015 no
`group_mean_macro_f1` são ruído de amostragem/seed. O veredicto aponta a
direção, o run completo confirma.
