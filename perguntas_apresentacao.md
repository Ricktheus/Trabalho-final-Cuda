# Banco de Perguntas — Apresentação (sem respostas)

> Versão só com as perguntas, para autoteste. As respostas estão em `roteiro_e_perguntas_apresentacao.md`.


## Slide 1 — Capa: Validação Paralela de Clusters em GPU

1. Em uma frase, qual é o trabalho?
2. O que significa "matrix-free"?
3. Por que usar GPU para esse problema?
4. Qual o ganho headline?
5. O ganho de velocidade custou exatidão?
6. Quais as três métricas e por que três?
7. Qual a disciplina e o contexto?
8. Que linguagem/plataforma?
9. Qual hardware foi usado?
10. O código está disponível?

## Slide 2 — Contextualização: O Problema

1. O que é clustering?
2. Por que validar um agrupamento?
3. O que é uma métrica *interna*?
4. Por que o custo é O(N²)?
5. O que é o "D" em O(N²·D)?
6. Por que "dobrar N quadruplica o trabalho"?
7. As três métricas têm o mesmo gargalo?
8. Por que não basta uma métrica?
9. Por que K-Means/DBSCAN não resolvem isso?
10. Por que GPU e não só otimizar a CPU?

## Slide 3 — Formulação Matemática: As Três Métricas

1. Defina o Índice de Dunn.
2. O que são `a(i)` e `b(i)` na Silhueta?
3. Qual o intervalo da Silhueta e o que significam os extremos?
4. O que mede `S_i` e `M_ij` no Davies-Bouldin?
5. Por que no Dunn "maior é melhor" e no DB "menor é melhor"?
6. Por que o Dunn é sensível a outliers?
7. A Silhueta global é o quê?
8. Qual métrica NÃO precisa de distâncias par-a-par?
9. Que distância vocês usam?
10. O resultado das métricas depende da implementação (CPU/GPU)?

## Slide 4 — Posicionamento: Artigo-base

1. Qual é o artigo-base?
2. Qual a estratégia do artigo?
3. Como vocês diferem dele?
4. Por que dizem "complementar" e não "melhor"?
5. Quando a abordagem deles seria preferível?
6. E quando a de vocês é preferível?
7. Poderiam combinar as duas ideias?
8. O artigo calcula as três métricas?
9. O que é "Sketch and Validate"?
10. Por que não reimplementaram o método do artigo para comparar?

## Slide 5 — Objetivos do Projeto

1. Qual é o objetivo geral?
2. Por que "exato" é um objetivo explícito?
3. O que significa "comparação justa"?
4. Por que reduzir memória é um objetivo?
5. O que é "breakdown de tempo por etapa"?
6. Qual hierarquia de memória vocês exploram?
7. Qual o teto de N visado?
8. Como garantem a corretude (objetivo 4)?
9. Por que medir escalabilidade?
10. Superar a CPU era suficiente ou foram além?

## Slide 6 — Decisão-chave: Matrix-free

1. O que a versão ingênua faria de errado?
2. Quanto ocuparia a matriz em N=100.000?
3. Por que 50.000 já é problema?
4. O que é o overflow de inteiro de 32 bits?
5. Para quanto cai a memória com matrix-free?
6. Qual é o "preço" da matrix-free?
7. Por que essa troca vale a pena?
8. A tabela é da memória real usada?
9. Matrix-free muda o resultado numérico?
10. O que de fato limita o N agora?

## Slide 7 — Metodologia Paralela: Índice de Dunn

1. Qual o mapeamento de paralelismo do Dunn?
2. Por que carregar `i` em memória compartilhada?
3. O que é grid-stride?
4. Por que `log₂256 = 8` passos?
5. Para que serve `__syncthreads()`?
6. O que cada bloco acumula?
7. Por que a CPU ainda faz uma redução final?
8. Por que 256 threads (e não 1024)?
9. A matriz N×N aparece em algum momento?
10. Por que o Dunn ainda é O(N²) na GPU?

## Slide 8 — Metodologia Paralela: Silhueta e Davies-Bouldin

1. Por que a Silhueta usa shared memory `blockDim × K`?
2. Como obtêm `a(i)` e `b(i)` no kernel?
3. Por que o DB usa centróides em vez de pares?
4. O que é `atomicAdd` e por que é preciso?
5. O que é o *fallback* por compare-and-swap?
6. Qual o custo do Davies-Bouldin?
7. Por que o DB é "marginal" no tempo total?
8. Que hierarquia de memória é explorada?
9. A Silhueta tem alguma limitação de K?
10. Por que acumular em double mesmo no modo float?

## Slide 9 — Baseline em CPU: Comparação Justa

1. Por que o baseline também é matrix-free?
2. Como o OpenMP paraleliza?
3. Por que comparar contra 1 thread E contra OpenMP?
4. As cláusulas de redução servem para quê?
5. Por que CPU e GPU na mesma máquina?
6. O OpenMP acelerou bastante?
7. Isso enfraquece o resultado?
8. Como mediram sequencial e paralelo com um binário só?
9. Como seria o OpenMP numa CPU melhor?
10. Por que não compararam contra uma CPU mais potente?

## Slide 10 — Validação Numérica: 3 Camadas

1. Quais são as três camadas?
2. Por que um caso analítico para o Dunn?
3. Por que make_blobs e não Iris?
4. Qual o erro vs. scikit-learn?
5. O que é a equivalência CPU≡GPU?
6. Qual a tolerância usada?
7. Por que validar antes de medir tempo?
8. Por que o erro não é exatamente zero vs. scikit-learn?
9. O caso analítico deu erro zero mesmo?
10. Como sabem que não há "bug espelhado" (mesmo erro nos dois)?

## Slide 11 — Resultado: Tempo de Execução vs N

1. Por que escala log-log?
2. O que a inclinação ≈ 2 confirma?
3. A GPU também é O(N²)?
4. Quais os tempos em N=100.000?
5. Por que as curvas de CPU-1 e OpenMP quase coincidem?
6. O que representa a distância vertical entre curvas?
7. Por que a vantagem da GPU cresce com N?
8. O tempo medido inclui transferências e leitura de dados?
9. Como garantem que os tempos são estáveis?
10. Em N pequeno a GPU compensa?

## Slide 12 — Resultado: Speed-up vs N

1. Qual o speed-up máximo?
2. Por que o speed-up cresce com N?
3. Por que as duas curvas de speed-up são tão próximas?
4. O que causa a anomalia em N=250 (OpenMP)?
5. O speed-up é "real" se o OpenMP é fraco?
6. O speed-up satura?
7. Por que não passar de 100k para mostrar speed-up maior?
8. Esse ganho é otimista ou conservador?
9. O que é "speed-up" exatamente?
10. O speed-up seria o mesmo em float?

## Slide 13 — Resultado: Breakdown de Tempo da GPU

1. Quais são as quatro etapas do breakdown?
2. Quem domina o tempo?
3. Os 99,9% são de quê exatamente?
4. Então a soma das etapas bate com o "GPU (s)" da Tabela 2?
5. Por que H2D é desprezível?
6. O que significa "compute-bound"?
7. Por que CUDA streams não ajudariam?
8. Por que o Davies-Bouldin é tão barato?
9. Como mediram o tempo de cada kernel?
10. Qual otimização faria sentido então?

## Slide 14 — Tabela de Resultados

1. Qual a linha mais importante?
2. O que significa "Corretude 100%"?
3. Como os tempos foram agregados?
4. Onde está o desvio-padrão?
5. Por que o speed-up vs OpenMP é quase igual ao vs 1-thread?
6. A versão antiga (com matriz) chegaria a 100k?
7. O ganho é constante ao longo de N?
8. Qual o erro vs. scikit-learn citado no rodapé?
9. Esses números são em double ou float?
10. Os dados de teste são quais?

## Slide 15 — Análise dos Resultados

1. Resuma a conclusão sobre escalabilidade.
2. Por que chamam o baseline de "honesto"?
3. Se o OpenMP é fraco, o resultado vale?
4. Por que o problema é compute-bound?
5. A decisão matrix-free se justificou?
6. Qual o trade-off do float?
7. Por que os ganhos são "conservadores"?
8. Qual a principal limitação reconhecida?
9. Quais os trabalhos futuros?
10. Qual a mensagem final de uma frase?

## 🎯 Perguntas gerais de fechamento (bônus)

1. Se tivessem que destacar UMA contribuição, qual seria?
2. O que vocês fariam diferente com mais tempo?
3. Por que não usaram uma biblioteca pronta (cuML/cuBLAS)?
4. O método funciona para clusters de formato irregular?
5. Como garantem reprodutibilidade?
6. Qual o papel do Python no trabalho?
7. O speed-up incluiria a geração dos dados?
8. É possível ir além de 100 mil pontos?
9. Por que três métricas e não só o Dunn do artigo?
10. Qual a relevância prática disso?
