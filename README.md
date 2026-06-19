# Guia de Análise e Auditoria - Etapa 2 (Computação de Alto Desempenho)

Este diretório contém todos os arquivos consolidados da **Etapa 2 (Modelagem e Resultados Preliminares)** do trabalho prático de Computação de Alto Desempenho. Este documento (`README.md`) foi estruturado especificamente para guiar uma outra Inteligência Artificial (ou revisor humano) na análise da arquitetura, corretude matemática e desempenho dos códigos desenvolvidos.

---

## 1. Contexto do Projeto

- **Instituição:** Universidade Federal de Goiás (UFG) - Instituto de Informática
- **Disciplina:** Computação de Alto Desempenho (CAD)
- **Professor:** Ricardo Augusto Pereira Franco
- **Objetivo:** Paralelizar em GPU usando CUDA três importantes métricas de validação de agrupamentos (Clustering):
  1. **Índice de Dunn** (Foco principal da proposta original)
  2. **Coeficiente de Silhueta**
  3. **Índice Davies-Bouldin**
- **Artigo de Referência (Baseline):** 
  *Parallel and scalable Dunn Index for the validation of big data clusters* (Periódico *Parallel Computing*, Elsevier).

---

## 2. Conteúdo do Diretório de Entrega

Este diretório contém os seguintes 8 arquivos essenciais:

1. **`baseline_cpu.cpp`**: Algoritmo sequencial em C++ que calcula as três métricas. Serve como baseline de comparação de desempenho.
2. **`metrics_cuda.cu`**: Código paralelo em CUDA C++ contendo os kernels e otimizações de GPU.
3. **`benchmark.py`**: Script de automação em Python para geração de dados sintéticos, compilação cruzada, execução de testes, validação matemática e plotagem de gráficos.
4. **`relatorio_etapa2.md`**: Relatório técnico em Markdown detalhando a fundamentação matemática, o fluxo da GPU e a análise de desempenho.
5. **`slides_proposta_cad.md`**: Slides da apresentação da Etapa 2 escritos no formato Marp.
6. **`slides_proposta_cad.pdf`**: Slides da apresentação exportados em PDF prontos para exibição.
7. **`curva_performance_cuda.png`**: Gráfico gerado contendo os tempos de execução (CPU vs GPU) e o Speed-up medidos no Google Colab.
8. **`curva_complexidade_cpu.png`**: Gráfico local contendo a curva de crescimento quadrático $O(N^2)$ da CPU.

---

## 3. Origem e Estrutura dos Dados

Para garantir um benchmark real e reprodutível, os dados de teste são gerados de forma sintética pelo script `benchmark.py` utilizando a biblioteca `scikit-learn` (`make_blobs`).
- **Dimensão ($D$):** 4 características (features) por ponto.
- **Clusters ($K$):** 5 centros gerados aleatoriamente.
- **Tamanhos testados ($N$):** $250, 500, 1000, 2000, 4000$ e $8000$ pontos.
- **Formato do Arquivo de Entrada (`.csv`):**
  - Primeira linha: `N D K` (Inteiros indicando total de pontos, dimensões e clusters).
  - Linhas seguintes: $N$ linhas contendo os valores de coordenadas flutuantes seguidos do rótulo da classe: `x_1 x_2 ... x_D label`.

---

## 4. Projeto e Otimização dos Kernels CUDA

A paralelização foi estruturada para mitigar o gargalo de memória global da GPU através de coalescência de acessos e reduções locais:

### A. Matriz de Distâncias (`pairwise_distances_kernel` - Coalescido)
- **Coalescência de Escrita:** Swappamos a indexação clássica. Definindo `j = blockIdx.x * blockDim.x + threadIdx.x` como o índice de coluna (que varia rápido entre threads consecutivas de uma warp) e `i = blockIdx.y * blockDim.y + threadIdx.y` como o índice de linha. Assim, escritas em `D[i * N + j]` são feitas de forma perfeitamente sequencial e coalescida na memória global da GPU.

### B. Índice de Dunn (`dunn_reduction_kernel`)
- **Shared Memory:** Cada bloco de 256 threads manipula uma linha da matriz.
- **Redução:** As threads acumulam localmente em memória compartilhada os valores de maior distância intra-cluster e menor inter-cluster daquela linha e realizam redução em árvore. A CPU faz a redução final de complexidade $O(N)$.

### C. Coeficiente de Silhueta (`silhouette_kernel`)
- **Shared Memory Dinâmica:** Aloca dinamicamente `blockDim.x * K` doubles para acumular as somas de distância de cada ponto aos $K$ clusters.
- Realiza redução paralela para condensar os valores das threads e calcular a silhueta local $s_i$ no dispositivo.

### D. Davies-Bouldin
- **Operações Atômicas:** Usa `atomicAdd` no Device com fallback de segurança para tipos `double` em arquiteturas mais antigas que Kepler/Maxwell (usando CAS - Compare-And-Swap).
- O cálculo da razão $R_{ij}$ é resolvido em paralelo com um thread por cluster.

---

## 5. Resultados de Auditoria e Validação Rigorosa

Os testes foram executados comparando a CPU sequencial (executada no Host local) e a GPU paralela (NVIDIA T4 executada no ambiente do Google Colab).

### A. Ground Truth contra referências externas:
- **Silhueta e Davies-Bouldin:** O script `benchmark.py` executa uma validação rigorosa comparando nossos resultados de CPU e GPU contra o **scikit-learn** original para $N=150$ (Iris). Obtivemos erro absoluto residual de apenas $\sim 2.28 \times 10^{-9}$ (limite de precisão de ponto flutuante em double), confirmando que a lógica matemática é 100% fiel à biblioteca canônica.
- **Índice de Dunn:** Como o `scikit-learn` não implementa Dunn de forma nativa, criamos um **Caso Analítico de Teste** com 4 pontos espaciais em 2 clusters de diâmetros conhecidos. O Dunn teórico esperado é de exatamente `2.000000`. Tanto a CPU quanto a GPU obtiveram `2.000000` (Erro = $0.00e+00$), descartando qualquer possibilidade de bugs espelhados.

### B. Tabela Comparativa de Performance
| Tamanho ($N$) | Tempo CPU ($s$) | Tempo GPU ($s$) | Speed-up ($x$) | Corretude (Dunn/Silh/DB Match) |
| :---: | :---: | :---: | :---: | :---: |
| 250 | 0.0010 s | 0.0012 s | 0.87x | SIM (100% Match contra Sklearn/Teórico) |
| 500 | 0.0032 s | 0.0016 s | 1.97x | SIM (100% Match contra Sklearn/Teórico) |
| 1000 | 0.0113 s | 0.0026 s | 4.29x | SIM (100% Match contra Sklearn/Teórico) |
| 2000 | 0.0456 s | 0.0051 s | 8.88x | SIM (100% Match contra Sklearn/Teórico) |
| 4000 | 0.2187 s | 0.0147 s | 14.89x | SIM (100% Match contra Sklearn/Teórico) |
| 8000 | 0.9191 s | 0.0274 s | **33.51x** | SIM (100% Match contra Sklearn/Teórico) |

- **Speed-up Máximo:** Atingiu **33.51x** no maior dataset ($N=8000$).

---

## 6. Como Auditar a Execução
Qualquer IA ou revisor pode auditar os resultados executando o script `benchmark.py` em um ambiente que possua `g++` e `nvcc` configurados no PATH (como o Google Colab).
1. Faça o upload deste diretório para o ambiente.
2. Execute o comando: `python benchmark.py`.
3. O script:
   - Compilará os códigos locais.
   - Executará a validação do Dunn analítico e das demais métricas contra `scikit-learn` (interrompe a execução caso haja divergência numérica).
   - Rodará os benchmarks para os tamanhos especificados e gerará a tabela e o gráfico de Speed-up final.
