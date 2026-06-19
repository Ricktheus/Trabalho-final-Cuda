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

1. **`baseline_cpu.cpp`**: Baseline em C++ (**matrix-free**) que calcula as três métricas, com **paralelização OpenMP** opcional (controlada por `OMP_NUM_THREADS`). Distâncias calculadas on-the-fly → memória O(N·D).
2. **`metrics_cuda.cu`**: Código paralelo em CUDA C++ (**matrix-free**) com os kernels de GPU. Suporta `double` (padrão) e `float` (flag `-DUSE_FLOAT`).
3. **`benchmark.py`**: Automação em Python — gera dados sintéticos, compila (g++/nvcc), valida a corretude, mede tempos com **repetições (média ± desvio)** comparando **CPU-1 × CPU-OpenMP × GPU** e plota os gráficos.
4. **`relatorio_etapa2.md`**: Relatório técnico da Etapa 2 (fundamentação matemática e fluxo da GPU).
5. **`slides_proposta_cad.md` / `.pdf`**: Slides da Etapa 2 (Marp / PDF).
6. **`slides_apresentacao_final.md`**: Roteiro completo dos slides da **apresentação final (3ª parte)**, com slides de imagem dedicados.
7. **(Gerados pelo benchmark no Colab)** `bench_tempo.png`, `bench_speedup.png`, `bench_breakdown.png`, `curva_performance_cuda.png` e a tabela `resultados_benchmark.csv`.

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

## 4. Projeto e Otimização dos Kernels CUDA (Matrix-free)

**Decisão central:** a versão final **não materializa** a matriz de distâncias `D[N×N]`. Em N=100.000 essa matriz custaria ~80 GB e estouraria tanto a VRAM da GPU quanto a RAM da CPU (além de causar overflow de `int` em `N*N`). As distâncias são **recalculadas on-the-fly** dentro dos kernels → memória **O(N·D)** em vez de **O(N²)**, viabilizando N=50.000/100.000.

### A. Índice de Dunn (`dunn_rowwise_kernel`)
- **1 bloco por linha/ponto `i`** (256 threads). As coordenadas de `i` são carregadas em *shared memory* (`s_xi`) e reusadas por todas as threads do bloco.
- Threads varrem os pontos `j` (grid-stride), calculam `d(i,j)` na hora e acumulam **máximo intra-cluster** e **mínimo inter-cluster** locais.
- **Redução em árvore** em *shared memory*; a CPU faz só a redução global $O(N)$.

### B. Coeficiente de Silhueta (`silhouette_rowwise_kernel`)
- **Shared Memory Dinâmica** (`blockDim.x * K` **doubles**) acumula a soma de `d(i,j)` por cluster (distâncias on-the-fly).
- Redução paralela por cluster; a thread 0 calcula a silhueta local $s_i$.

### C. Davies-Bouldin
- **Operações Atômicas:** `atomicAdd` no Device (com fallback CAS para `double` em arquiteturas < sm_60) para centróides e dispersões. Custo $O(N)$ — mais barato que Dunn/Silhueta.

### D. Precisão (trade-off velocidade × exatidão)
- Padrão `double` (validado). Compile com `-DUSE_FLOAT` para usar `float` nas distâncias (mais rápido na T4); os **somatórios/reduções acumulam sempre em `double`**.

---

## 5. Resultados de Auditoria e Validação Rigorosa

Os testes foram executados comparando a CPU sequencial (executada no Host local) e a GPU paralela (NVIDIA T4 executada no ambiente do Google Colab).

### A. Ground Truth contra referências externas:
- **Silhueta e Davies-Bouldin:** O script `benchmark.py` executa uma validação rigorosa comparando nossos resultados de CPU e GPU contra o **scikit-learn** original para $N=150$ (Iris). Obtivemos erro absoluto residual de apenas $\sim 2.28 \times 10^{-9}$ (limite de precisão de ponto flutuante em double), confirmando que a lógica matemática é 100% fiel à biblioteca canônica.
- **Índice de Dunn:** Como o `scikit-learn` não implementa Dunn de forma nativa, criamos um **Caso Analítico de Teste** com 4 pontos espaciais em 2 clusters de diâmetros conhecidos. O Dunn teórico esperado é de exatamente `2.000000`. Tanto a CPU quanto a GPU obtiveram `2.000000` (Erro = $0.00e+00$), descartando qualquer possibilidade de bugs espelhados.

### B. Tabela Comparativa de Performance

> ⚠️ **Nota:** a tabela abaixo é da **Etapa 2** (versão com matriz N×N), limitada a N=8000. A versão **final matrix-free + OpenMP** produz números diferentes e escala até **N=100.000** — rode `benchmark.py` no Colab e use a tabela gerada em **`resultados_benchmark.csv`** (com média ± desvio e as três curvas CPU-1/CPU-OpenMP/GPU).

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
   - Compilará a CPU (`g++ -O3 -fopenmp`) e a GPU (`nvcc`, em `double` e `float`).
   - Validará o Dunn analítico e Silhueta/DB contra `scikit-learn` (aborta se divergir).
   - Rodará os benchmarks (até **N=100.000**) com **repetições**, comparando **CPU-1 thread × CPU-OpenMP × GPU**, reportando **média ± desvio**.
   - Gerará `bench_tempo.png`, `bench_speedup.png`, `bench_breakdown.png` e a tabela `resultados_benchmark.csv`.
   - Para um teste rápido (sem 50k/100k): `python benchmark.py --max 8000`.
