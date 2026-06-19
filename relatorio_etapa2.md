# Relatório Técnico Preliminar - Etapa 2 (Modelagem)
**Disciplina:** Computação de Alto Desempenho (CAD)  
**Professor:** Ricardo Augusto Pereira Franco  
**Equipe:**
- Henrique M. M. Miranda - 202405479
- Cindy Stephanie Gomes Rabelo - 202403898
- Eduardo Dias Peixoto - 202010395
- Luiany Goncalves Carvalho - 202303351
**Tema:** Cálculo Paralelo de Índices de Validação de Clusters em GPU (Dunn, Silhueta e Davies-Bouldin)

---

## 1. Introdução e Contexto

Nesta segunda etapa do trabalho prático, consolidamos a modelagem matemática e arquitetural do cálculo das três principais métricas de validação de agrupamentos (*clusters*): **Índice de Dunn**, **Coeficiente de Silhueta** e **Índice Davies-Bouldin**. 

O gargalo computacional comum dessas métricas reside na necessidade de computar distâncias euclidianas par-a-par entre todos os pontos do dataset, resultando em uma complexidade de tempo quadrática ($O(N^2)$). Enquanto na CPU sequencial essa barreira inviabiliza a validação de grandes conjuntos de dados, o paralelismo massivo das GPUs modernas surge como a solução ideal.

Apresentamos a seguir o baseline sequencial implementado em C++ e a arquitetura detalhada dos kernels paralelos CUDA projetados para serem executados no **Google Colab** (visando contornar a ausência de uma GPU NVIDIA local).

---

## 2. Fundamentação Matemática das Métricas

Seja um conjunto de dados $X = \{x_1, x_2, \dots, x_N\}$ em um espaço de dimensão $D$, particionado em $K$ clusters $\{C_1, C_2, \dots, C_K\}$.

### 2.1. Índice de Dunn
O Índice de Dunn avalia a qualidade do agrupamento buscando maximizar a separação inter-cluster e minimizar a dispersão intra-cluster:
$$Dunn = \frac{\min_{a \neq b} \delta(C_a, C_b)}{\max_{c} \Delta(C_c)}$$
Onde:
- $\delta(C_a, C_b)$ é a distância mínima entre qualquer ponto de $C_a$ e qualquer ponto de $C_b$:
  $$\delta(C_a, C_b) = \min_{x \in C_a, y \in C_b} d(x, y)$$
- $\Delta(C_c)$ é o diâmetro do cluster $C_c$, definido como a maior distância entre pontos pertencentes ao mesmo cluster:
  $$\Delta(C_c) = \max_{x, y \in C_c} d(x, y)$$

Um valor de Dunn **maior** indica agrupamentos mais densos e bem separados.

### 2.2. Coeficiente de Silhueta
Para cada ponto $i$, a silhueta $s_i$ mede a adequação do ponto ao seu cluster comparado aos vizinhos:
$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
Onde:
- $a_i$ é a distância média de $x_i$ a todos os outros pontos do mesmo cluster $C_{own}$:
  $$a_i = \frac{1}{|C_{own}| - 1} \sum_{j \in C_{own}, j \neq i} d(x_i, x_j)$$
- $b_i$ é a menor distância média de $x_i$ a pontos de qualquer outro cluster $C \neq C_{own}$:
  $$b_i = \min_{C \neq C_{own}} \left( \frac{1}{|C|} \sum_{j \in C} d(x_i, x_j) \right)$$

O score global da Silhueta é a média de $s_i$ para todos os pontos. Varia entre $-1$ e $1$.

### 2.3. Índice Davies-Bouldin
Baseia-se na similaridade entre clusters $R_{ij}$, que combina a dispersão interna $S$ com a distância entre os centroides $M_{ij}$:
$$R_{ij} = \frac{S_i + S_j}{M_{ij}}$$
Onde:
- $S_i$ é a dispersão interna do cluster $i$ (distância média dos pontos ao centroide $\mu_i$):
  $$S_i = \frac{1}{|C_i|} \sum_{x \in C_i} ||x - \mu_i||$$
- $M_{ij}$ é a distância euclidiana entre os centroides dos clusters $i$ e $j$:
  $$M_{ij} = ||\mu_i - \mu_j||$$

O índice Davies-Bouldin global é a média das piores razões de similaridade para cada cluster:
$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} R_{ij}$$

Valores **menores** indicam agrupamentos melhores.

---

## 3. Metodologia de Paralelização em GPU (CUDA)

Desenvolvemos kernels específicos para cada etapa de cálculo, focando no uso de **Memória Compartilhada (Shared Memory)** para evitar acessos repetidos à Memória Global e técnica de **Redução Paralela** para encontrar mínimos/máximos de forma eficiente.

```
+------------------------------------------------------------------------+
|                      FLUXO DE EXECUÇÃO GPU (CUDA)                      |
+------------------------------------------------------------------------+
|                                                                        |
|  [Dados Sintéticos]                                                    |
|         │                                                              |
|         ▼ (H2D)                                                        |
|  [Memória Global GPU] ──► [pairwise_distances_kernel] ─► [Matriz D]    |
|                                                            │           |
|  ┌─────────────────────────────────────────────────────────┼──────────┐|
|  │                       Método Dunn                       │          │|
|  │                                                         ▼          │|
|  │                      [dunn_reduction_kernel]                       │|
|  │            (Redução em Shared Memory por bloco/linha)              │|
|  │                                 │                                  │|
|  │                                 ▼ (D2H)                            │|
|  │                      [Redução Global na CPU]                       │|
|  └────────────────────────────────────────────────────────────────────┘|
|  ┌────────────────────────────────────────────────────────────────────┐|
|  │                     Método Silhueta                                │|
|  │                                 │                                  │|
|  │                                 ▼                                  │|
|  │                       [silhouette_kernel]                          │|
|  │  (Acumulação dinâmica em Shared Memory por Cluster por thread/bloco)│|
|  │                                 │                                  │|
|  │                                 ▼ (D2H)                            │|
|  │                        [Média Global CPU]                          │|
|  └────────────────────────────────────────────────────────────────────┘|
|  ┌────────────────────────────────────────────────────────────────────┐|
|  │                    Método Davies-Bouldin                           │|
|  │                                 │                                  │|
|  │                                 ▼                                  │|
|  │ [compute_centroids_kernel] ─► [compute_dispersion_kernel]           │|
|  │             (Soma Atômica)    │           (Soma Atômica)           │|
|  │                               ▼                                    │|
|  │                       [compute_db_kernel]                          │|
|  │                                 │                                  │|
|  │                                 ▼ (D2H)                            │|
|  │                      [Cálculo Davies-Bouldin]                      │|
|  └────────────────────────────────────────────────────────────────────┘|
+------------------------------------------------------------------------+
```

### 3.1. Kernel de Distâncias Par-a-Par
Dispara uma grade bidimensional de threads. Cada thread $(x, y)$ calcula a distância euclidiana entre o ponto $x$ e o ponto $y$ e armazena em uma matriz linearizada `D[x * N + y]`.

### 3.2. Kernel de Redução para Dunn (`dunn_reduction_kernel`)
Dispara $N$ blocos (um para cada linha da matriz de distâncias).
- Cada thread do bloco varre a linha de distâncias correspondente de forma intercalada, acumulando localmente a menor distância entre elementos de clusters diferentes (`min_inter`) e a maior distância entre elementos do mesmo cluster (`max_intra`).
- Ao final, as threads do bloco realizam uma **redução em árvore usando memória compartilhada** (`__shared__`) para achar o mínimo e o máximo do bloco.
- A thread 0 escreve o resultado do bloco em vetores globais na VRAM.
- A CPU lê esses vetores ($O(N)$ elementos) e encontra o mínimo/máximo global.

### 3.3. Kernel de Silhueta (`silhouette_kernel`)
Dispara $N$ blocos (um bloco por ponto).
- Aloca **Memória Compartilhada Dinâmica** (`extern __shared__ double shared_sums[]`) de tamanho `blockDim.x * K` para permitir o cálculo paralelo das somas de distâncias para cada um dos $K$ clusters.
- Cada thread acumula as distâncias para o ponto de interesse aos demais pontos nos índices correspondentes dos clusters.
- Um passo de redução soma os valores das threads por cluster.
- A thread 0 calcula $a_i$, $b_i$ e computa a silhueta local $s_i$.
- O vetor $s$ é retornado à CPU para extração da média global.

### 3.4. Kernels de Davies-Bouldin
- **Centróides:** Thread por ponto adiciona atonicamente (`atomicAdd`) as coordenadas aos acumuladores dos centroides do cluster mapeado. Um kernel subsequente faz a divisão pelo tamanho do cluster.
- **Dispersão:** Cada thread calcula a distância do ponto ao centroide calculado e adiciona atonicamente à dispersão do cluster. Um kernel subsequente divide pelo tamanho do cluster.
- **DB Ratios:** Cada thread calcula a pior razão para o cluster $i$ contra os outros centroides. A CPU coleta e calcula a média final.

---

## 4. Metodologia Experimental e Resultados (CPU vs GPU)

A metodologia experimental consiste em executar os códigos compilados em C++ (CPU baseline) e CUDA C++ (GPU paralelo) sobre conjuntos de dados sintéticos do tipo Blobs gerados aleatoriamente com diferentes tamanhos ($N = 250, 500, 1000, 2000, 4000, 8000$) em dimensão $D=4$ e $K=5$ clusters.

Os experimentos de CPU foram realizados no processador do Host, e os testes de GPU foram executados em uma placa **NVIDIA T4** (no ambiente do Google Colab). A corretude matemática de todos os kernels paralelos (incluindo reduções e memória compartilhada) foi validada, batendo 100% com o baseline sequencial.

### Tabela Comparativa de Tempos e Speed-up
| Tamanho ($N$) | Tempo CPU ($s$) | Tempo GPU ($s$) | Speed-up ($x$) | Corretude (Dunn Match) |
| :---: | :---: | :---: | :---: | :---: |
| 250 | 0.0010 | 0.0012 | 0.87x | SIM (100%) |
| 500 | 0.0032 | 0.0016 | 1.97x | SIM (100%) |
| 1000 | 0.0113 | 0.0026 | 4.29x | SIM (100%) |
| 2000 | 0.0456 | 0.0051 | 8.88x | SIM (100%) |
| 4000 | 0.2187 | 0.0147 | 14.89x | SIM (100%) |
| 8000 | 0.9191 | 0.0274 | **33.51x** | SIM (100%) |

### Análise de Desempenho
- **Overhead Inicial:** Para o menor dataset ($N = 250$), a diferença de desempenho é praticamente nula (Speed-up de $0.87\times$). Isso demonstra que a otimização de memória coalescida e o tamanho do bloco reduziram o impacto do overhead de cópia Host-to-Device e Device-to-Host.
- **Escalabilidade do Speed-up:** Conforme o tamanho $N$ do dataset cresce, o tempo da CPU sequencial aumenta quadraticamente ($O(N^2)$), alcançando **0.9191 segundos** para $N=8000$. Em contrapartida, a GPU mantém um tempo extremamente baixo, precisando de apenas **0.0274 segundos** para processar todas as métricas paralelas.
- **Speed-up Máximo:** Atingimos um Speed-up expressivo de **$33.51\times$** no maior tamanho ($N=8000$). Esse ganho continuará crescendo para volumes ainda maiores, ilustrando a grande eficiência do paralelismo de dados e do uso de memória compartilhada.

O gráfico de performance contendo as curvas de tempos comparados e de Speed-up foi salvo com sucesso em `curva_performance_cuda.png`.

---

## 5. Como Executar os Testes de GPU no Google Colab (Passo a Passo)

Como a sua máquina local possui uma GPU AMD Radeon, siga os passos abaixo para compilar o código CUDA, rodar os experimentos na GPU NVIDIA T4 e gerar os gráficos finais de speed-up:

1. **Abra o Google Colab:** Acesse [colab.research.google.com](https://colab.research.google.com) e crie um novo Notebook.
2. **Ative a GPU:** No menu superior do Colab, clique em **Ambiente de execução** -> **Alterar tipo de ambiente de execução** -> Selecione **T4 GPU** em *Acelerador de Hardware* e clique em Salvar.
3. **Carregue os Arquivos:** Clique no ícone de pasta (arquivos) na barra lateral esquerda do Colab e faça o upload de três arquivos da pasta do seu projeto:
   - [baseline_cpu.cpp](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/baseline_cpu.cpp)
   - [metrics_cuda.cu](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu)
   - [benchmark.py](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/benchmark.py)
4. **Execute o Benchmark:** Crie uma célula de código no Colab e execute:
   ```python
   !python benchmark.py
   ```
5. **Colete os Resultados:**
   - O Colab irá compilar automaticamente os dois códigos (com `g++` e `nvcc`), gerará os dados, rodará os benchmarks e imprimirá a tabela completa de Speed-up.
   - O gráfico consolidado `curva_performance_cuda.png` contendo os tempos comparados e a curva de speed-up ideal estará disponível na aba de arquivos do Colab. Basta clicar com o botão direito e escolher **Fazer download**.

---

## 6. Tarefas Previstas para a Finalização do Trabalho (Etapa Final)

Com a metodologia e baseline da GPU validados, as tarefas planejadas para a entrega final do trabalho prático incluem:
1. **Otimizações de Memória no Kernel CUDA:** Implementar *Tiling* de memória compartilhada para o kernel de matriz de distâncias, diminuindo os acessos à memória global.
2. **Explorar streams do CUDA:** Implementar execução assíncrona usando Streams para sobrepor a cópia de memória Host-Device com a execução de kernels.
3. **Escalar o Dataset:** Testar a aplicação escalando o dataset para $N = 50.000$ e $N = 100.000$ pontos para avaliar o limite físico do barramento PCIe e da VRAM.
4. **Redação da Monografia Final:** Estruturar o documento do trabalho prático conforme as diretrizes acadêmicas.
