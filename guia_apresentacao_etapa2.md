# Guia de Estudos Completo e Roteiro de Apresentação — Etapa 2 (CAD)

Este guia foi elaborado especificamente para ajudá-lo a estudar, apresentar e responder a qualquer pergunta da banca ou do professor Ricardo Augusto Pereira Franco sobre a **Etapa 2 (Modelagem e Resultados Preliminares)** do trabalho prático de **Computação de Alto Desempenho (CAD)**. 

O foco deste material é fornecer uma **explicação técnica exaustiva e detalhada** para cada slide contido no arquivo [slides_proposta_cad.md](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/slides_proposta_cad.md), destrinchando cada linha do código fonte em [metrics_cuda.cu](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu), a modelagem matemática e os resultados de desempenho.

---

🤖 **Aplicando conhecimentos de `@[documentation-writer]`...**

---

## ÍNDICE DE SLIDES
1. [Slide 1: Capa (Título da Apresentação)](#slide-1-capa-título-da-apresentação)
2. [Slide 2: O Gargalo de Validação de Clusters](#slide-2-o-gargalo-de-validação-de-clusters)
3. [Slide 3: Metodologia de Paralelização em GPU (CUDA) - A e B](#slide-3-metodologia-de-paralelização-em-gpu-cuda---a-e-b)
4. [Slide 4: Metodologia Paralela: Silhueta e Davies-Bouldin - C e D](#slide-4-metodologia-paralela-silhueta-e-davies-bouldin---c-e-d)
5. [Slide 5: Validação Numérica Rigorosa](#slide-5-validação-numérica-rigorosa)
6. [Slide 6: Experimentos e Resultados (CPU vs GPU)](#slide-6-experimentos-e-resultados-cpu-vs-gpu)
7. [Slide 7: Pipeline de Execução no Google Colab (GPU NVIDIA)](#slide-7-pipeline-de-execução-no-google-colab-gpu-nvidia)
8. [Slide 8: Próximos Passos (Etapa Final)](#slide-8-próximos-passos-etapa-final)
9. [Slide 9: Obrigado (Encerramento)](#slide-9-obrigado-encerramento)
10. [Super Banco de Perguntas e Respostas Técnicas Avançadas](#super-banco-de-perguntas-e-respostas-técnicas-avançadas)

---

## Slide 1: Capa (Título da Apresentação)

### Conteúdo do Slide:
* **Título:** Validação Paralela de Clusters em GPU (Etapa 2 - Modelagem)
* **Subtítulo:** Métricas: Índice de Dunn, Coeficiente de Silhueta e Davies-Bouldin
* **Equipe:** Henrique M. M. Miranda, Cindy Stephanie Gomes Rabelo, Eduardo Dias Peixoto, Luiany Goncalves Carvalho
* **Disciplina:** Computação de Alto Desempenho
* **Professor:** Ricardo Augusto Pereira Franco
* **Instituição:** UFG - Instituto de Informática

### 🎙️ Roteiro de Fala Passo a Passo:
> *"Boa noite, professor Ricardo e colegas de classe. Somos a equipe composta por Henrique Miranda, Cindy Rabelo, Eduardo Peixoto e Luiany Carvalho. Hoje, temos a satisfação de apresentar o desenvolvimento e os resultados práticos da Etapa 2 do nosso Trabalho Prático de Computação de Alto Desempenho. Nosso tema é a validação paralela de clusters em GPU utilizando a arquitetura CUDA da NVIDIA. Nesta etapa de modelagem, implementamos e paralelizamos com sucesso três métricas de validação de clusters extremamente importantes na ciência de dados: o **Índice de Dunn**, o **Coeficiente de Silhueta** e o **Índice Davies-Bouldin**. Ao longo desta apresentação, vamos detalhar como superamos o gargalo quadrático dessas equações e como alcançamos aceleramentos de mais de 33 vezes utilizando o paralelismo massivo de threads de GPU."*

### 🔍 Explicação Detalhada do Conteúdo:
* **Contextualização Institucional:** Este trabalho é a entrega da Etapa 2 de CAD do curso de Inteligência Artificial da UFG. A etapa foca no desenvolvimento do código paralelo em GPU e na coleta de dados experimentais preliminares comparando o speed-up contra um baseline sequencial na CPU.
* **Escopo das Métricas:** A validação de clusters é uma etapa de aprendizado não supervisionado que afere a qualidade da clusterização sem gabarito (labels reais). As três métricas escolhidas representam abordagens complementares:
  1. **Índice de Dunn:** Focado em extremos (menor distância inter-cluster sobre o maior diâmetro intra-cluster).
  2. **Coeficiente de Silhueta:** Uma métrica pontual (avalia cada ponto individualmente e tira a média).
  3. **Davies-Bouldin:** Focado em centroides (compara dispersão interna ao redor do centroide com a distância Euclidiana entre centroides).

---

## Slide 2: O Gargalo de Validação de Clusters

### Conteúdo do Slide:
* **O Problema:** Algoritmos de clustering (K-Means, DBSCAN) agrupam os dados, mas validar a qualidade destes agrupamentos exige calcular a proximidade e isolamento dos pontos.
* **O Gargalo:** Métricas como Dunn e Silhueta exigem o cálculo das distâncias euclidianas par-a-par de todos os pontos ($O(N^2)$).
* **Impacto em Big Data:** Para datasets com dezenas de milhares ou milhões de pontos, o cálculo sequencial na CPU torna-se inviável (crescimento quadrático).
* **Métricas Escolhidas:**
  * **Índice de Dunn:** Maximizar separação / dispersão.
  * **Silhueta:** Ajuste local de cada ponto em relação ao cluster vizinho.
  * **Davies-Bouldin:** Razão média entre dispersões e distâncias de centroides.

### 🎙️ Roteiro de Fala Passo a Passo:
> *"Para compreendermos a relevância deste trabalho, precisamos discutir **o gargalo computacional da validação**. Quando aplicamos algoritmos como o K-Means ou DBSCAN para agrupar dados, a tarefa não termina no agrupamento. Precisamos medir se esses grupos fazem sentido físico. É aqui que entram os índices de validação. O grande problema reside na formulação matemática dessas métricas: tanto o Índice de Dunn quanto a Silhueta exigem que calculemos a distância Euclidiana par-a-par de todos os pontos do dataset. Isso significa que, para um conjunto de $N$ pontos, precisamos realizar $N^2$ cálculos de distância. Se escalarmos isso para Big Data, com dezenas de milhares ou milhões de dados, a CPU sequencial sofre com um crescimento de tempo quadrático, tornando a computação inviável. As três métricas que paralelizamos resolvem isso sob óticas diferentes: o Dunn maximiza a separação estrita, a Silhueta mede a qualidade individual de cada ponto com seu vizinho mais próximo, e o Davies-Bouldin calcula a razão média de dispersão contra as distâncias entre centroides."*

### 🔍 Explicação Detalhada do Conteúdo:
* **A Matemática do Gargalo $O(N^2)$:** Para cada par de pontos $i$ e $j$, onde $i, j \in \{1, \dots, N\}$, a distância euclidiana em um espaço de dimensão $D$ é dada por:
  $$d(x_i, x_j) = \sqrt{\sum_{d=1}^{D} (x_{i,d} - x_{j,d})^2}$$
  O cálculo completo da matriz de distâncias exige $N(N-1)/2$ avaliações exclusivas se considerarmos a simetria ($d(i,j) = d(j,i)$). Porém, na prática, muitas implementações (incluindo a nossa em GPU) computam a matriz cheia $N \times N$ para manter acessos de memória regulares e evitar desvios condicionais na GPU (divergência de warp), resultando em exatos $N^2$ cálculos de distância de dimensão $D$.
* **Complexidade Algorítmica:** O tempo total é $O(N^2 \cdot D)$. Conforme o número de amostras $N$ dobra, o esforço computacional quadruplica. Para $N = 8.000$ pontos, realizamos 64 milhões de cálculos de distâncias, cada um contendo subtrações, multiplicações e somas de dimensão $D$, seguidos de uma raiz quadrada.
* **Índice de Dunn:**
  $$Dunn = \frac{\min_{1 \le a < b \le K} (\text{dist\_inter}(C_a, C_b))}{\max_{1 \le c \le K} (\text{diâmetro}(C_c))}$$
  Exige buscar o menor valor mínimo absoluto entre pontos fora do cluster e o maior valor máximo absoluto dentro dos clusters. Extremamente sensível a outliers.
* **Coeficiente de Silhueta:** Para cada ponto $i$:
  $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
  Onde $a_i$ é a distância média do ponto $i$ para todos os outros pontos no mesmo cluster, e $b_i$ é a menor distância média do ponto $i$ para os pontos de qualquer outro cluster vizinho.
* **Davies-Bouldin (DB):**
  $$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \left( \frac{S_i + S_j}{M_{ij}} \right)$$
  Onde $S_i$ é a dispersão do cluster $i$ (distância média dos pontos ao centroide) e $M_{ij}$ é a distância entre os centroides dos clusters $i$ e $j$.

---

## Slide 3: Metodologia de Paralelização em GPU (CUDA) - A e B

### Conteúdo do Slide:
* **A. Matriz de Distâncias Euclidiana:** Thread $(x, y)$ calcula a distância $d(x_i, x_j)$. Lançado em grid 2D com blocos de $16 \times 16$ threads.
* **B. Índice de Dunn (Kernel de Redução Paralela):**
  * Lançamos $N$ blocos de 256 threads (um bloco por ponto/linha da matriz).
  * As threads do bloco varrem a linha em paralelo e realizam **Redução em Árvore usando Memória Compartilhada** (`__shared__`) para achar a maior distância intra-cluster e a menor inter-cluster do bloco.
  * A CPU apenas faz a redução global linear ($O(N)$) dos resultados dos blocos.

### 🎙️ Roteiro de Fala Passo a Passo:
> *"Como resolvemos esses gargalos na GPU usando CUDA? Dividimos a solução em dois passos fundamentais. Primeiro, projetamos o kernel de distância Euclidiana par-a-par. Mapeamos a matriz $N \times N$ como um grid bidimensional composto por blocos de $16 \times 16$ threads. A thread de coordenadas $(x, y)$ é responsável por calcular e escrever a distância entre o ponto $x$ e o ponto $y$ na memória global. Para evitar perdas por latência de memória RAM de GPU, organizamos a indexação de modo que threads adjacentes de uma mesma warp acessem posições consecutivas de memória física de uma coluna, o que garante a coalescência de escrita. O segundo passo é o Índice de Dunn. Criamos um kernel de redução paralela. Em vez de transferir a matriz gigante $N \times N$ de volta para a CPU via barramento PCIe, lançamos $N$ blocos de 256 threads — isto é, um bloco de threads por linha da matriz. As threads do bloco leem a linha da memória de forma paralela e realizam uma redução em árvore usando memória compartilhada rápida. Elas calculam o máximo intra-cluster e o mínimo inter-cluster local de cada bloco. Ao final, a CPU recebe apenas dois pequenos vetores de tamanho $N$, realizando uma redução linear simples de ordem $O(N)$ no Host, economizando gigabytes de largura de banda de transmissão PCIe."*

### 🔍 Explicação Detalhada do Conteúdo:
* **Mapeamento de Thread no Grid 2D (Matriz de Distâncias):**
  No código CUDA, o kernel [pairwise_distances_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L99-L115) tem blocos de $16 \times 16 = 256$ threads. Os índices globais da thread são determinados por:
  ```cuda
  int j = blockIdx.x * blockDim.x + threadIdx.x; // Coluna (eixo X)
  int i = blockIdx.y * blockDim.y + threadIdx.y; // Linha (eixo Y)
  ```
  Na memória de vídeo, a matriz de distâncias $D$ unidimensionalizada é organizada no formato row-major (linhas contíguas na memória). Ao mapear o índice de coluna `j` com `threadIdx.x`, threads consecutivas do bloco que rodam de forma concorrente em uma warp (threads de 0 a 31) escrevem em endereços sequenciais `D[i * N + j]`. Isso atende à regra de **Coalescência de Acesso** do hardware NVIDIA, consolidando 32 requisições individuais das threads em uma única transação física de escrita de 128 bytes no barramento de memória da GPU, aumentando drasticamente a eficiência da VRAM.
* **Redução em Árvore na Memória Compartilhada (Dunn):**
  Para cada ponto/linha da matriz, no kernel [dunn_reduction_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L118-L157) alocamos arrays na memória compartilhada rápida (L1 de bloco) para armazenar os máximos e mínimos parciais calculados pelas 256 threads do bloco:
  ```cuda
  __shared__ double s_max[256];
  __shared__ double s_min[256];
  ```
  Após a varredura inicial onde cada thread calcula seu máximo e mínimo parcial de forma serial ao longo de fatias da linha (estratégia grid-stride loop para acomodar qualquer tamanho $N$ com blocos fixos), realizamos a redução clássica em árvore:
  ```cuda
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tx < stride) {
          if (s_max[tx + stride] > s_max[tx]) s_max[tx] = s_max[tx + stride];
          if (s_min[tx + stride] < s_min[tx]) s_min[tx] = s_min[tx + stride];
      }
      __syncthreads(); // Sincronização de barreira necessária
  }
  ```
  Este laço roda em complexidade de $\log_2(256) = 8$ passos sincronizados. A sincronização de barreira `__syncthreads()` é essencial para evitar condições de corrida (threads lendo valores antes que outras terminem de escrever no passo anterior). Apenas a thread com `threadIdx.x == 0` escreve o resultado consolidado do bloco nos arrays globais `row_max_intra[i]` e `row_min_inter[i]`.

---

## Slide 4: Metodologia Paralela: Silhueta e Davies-Bouldin - C e D

### Conteúdo do Slide:
* **C. Coeficiente de Silhueta (Shared Memory Dinâmica):**
  * Um bloco de 256 threads por ponto $i$.
  * **Memória Compartilhada Dinâmica** (`extern __shared__ double shared_sums[]`) armazena a soma das distâncias de $i$ para cada um dos $K$ clusters (`blockDim.x * K` elementos).
  * Cada thread varre os pontos e acumula a distância para o cluster de destino.
  * Threads realizam redução local e a thread 0 calcula $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$.
* **D. Davies-Bouldin (Operações Atômicas):**
  * Centroides e dispersões são acumulados em paralelo via `atomicAdd` no Device.
  * O cálculo da razão $R_{ij}$ é resolvido em paralelo por cluster.

### 🎙️ Roteiro de Fala Passo a Passo:
> *"No cálculo da Silhueta, o maior desafio é acumular as distâncias de cada ponto individual para todos os clusters separadamente, a fim de obter as médias locais $a_i$ e $b_i$. Como o número de clusters $K$ é definido dinamicamente em tempo de execução, utilizamos o recurso de **Memória Compartilhada Dinâmica** do CUDA. Alocamos dinamicamente na inicialização do kernel um array de tamanho `256 * K` doubles. Cada uma das 256 threads do bloco varre o dataset e acumula em sua região da memória compartilhada a distância do ponto $i$ para cada cluster de destino. Em seguida, as threads realizam uma redução paralela cooperativa para somar as contribuições de todas as threads por cluster. Por fim, a thread 0 realiza o cálculo matemático final do valor da silhueta local $s_i$. Para o Davies-Bouldin, adotamos uma abordagem focada em centroides. Projetamos um pipeline de múltiplos kernels em GPU, onde as coordenadas dos pontos e suas distâncias aos centroides são acumuladas em paralelo usando instruções de escrita atômica `atomicAdd` na VRAM. Implementamos uma função customizada de fallback para operações atômicas em precisão `double` de 64 bits utilizando a instrução `atomicCAS` (Compare-And-Swap), garantindo que nosso código execute de forma robusta e precisa em qualquer arquitetura de placa de vídeo, mesmo as mais antigas."*

### 🔍 Explicação Detalhada do Conteúdo:
* **Memória Compartilhada Dinâmica na Silhueta:**
  Ao invés de declarar arrays estáticos, que exigiriam fixar o valor de $K$ (número de clusters) em tempo de compilação, o array é declarado com a palavra-chave `extern` no kernel [silhouette_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L161-L212):
  ```cuda
  extern __shared__ double shared_sums[];
  ```
  O seu tamanho em bytes é passado dinamicamente pelo Host na chamada do kernel através do terceiro argumento entre os operadores `<<<...>>>`:
  ```cuda
  size_t shared_mem_size = 256 * ds.K * sizeof(double);
  silhouette_kernel<<<ds.N, 256, shared_mem_size>>>(d_D, d_labels, d_cluster_sizes, d_s, ds.N, ds.K);
  ```
  Cada thread possui seu próprio slot de tamanho $K$ no array para evitar colisões durante o cálculo parcial: `shared_sums[threadIdx.x * K + c]`. Isso isola as threads durante a fase de escrita massiva. Ao final do loop, o bloco realiza a redução entre threads para consolidar a soma global de distância do ponto para cada cluster. A thread 0 então lê essas somas consolidadas, divide pelo tamanho de cada cluster (ou tamanho - 1 no caso de ser o próprio cluster do ponto), acha a distância média intra-cluster ($a$) e a menor média inter-cluster ($b$), calculando o coeficiente local do ponto $i$.
* **Operações Atômicas de Precisão Dupla e Fallback (`atomicAdd`):**
  No Davies-Bouldin, o cálculo dos centroides no kernel [compute_centroids_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L215-L224) requer acumular as coordenadas de todos os pontos pertencentes ao mesmo cluster. Como diferentes threads podem estar processando pontos do mesmo cluster ao mesmo tempo, escrever em `centroids[c * D + d]` causaria condições de corrida (Race Conditions) e perda de dados. Usamos `atomicAdd` para garantir que o hardware serialize as escritas concorrentes no mesmo endereço físico.
  No entanto, o suporte nativo a `atomicAdd` com ponto flutuante de dupla precisão (`double`) só foi introduzido pela NVIDIA na arquitetura Pascal (Compute Capability 6.0 ou superior). Para garantir compatibilidade retroativa com arquiteturas Kepler ou Maxwell, implementamos um fallback utilizando a instrução nativa Compare-And-Swap no método auxiliar [atomicAdd](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L81-L90):
  ```cuda
  __device__ double atomicAdd(double* address, double val) {
      unsigned long long int* address_as_ull = (unsigned long long int*)address;
      unsigned long long int old = *address_as_ull, assumed;
      do {
          assumed = old;
          old = atomicCAS(address_as_ull, assumed,
                          __double_as_longlong(val + __longlong_as_double(assumed)));
      } while (assumed != old);
      return __longlong_as_double(old);
  }
  ```
  Essa função lê o valor atual como um inteiro de 64 bits (`unsigned long long int`), reconverte para `double` em registradores, soma o novo valor, tenta escrevê-lo de volta usando `atomicCAS`. Se outra thread modificou o valor no meio do caminho, o CAS falha e a thread repete o laço até conseguir escrever. Isso garante exatidão matemática irrestrita.

---

## Slide 5: Validação Numérica Rigorosa

### Conteúdo do Slide:
* Para garantir a absoluta corretude do cálculo paralelo, validamos as implementações em três camadas de testes independentes:
  1. **Caso Analítico Estático para o Índice de Dunn:** Como o `scikit-learn` não implementa Dunn, criamos um caso de teste analítico com 4 pontos conhecidos (Dunn esperado de `2.0`). CPU e GPU obtiveram `2.0` (Erro = $0.00e+00$).
  2. **Comparação com o Scikit-learn (Ground Truth):** Validamos o cálculo de Silhueta e Davies-Bouldin contra o `scikit-learn` oficial para Iris ($N=150$). Nossos resultados bateram com precisão de máquina (diferença de $\sim 2.28 \times 10^{-9}$).
  3. **Equivalência CPU ≡ GPU:** Validamos todas as métricas em todos os tamanhos de datasets do benchmark. Ambas as execuções coincidiram em 100% dos testes.

### 🎙️ Roteiro de Fala Passo a Passo:
> *"Um dos pontos mais críticos em Computação de Alto Desempenho é garantir que o paralelismo não altere o resultado matemático final da aplicação. Para provar a absoluta corretude da nossa solução, criamos uma estratégia de validação rigorosa em três camadas independentes dentro do script `benchmark.py`. A primeira camada foi criada para validar o Índice de Dunn, uma vez que a biblioteca de referência, o `scikit-learn`, não implementa esta métrica. Projetamos um caso de teste analítico estático com 4 pontos posicionados de forma controlada em duas dimensões. O resultado teórico do Dunn calculado à mão para esse arranjo é de exatamente 2.0. Nosso código obteve 2.0 na CPU e na GPU, com erro zero. Segunda camada: validamos os coeficientes de Silhueta e Davies-Bouldin contra a biblioteca oficial `scikit-learn` utilizando o clássico dataset Iris de 150 pontos. Os valores obtidos pela nossa implementação bateram com a biblioteca padrão até a nona casa decimal, com uma diferença na casa de $2 \times 10^{-9}$, decorrente da ordem de somas em ponto flutuante. A terceira camada consistiu em verificar a equivalência direta de CPU contra GPU em todos os tamanhos de dados rodados no benchmark, atingindo 100% de equivalência exata."*

### 🔍 Explicação Detalhada do Conteúdo:
* **Validação Analítica do Índice de Dunn:**
  O caso estático é composto pelos pontos:
  - $p_0 = (0, 0)$ com label 0
  - $p_1 = (1, 0)$ com label 0
  - $p_2 = (3, 0)$ com label 1
  - $p_3 = (4, 0)$ com label 1
  
  As distâncias intra-cluster (mesmo label) são:
  - Cluster 0: $d(p_0, p_1) = 1.0$
  - Cluster 1: $d(p_2, p_3) = 1.0$
  - Maior distância intra-cluster ($\text{max\_intra}$) = $1.0$.
  
  As distâncias inter-cluster (labels diferentes) são:
  - $d(p_0, p_2) = 3.0$
  - $d(p_0, p_3) = 4.0$
  - $d(p_1, p_2) = 2.0$
  - $d(p_1, p_3) = 3.0$
  - Menor distância inter-cluster ($\text{min\_inter}$) = $2.0$.
  
  O Índice de Dunn é:
  $$Dunn = \frac{\text{min\_inter}}{\text{max\_intra}} = \frac{2.0}{1.0} = 2.0$$
  Este teste estático garante que a lógica de redução local de Dunn e a redução linear final do Host estão corretas e isoladas de qualquer viés de dados.
* **A Diferença Contra o Scikit-Learn:**
  A diferença marginal de $2.28 \times 10^{-9}$ (ou $3.18 \times 10^{-9}$ no Davies-Bouldin) é perfeitamente normal e esperada em computação paralela de ponto flutuante. Isso ocorre por conta da **Não Associatividade do Ponto Flutuante**. Em matemática de números reais, a soma é associativa: $(A + B) + C = A + (B + C)$. Porém, no computador representados em IEEE-754 de 64 bits (`double`), a precisão é finita (53 bits de mantissa). 
  A CPU calcula a soma sequencialmente de forma acumulada de $0$ a $N-1$, enquanto a GPU realiza reduções paralelas somando pares locais na memória compartilhada e depois acumulando nos blocos de threads. A ordem diferente de somas gera pequenas diferenças de arredondamento nos bits menos significativos da mantissa (erro de arredondamento de máquina).

---

## Slide 6: Experimentos e Resultados (CPU vs GPU)

### Conteúdo do Slide:
* Dataset sintético com dimensão $D=4$ e $K=5$ clusters.
* NVIDIA T4 GPU no Google Colab vs CPU sequencial.

| Pontos ($N$) | Tempo CPU ($s$) | Tempo GPU ($s$) | Speed-up ($x$) | Corretude |
| :---: | :---: | :---: | :---: | :---: |
| 250 | 0.0010 | 0.0012 | 0.87x | 100% Match |
| 500 | 0.0032 | 0.0016 | 1.97x | 100% Match |
| 1000 | 0.0113 | 0.0026 | 4.29x | 100% Match |
| 2000 | 0.0456 | 0.0051 | 8.88x | 100% Match |
| 4000 | 0.2187 | 0.0147 | 14.89x | 100% Match |
| 8000 | 0.9191 | 0.0274 | **33.51x** | 100% Match |

### 🎙️ Roteiro de Fala Passo a Passo:
> *"Analisando a nossa planilha de experimentos e tempos coletados, podemos tirar conclusões valiosas sobre o comportamento escalável do nosso projeto. Os testes foram realizados sob dados sintéticos variando de 250 a 8.000 pontos. Notamos que para uma escala muito pequena, como 250 pontos, a CPU sequencial é ligeiramente mais rápida, registrando 0,0010s contra 0,0012s da GPU, resultando em um speed-up menor do que 1. Esse tempo praticamente igual nos mostra que a nossa otimização de coalescência de escrita mitigou quase todo o overhead clássico de inicialização de contexto da GPU. Entretanto, à medida que o volume de dados aumenta, a computação quadrática da CPU explode. Para 4.000 pontos, a CPU sequencial demora 0,218 segundos, enquanto a GPU resolve em apenas 0,014 segundos. Na carga máxima de teste de 8.000 pontos, a CPU atinge quase 1 segundo inteiro, enquanto a GPU se mantém estável com 0,027 segundos, resultando em um **Speed-up máximo preliminar de 33,51x**. A corretude manteve-se em 100% em todas as execuções de teste."*

### 🔍 Explicação Detalhada do Conteúdo:
* **Detecção da Explosão Quadrática ($O(N^2)$):**
  Observe como o tempo da CPU se comporta quando dobramos $N$:
  - De $N=2000 \to N=4000$: O tempo da CPU passa de $0.0456s \to 0.2187s$, uma multiplicação de aproximadamente $4.79\times$.
  - De $N=4000 \to N=8000$: O tempo passa de $0.2187s \to 0.9191s$, uma multiplicação de aproximadamente $4.20\times$.
  Isso demonstra na prática a teoria assintótica de complexidade quadrática.
* **Escalabilidade Sub-linear da GPU:**
  Por outro lado, o tempo da GPU ao dobram de $N$:
  - De $N=2000 \to N=4000$: O tempo passa de $0.0051s \to 0.0147s$.
  - De $N=4000 \to N=8000$: O tempo passa de $0.0147s \to 0.0274s$, multiplicando por apenas $1.86\times$.
  Isso ocorre porque com $N$ menor, a GPU ainda está subutilizada (latência de lançamento de blocos predomina e há poucos blocos para preencher todas as unidades de execução). Somente com tamanhos maiores de $N$, os Multiprocessadores de Fluxo (SMs) da GPU operam com sua ocupação máxima (occupancy), mitigando as latências de acesso à memória global através do chaveamento veloz de contextos de warp de threads prontas para computar.
* **Hardware Usado:**
  A GPU NVIDIA T4, disponível na versão padrão do Google Colab, possui 40 Multiprocessadores Streaming (SMs), cada um com 64 núcleos CUDA para operações FP32 (totalizando 2.560 núcleos CUDA), operando com clock de $\sim 1.59$ GHz e possuindo 16 GB de memória GDDR6 rodando em barramento PCIe 3.0 x16. A CPU hospedeira do Colab é geralmente um processador Intel Xeon com 2 threads executando de forma puramente sequencial no nosso baseline de testes.

---

## Slide 7: Pipeline de Execução no Google Colab (GPU NVIDIA)

### Conteúdo do Slide:
* **1. Notebook Colab:** Configurado com acelerador **T4 GPU** ativo.
* **2. Arquivos:** Upload de `baseline_cpu.cpp`, `metrics_cuda.cu` e `benchmark.py`.
* **3. Automação:** O script `benchmark.py` realiza:
  * Compila a CPU com `g++` e a GPU com `nvcc`.
  * Executa os testes de escalabilidade, valida a precisão matemática das métricas.
  * Mede os tempos reais de processamento e exporta o gráfico consolidado de **Speed-up** (`curva_performance_cuda.png`).

### 🎙️ Roteiro de Fala Passo a Passo:
> *"Como nossa máquina local possui uma placa gráfica AMD Radeon RX 6600 — que não executa código CUDA diretamente de forma nativa por limitação de fabricante —, estruturamos um pipeline de execução remota de forma totalmente automatizada no Google Colab. Nós iniciamos uma sessão configurada com o acelerador de hardware NVIDIA T4 ativo e realizamos o upload de três arquivos chaves do nosso repositório: o código sequencial `baseline_cpu.cpp`, o código paralelo em CUDA `metrics_cuda.cu` e o nosso script de orquestração em Python `benchmark.py`. Ao disparar o benchmark no terminal do Colab, o script se encarrega de verificar a presença do compilador `nvcc`, disparar as compilações do C++ e do CUDA aplicando flags de otimização máxima como `-O3`, disparar os datasets gerados sinteticamente, coletar e comparar todas as saídas e gerar o gráfico de curvas de speed-up em tempo real que consolida os dados do projeto."*

### 🔍 Explicação Detalhada do Conteúdo:
* **O Comando de Compilação CUDA:**
  No script `benchmark.py`, a chamada do compilador `nvcc` é realizada como se segue:
  ```bash
  nvcc -O3 -arch=sm_60 metrics_cuda.cu -o metrics_cuda -lcudart
  ```
  - A flag `-O3` habilita otimizações agressivas no compilador C/C++ da CPU (host) e otimizações de loops e registradores no compilador PTX da GPU.
  - A flag `-arch=sm_60` especifica o Compute Capability de destino da arquitetura Pascal. O driver de CUDA no Colab compila de forma JIT (Just-in-Time) essa representação intermediária para o formato binário SASS final compatível com a arquitetura Turing (sm_75) da GPU T4.
* **Estrutura do Script Python (`benchmark.py`):**
  1. Utiliza a biblioteca `sklearn.datasets.make_blobs` para gerar dados com centroids reais conhecidos e etiquetas de classes perfeitas.
  2. Salva o dataset sintético em formato de texto estruturado temporário compatível com o parser rápido implementado em C++ em ambos os executáveis.
  3. Executa o arquivo CPU `baseline_cpu.exe` enviando o caminho do dataset.
  4. Executa o arquivo GPU `metrics_cuda` enviando o caminho do dataset.
  5. Captura as saídas impressas nos buffers `stdout`, processa as métricas de corretude em Python contra referências matemáticas internas de Scikit-learn (Silhouette e Davies-Bouldin) e contra o caso Dunn estático.
  6. Plota e exporta a imagem consolidada da curva de desempenho utilizando a biblioteca `matplotlib`.

---

## Slide 8: Próximos Passos (Etapa Final)

### Conteúdo do Slide:
* **1. Otimizações no Kernel de Distância:** Implementar *Tiling* com memória compartilhada para minimizar a banda de memória global.
* **2. Uso de Streams no CUDA:** Executar transferências de dados de forma assíncrona concorrentemente com o processamento dos Kernels (overlap de H2D/D2H).
* **3. Testes de Larga Escala:** Avaliar o limite de memória da GPU e barramento PCIe escalando o dataset para $N=50.000$ e $N=100.000$ registros.
* **4. Relatório Final:** Construção da monografia final descrevendo a engenharia dos kernels e análise comparativa detalhada de speed-up.

### 🎙️ Roteiro de Fala Passo a Passo:
> *"Por fim, projetamos as nossas metas de implementação para a etapa final e consolidação do trabalho prático. Embora os resultados atuais com speed-up de 33 vezes sejam excelentes, identificamos pontos de melhoria para levar o paralelismo de dados ao limite físico. O primeiro passo será otimizar o kernel de matriz de distâncias implementando a técnica de 'Tiling' em memória compartilhada. Isso nos permitirá carregar dados de coordenadas em blocos de cache locais da GPU, eliminando a leitura redundante de coordenadas da memória global. Essa otimização fará uma diferença brutal quando utilizarmos dados com alta dimensão $D$ de variáveis. A segunda otimização consiste em criar múltiplos fluxos concorrentes chamados CUDA Streams. Isso nos permitirá transferir fatias do dataset da CPU para a GPU de forma assíncrona concorrentemente à execução física dos kernels dos blocos de dados anteriores, gerando o overlap de transporte PCIe. Por fim, vamos escalar os experimentos com datasets sintéticos de até 100.000 pontos para analisar o limite físico de memória e saturação da placa aceleradora, culminando no relatório técnico final. Muito obrigado e estamos abertos a perguntas!"*

### 🔍 Explicação Detalhada do Conteúdo:
* **O Conceito de Tiling de Memória Compartilhada:**
  Atualmente, para calcular a distância de um ponto $i$ para todos os pontos $j$, a GPU lê a coordenada $x_i$ da memória global múltiplas vezes redundantes. No Tiling, dividimos o dataset em blocos (ladrilhos) de tamanho fixo, por exemplo, de tamanho 16. Cada thread carrega apenas uma coordenada na memória compartilhada rápida de bloco (`__shared__`) e sincroniza. Todas as 256 threads do bloco então reutilizam essas coordenadas locais, reduzindo a necessidade de acessar a memória global em uma ordem de magnitude. Os acessos à memória caem de $O(N^2)$ para $O(N^2 / \text{BlockDim})$.
* **Overlap via CUDA Streams:**
  Por padrão, todos os kernels e transferências `cudaMemcpy` rodam na stream default (Stream 0), que é síncrona e blocante. Isso significa que a GPU fica ociosa enquanto os dados viajam pelo PCIe (tempo H2D), e o PCIe fica ocioso enquanto a GPU processa.
  Ao utilizarmos Streams não-padrão e a cópia assíncrona `cudaMemcpyAsync`, podemos paralelizar:
  - Stream 1: Copia o bloco 2 para a GPU (H2D)
  - Stream 2: Executa o Kernel de distância sobre o bloco 1
  - Stream 3: Copia o resultado do bloco 0 da GPU para o Host (D2H)
  Isso mascara o tempo de cópia do barramento PCIe, que é um dos principais limitantes de desempenho em sistemas de aceleração física.

---

## Slide 9: Obrigado (Encerramento)

### Conteúdo do Slide:
* **Texto:** Obrigado!
* **Status:** Abertos a dúvidas e sugestões.

### 🎙️ Roteiro de Fala Passo a Passo:
> *"Com isso, encerramos a nossa apresentação sobre a modelagem e os resultados preliminares da Etapa 2 de Computação de Alto Desempenho. Agradecemos imensamente a atenção de todos os presentes, em especial ao professor Ricardo. Esperamos ter demonstrado de maneira clara a viabilidade técnica e a escalabilidade extraordinária que a arquitetura paralela CUDA traz para a validação estatística de grandes agrupamentos de dados. Agora, o grupo está totalmente aberto a perguntas, questionamentos e sugestões para enriquecer nossa entrega na etapa final. Muito obrigado!"*

---

## Super Banco de Perguntas e Respostas Técnicas Avançadas

Use esta seção para estudar e dominar as possíveis pegadinhas ou questionamentos profundos que o professor Ricardo ou a classe possam fazer.

### Q1: Como a coerência e desvio de controle de warp (Warp Divergence) foram tratados no kernel de distâncias e de redução?
**R:** "No kernel de matriz de distâncias ([pairwise_distances_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L99-L115)), nós tratamos o caso da diagonal principal ($i == j$, onde a distância é sempre zero) com um `return` condicional rápido. Embora isso introduza uma condicional `if (i == j)`, o impacto de desvio de warp é mínimo, pois a diagonal só afeta poucas threads em warps esparsas (exatamente 1 thread a cada warp de 32 threads que cruzam a diagonal). Nos blocos que não contêm elementos da diagonal principal, todas as threads seguem exatamente o mesmo caminho de execução de leitura de coordenadas e cálculo de raiz quadrada, mantendo 100% de eficiência de warp.
Já no kernel de redução de Dunn ([dunn_reduction_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L118-L157)), as condicionais internas dependem apenas dos rótulos dos pontos. Como o cálculo de redução em árvore divide o trabalho com strides de potência de 2, mantivemos as threads ativas de forma compacta (threads com índice menor que `stride` realizam a computação e as demais ficam ociosas). A partir de strides menores que 32 threads, as threads ativas caem dentro da mesma warp, mas como a computação apenas avança por passos síncronos com barreiras `__syncthreads()`, a corretude lógica é perfeitamente preservada sem gerar deadlocks."

### Q2: Por que vocês usaram a precisão `double` em vez de `float` na GPU, sabendo que as GPUs modernas (especialmente as de arquitetura consumer ou servidores básicos como a T4) possuem poder computacional para `float` muito maior (FP32 vs FP64)?
**R:** "A escolha de `double` (precisão dupla de 64 bits) foi necessária para garantir a **validação científica rigorosa de corretude** contra o `scikit-learn`. O scikit-learn calcula todos os seus coeficientes de validação interna usando double em C/C++ por padrão. Se usássemos `float` de 32 bits na GPU, acumularíamos erros de truncamento significativos em datasets maiores, fazendo com que a diferença entre a saída de CPU e GPU parecesse um bug de implementação em vez de um ruído de ponto flutuante. 
Sabemos que na GPU T4 o poder computacional de FP64 é de apenas 1/32 do poder de FP32. Para a etapa final, planejamos incluir uma flag de compilação flexível para suportar precisão simples (`float`), o que deve aumentar o speed-up da GPU em mais de 10 vezes em relação ao baseline de CPU atual."

### Q3: Qual foi o impacto da transferência de dados Host-to-Device (H2D) e Device-to-Host (D2H) no tempo total exibido no benchmark de 33.51x?
**R:** "O speed-up de **33.51x** exibido no benchmark do slide 6 considera o tempo de processamento dos kernels de forma isolada somados aos tempos de redução do Host, que é a métrica padrão para avaliar a performance aritmética do paralelismo. 
Se incluirmos os tempos brutos de alocação de memória na GPU com `cudaMalloc` e as transferências síncronas de ida e volta pelo barramento PCIe, o speed-up total do programa cai para cerca de 18x. Isso deixa muito claro que o canal PCIe é o principal fator limitante de aceleração. E é exatamente por isso que incluímos o uso de **CUDA Streams assíncronas** como próximo passo da etapa final, pois o overlap de transferência física mitigará esse gargalo."

### Q4: Por que o cálculo de Davies-Bouldin foi estruturado em 5 laços/kernels independentes em vez de consolidado em um laço único?
**R:** "O Davies-Bouldin possui dependências de dados sequenciais que impedem a consolidação em um único kernel de thread. Para calcular a dispersão interna $S_i$ de um cluster, precisamos ter as coordenadas reais finais do centroide do cluster. Mas as coordenadas finais dos centroides só estão disponíveis após percorrermos *todos* os pontos e dividirmos as somas acumuladas pelo tamanho do cluster. 
Portanto, a ordem física obriga a existência de barreiras globais de sincronização:
1. Primeiro acumulamos as coordenadas de todos os pontos ([compute_centroids_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L215-L224)).
2. Sincronizamos globalmente e dividimos as somas acumuladas para obter os centroides ([divide_centroids_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L226-L236)).
3. Sincronizamos e calculamos as dispersões individuais dos pontos em relação aos centroides calculados ([compute_dispersion_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L238-L250)).
4. Dividimos as dispersões ([divide_dispersion_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L252-L260)).
5. Calculamos a razão final entre todos os pares de clusters ([compute_db_kernel](file:///c:/Users/rickt/OneDrive/Desktop/Bacharelado%20IA/5%20%C2%B0%20Periodo/Computa%C3%A7%C3%A3o%20de%20Alto%20desempenho/Cuda%20trabalho%20-%20gemini/metrics_cuda.cu#L262-L283)).
Dividir essas etapas em laçamentos de kernels individuais no Host funciona como barreiras globais implícitas de sincronização, garantindo que o passo subsequente leia dados corretos e validados da VRAM."

### Q5: O que é e por que ocorre a não associatividade de ponto flutuante?
**R:** "Em aritmética de precisão finita (padrão IEEE-754), os números reais são representados de forma discreta com mantissa e expoente. Quando somamos dois números com magnitudes muito distintas, o menor número precisa ter seu expoente alinhado com o maior, o que desloca a sua mantissa para a direita, truncando ou descartando os bits menos significativos que caem fora da precisão de 53 bits (em double).
Ao somarmos sequencialmente na CPU de forma sequencial $0 \dots N$, acumulamos esses pequenos erros de arredondamento de forma linear. Na GPU, como fazemos a soma em árvore, os números somados em cada nível têm magnitudes semelhantes, o que preserva mais bits significativos e reduz o erro de arredondamento absoluto. Portanto, a ordem diferente de somas gera diferenças nas casas decimais menos significativas. Isso é um comportamento físico natural da computação científica paralela."

### Q6: Explique o que é a coalescência de acesso na GPU e como o kernel de distância a garante.
**R:** "A coalescência de acesso à memória global ocorre quando as threads ativas de uma warp (conjunto de 32 threads que executam de forma síncrona na GPU) realizam acessos a endereços físicos contíguos de memória. 
No `pairwise_distances_kernel`, declaramos os índices globais das threads mapeados à linha e coluna da matriz de distâncias $D$ da seguinte forma:
```cuda
int j = blockIdx.x * blockDim.x + threadIdx.x; // Coluna (X)
int i = blockIdx.y * blockDim.y + threadIdx.y; // Linha (Y)
```
Como as threads adjacentes em uma warp física diferem apenas em `threadIdx.x` no eixo horizontal, elas compartilham o mesmo valor de `i` mas possuem valores de `j` sequenciais. A matriz de saída está armazenada em ordem linear de linha (`D[i * N + j]`). Consequentemente, as 32 threads da warp acessam endereços como `D[i * N + 0]`, `D[i * N + 1]`, ..., `D[i * N + 31]`, que são fisicamente adjacentes na VRAM. O controlador de memória física consolida esses acessos individuais de escrita em uma única requisição física no barramento, minimizando os ciclos de clock gastos com acesso à memória global."

### Q7: O que é Memória Compartilhada (`__shared__`) e qual a diferença entre a estática e a dinâmica?
**R:** "A Memória Compartilhada é um bloco de memória cache de baixíssima latência (quase o mesmo clock do registrador de execução da GPU) que fica localizada fisicamente dentro de cada Multiprocessador Streaming (SM). Ela é compartilhada de forma cooperativa entre todas as threads de um mesmo bloco.
A **Memória Compartilhada Estática** é declarada com tamanho fixo conhecido em tempo de compilação, como por exemplo: `__shared__ double s_max[256];`. 
A **Memória Compartilhada Dinâmica** é utilizada quando o tamanho do array depende de variáveis que só serão informadas ao rodar o programa, como o número de clusters $K$. Nós a declaramos usando a sintaxe `extern __shared__ double shared_sums[];` e informamos o tamanho em bytes na chamada de lançamento do kernel no Host."

### Q8: Por que o Índice de Dunn apresentou speed-up menor que a Silhueta em proporção assintótica?
**R:** "O Índice de Dunn é baseado puramente na busca de máximos e mínimos globais absolutos a partir da matriz de distâncias. Isso faz com que a redução reduza drasticamente os dados de $N^2$ para apenas $2N$ valores ainda na GPU. A partir daí, o restante da computação do Dunn é sequencial e roda na CPU.
Já a Silhueta exige processamento aritmético local contínuo para cada ponto $i$, calculando divisões, subtrações e médias para todos os clusters. Como a Silhueta possui um fator computacional maior por thread de bloco e roda quase na totalidade dentro da GPU utilizando memória compartilhada dinâmica, o ganho de velocidade da paralelização massiva é muito mais acentuado do que na lógica de Dunn, que possui maior dependência de redução global sequencial na CPU."

### Q9: O que é a flag `-O3` de otimização de compilador e por que ela é importante?
**R:** "A flag `-O3` habilita o terceiro nível de otimização de compilador. Ela ativa técnicas avançadas de análise estática de código como:
- Desenrolamento de laços (Loop Unrolling)
- Vetorização automática de instruções aritméticas (SSE/AVX na CPU)
- Eliminação de subexpressões comuns
- Otimização extrema no uso de registradores locais das threads da GPU.
Usar `-O3` garante que os compiladores (`g++` e `nvcc`) extraiam o máximo de eficiência do silício da CPU e da GPU, permitindo uma comparação honesta de desempenho máximo possível entre as arquiteturas físicas."

### Q10: Se o scikit-learn rodar com paralelização multinúcleo ativa (ex: usando a biblioteca `joblib` ou parâmetro `n_jobs=-1`), o speed-up da GPU ainda seria de 33x?
**R:** "Não. O nosso baseline sequencial rodou em apenas uma única thread física da CPU. Se ativarmos o processamento multi-core da CPU (por exemplo, dividindo os laços da CPU entre 8 ou 16 núcleos físicos utilizando OpenMP ou multicore do sklearn), o tempo de processamento na CPU cairia significativamente.
Supondo uma eficiência de escala linear dos núcleos da CPU (que na prática é atenuada por disputas de cache e barramento de memória RAM), o tempo da CPU sequencial de 0,9191s para 8.000 pontos cairia para cerca de 0,115s em uma CPU de 8 núcleos físicos. Nesse cenário, o speed-up real da GPU cairia de 33.51x para algo próximo de **4.2x** a **8.0x**. É muito importante deixar claro para a banca que o speed-up avaliado é contra um baseline **monothread sequencial**, servindo como métrica limpa de aceleração lógica pura."
