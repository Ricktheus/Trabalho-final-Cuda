# Explicação Completa do Código — Trabalho Final (Validação de Clusters em GPU)

> **Para que serve este documento:** explicar **cada arquivo e cada bloco de código** do projeto, com o *porquê* de cada decisão, para que qualquer integrante consiga defender qualquer linha na apresentação. Leia junto com o `GUIA_CONCEITUAL.md` (que explica os *conceitos* com analogias).

---

## 0. Visão geral: o que cada arquivo faz

| Arquivo | Papel |
|---|---|
| `baseline_cpu.cpp` | **Baseline** em C++ (CPU). Calcula as 3 métricas. Tem versão sequencial **e** paralela (OpenMP). É o que a GPU precisa **superar**. |
| `metrics_cuda.cu` | A **proposta**: os mesmos cálculos paralelizados em **GPU com CUDA**. |
| `benchmark.py` | O **maestro**: gera dados, compila tudo, **valida a corretude**, mede tempos com repetições e dispara os gráficos. |
| `gerar_graficos.py` | Lê o CSV de resultados e desenha os **gráficos finais** (não precisa de GPU). |
| `resultados_benchmark.csv` | A **tabela de números** finais (fonte da verdade para slides e artigo). |
| `slides_apresentacao_final.md` | Roteiro dos slides da apresentação final. |

**Fluxo de uma execução completa (no Colab com GPU):**

```
benchmark.py
  ├─ compila baseline_cpu.cpp  (g++ -O3 -fopenmp)
  ├─ compila metrics_cuda.cu   (nvcc, em double e float)
  ├─ VALIDA: caso analítico do Dunn + comparação com scikit-learn
  ├─ para cada N (250 ... 100000):
  │     ├─ roda CPU-1thread, CPU-OpenMP e GPU  (várias repetições)
  │     ├─ confere CPU ≡ GPU (corretude)
  │     └─ guarda média ± desvio
  ├─ salva resultados_benchmark.csv
  └─ chama gerar_graficos.py  →  bench_tempo / bench_speedup / bench_breakdown .png
```

---

## 1. Formato dos dados

Todos os programas leem o mesmo arquivo `.csv` (na verdade separado por espaços):

```
N D K                  <- 1ª linha: nº de pontos, nº de dimensões, nº de clusters
x1 x2 x3 x4 label      <- N linhas: as D coordenadas + o rótulo do cluster
...
```

Exemplo (4 pontos, 2 dimensões, 2 clusters) — é o **caso de teste do Dunn**:
```
4 2 2
0.0 0.0 0
1.0 0.0 0
5.0 0.0 1
5.0 2.0 1
```

Os dados de benchmark são **sintéticos**, gerados com `make_blobs` do scikit-learn (nuvens gaussianas), com `D=4` dimensões e `K=5` clusters.

---

## 2. `baseline_cpu.cpp` — o baseline da CPU

### 2.1. A estrutura `Dataset`
```cpp
struct Dataset {
    int N, D, K;
    std::vector<double> X;       // coordenadas "achatadas": ponto i, dim d => X[i*D + d]
    std::vector<int> labels;     // rótulo de cada ponto
    std::vector<int> unique_labels;
};
```
- `X` é um vetor **linear** (1D) representando uma matriz N×D. O ponto `i`, dimensão `d`, fica em `X[i*D + d]`. Isso é mais rápido que um vetor de vetores (memória contígua).

### 2.2. `load_dataset` — leitura + normalização dos rótulos
- Lê `N D K`, depois `N` linhas.
- **Detalhe importante:** os rótulos do arquivo podem ser quaisquer inteiros (ex.: 7, 42...). O código os **remapeia para 0..K-1** usando um `std::set` (que ordena os valores únicos) e `std::find`. Isso garante que `labels[i]` sempre sirva de **índice** direto em vetores de tamanho K.

### 2.3. `dist_ij` — a distância calculada *na hora* (coração do "matrix-free")
```cpp
static inline double dist_ij(const Dataset& ds, int i, int j) {
    double sum = 0.0;
    for (int d = 0; d < ds.D; ++d) {
        double diff = ds.X[i*ds.D + d] - ds.X[j*ds.D + d];
        sum += diff * diff;
    }
    return std::sqrt(sum);     // distância euclidiana
}
```
- É a fórmula da distância euclidiana: $\sqrt{\sum_d (x_{i,d}-x_{j,d})^2}$.
- **Por que isso importa:** a versão antiga guardava todas as distâncias numa matriz N×N. Agora **recalculamos** quando precisamos. Custa um pouco mais de conta, mas economiza memória gigantesca (ver seção 4.7).

### 2.4. `compute_dunn_index` — Dunn + redução OpenMP
```cpp
double max_intra = 0.0;
double min_inter = +infinito;

#pragma omp parallel for schedule(dynamic, 64) \
        reduction(max:max_intra) reduction(min:min_inter)
for (int i = 0; i < ds.N; ++i)
    for (int j = i+1; j < ds.N; ++j) {
        double dist = dist_ij(ds, i, j);
        if (labels[i] == labels[j]) max_intra = max(max_intra, dist);  // mesmo cluster
        else                        min_inter = min(min_inter, dist);  // clusters diferentes
    }
return (max_intra==0) ? 0 : min_inter / max_intra;
```
- **Dunn** = (menor distância entre clusters diferentes) / (maior distância dentro de um cluster).
- O `j` começa em `i+1`: como a distância é simétrica, só olhamos cada par **uma vez** (metade do triângulo).
- **`#pragma omp parallel for`**: distribui as iterações do laço `i` entre as threads da CPU.
- **`reduction(max:...)` / `reduction(min:...)`**: cada thread mantém seu próprio `max_intra`/`min_inter` privado e, no fim, o OpenMP combina todos pegando o máximo/mínimo global. Isso evita **condição de corrida** (duas threads escrevendo na mesma variável ao mesmo tempo).
- **`schedule(dynamic, 64)`**: como o laço interno é triangular (linhas de cima têm mais trabalho que as de baixo), o agendamento dinâmico em blocos de 64 equilibra a carga entre as threads.

### 2.5. `compute_silhouette_index` — Silhueta + OpenMP
Para cada ponto `i`:
1. Soma as distâncias de `i` a **cada cluster** num vetor `dist_sum[K]`.
2. `a` = distância média ao **próprio** cluster (divide por `tamanho-1`, porque não conta a distância a si mesmo).
3. `b` = **menor** distância média a um cluster **vizinho**.
4. silhueta do ponto: `s_i = (b - a) / max(a, b)`.
5. Soma todos os `s_i` e divide por N → silhueta média.
- Aqui o `reduction(+:silhouette_sum)` soma as contribuições das threads.
- Cada thread cria seu próprio `dist_sum` local (declarado **dentro** do laço) → sem conflito entre threads.

### 2.6. `compute_davies_bouldin_index` — DB (barato, O(N))
1. **Centróides:** soma as coordenadas de cada cluster e divide pelo tamanho (o "centro de massa").
2. **Dispersão `S[c]`:** distância média dos pontos do cluster `c` ao seu centróide.
3. **Razão:** para cada cluster `i`, acha o pior `(S_i + S_j) / M_ij`, onde `M_ij` é a distância entre os centróides. DB = média dessas piores razões.
- Note que DB **não** usa distâncias par-a-par → custa O(N), não O(N²). Por isso é praticamente instantâneo nos gráficos.

### 2.7. `main` — medição de tempo e saída
- Mede o tempo de cada etapa com `std::chrono`.
- Lê `OMP_NUM_THREADS` (via `omp_get_max_threads()`) e imprime `Threads:` — é assim que o benchmark roda a **mesma** binária como "1 thread" ou "N threads".
- Imprime as métricas e os tempos em linhas `chave: valor`, que o Python lê depois.

> **Compilação:** `g++ -O3 -fopenmp baseline_cpu.cpp -o baseline_cpu`
> Sem `-fopenmp` ele ainda compila (os `#pragma` são ignorados) e roda sequencial — graças ao `#ifdef _OPENMP`.

---

## 3. `metrics_cuda.cu` — a versão GPU (CUDA)

### 3.1. Precisão configurável (`real_t`)
```cpp
#ifdef USE_FLOAT
  typedef float real_t;   ...
#else
  typedef double real_t;  ...
#endif
```
- Compilando com `-DUSE_FLOAT`, as coordenadas e as distâncias usam `float` (mais rápido na T4, que é fraca em `double`). Sem a flag, usa `double` (mais preciso). **As somas/reduções acumulam sempre em `double`** para não perder precisão.

### 3.2. `atomicAdd` para `double` (compatibilidade)
- GPUs antigas (anteriores à arquitetura Pascal/sm_60) não têm `atomicAdd` para `double` em hardware. O bloco com `atomicCAS` fornece uma versão de software (só é usada se a GPU for antiga). Na T4 (sm_75) usa-se a nativa.

### 3.3. `dev_dist` — distância on-the-fly na GPU
Igual ao `dist_ij` da CPU, mas roda no device. Recebe o ponto `i` já em memória rápida (`xi`) e o ponto `j` da memória global. Acumula em `real_t` (rápido) e devolve `double`.

### 3.4. `dunn_rowwise_kernel` — o kernel do Dunn (linha por bloco)
**Ideia:** lançamos **N blocos** de **256 threads**. O bloco `i` cuida do ponto `i`.

```cpp
int i = blockIdx.x;     // qual ponto este bloco trata
int tx = threadIdx.x;   // qual thread dentro do bloco (0..255)

__shared__ real_t s_xi[MAX_D];      // coordenadas do ponto i (memória compartilhada)
__shared__ double s_max[256], s_min[256];

// 1) carrega x_i na shared memory (reuso por todas as threads do bloco)
for (int d = tx; d < D; d += blockDim.x) s_xi[d] = X[i*D + d];
__syncthreads();

// 2) cada thread varre uma fatia dos pontos j (grid-stride)
double local_max = 0, local_min = +inf;
for (int j = tx; j < N; j += blockDim.x) {
    if (j == i) continue;
    double dist = dev_dist(s_xi, X, j, D);     // distância calculada na hora
    if (labels[j] == labels[i]) local_max = max(local_max, dist);
    else                        local_min = min(local_min, dist);
}

// 3) reduz os 256 valores parciais em um só (redução em árvore)
s_max[tx] = local_max; s_min[tx] = local_min; __syncthreads();
for (int stride = 128; stride > 0; stride /= 2) {
    if (tx < stride) { s_max[tx] = max(s_max[tx], s_max[tx+stride]);
                       s_min[tx] = min(s_min[tx], s_min[tx+stride]); }
    __syncthreads();
}

// 4) a thread 0 grava o resultado da linha
if (tx == 0) { row_max_intra[i] = s_max[0]; row_min_inter[i] = s_min[0]; }
```

Pontos para defender:
- **`__shared__`**: memória ultrarrápida compartilhada pelas threads do bloco. Carregamos `x_i` ali **uma vez** e as 256 threads reusam.
- **grid-stride (`j += blockDim.x`)**: 256 threads varrem N pontos em "pente", então um bloco cobre uma linha inteira independentemente de N.
- **redução em árvore**: em 8 passos (log₂256) combinamos 256 valores em 1. `__syncthreads()` garante que todas as threads terminem um passo antes do próximo (senão, condição de corrida).
- **A CPU faz só a redução final O(N)**: recebe dois vetores de tamanho N (`row_max_intra`, `row_min_inter`) e tira o máximo/mínimo global. Em vez de trazer a matriz N×N (gigante) de volta, trazemos só 2 vetores pequenos.

### 3.5. `silhouette_rowwise_kernel` — Silhueta
- Também **1 bloco por ponto**. Usa **memória compartilhada dinâmica** `shared_sums[blockDim.x * K]` (tamanho definido no lançamento, porque depende de K).
- Cada thread acumula as distâncias de `i` para os pontos `j` **separadas por cluster** (`shared_sums[tx*K + cluster_de_j]`).
- Depois há uma **redução por cluster** (as K primeiras threads somam suas colunas).
- A thread 0 calcula `a`, `b` e `s_i`.

### 3.6. Kernels do Davies-Bouldin
Quatro/cinco kernels pequenos encadeados:
1. `compute_centroids_kernel`: cada ponto soma suas coordenadas no centróide do seu cluster via **`atomicAdd`** (várias threads escrevendo no mesmo centróide → precisa ser atômico).
2. `divide_centroids_kernel`: divide pela contagem → média.
3. `compute_dispersion_kernel`: cada ponto soma sua distância ao centróide (atomicAdd).
4. `divide_dispersion_kernel`: divide → dispersão média.
5. `compute_db_kernel`: 1 thread por cluster acha a pior razão.

### 3.7. `main` — orquestração na GPU
Sequência:
1. `cudaMalloc` para `d_X` e `d_labels` — **note que NÃO há mais `d_D` (a matriz N×N)**. Só O(N·D).
2. `cudaMemcpy` Host→Device (copia os dados para a GPU). Tempo medido = `Time_H2D`.
3. Para cada métrica: lança o kernel, mede com `cudaEvent` (cronômetro da GPU), copia o resultado pequeno de volta (Device→Host) e faz a redução final na CPU.
4. Imprime métricas + tempos por etapa (é isso que alimenta o gráfico de breakdown).

### 3.8. Por que "matrix-free" (a decisão mais importante)
A matriz N×N de `double` ocupa `N² × 8 bytes`:

| N | Matriz N×N | Cabe? |
|---:|---:|:--|
| 8.000 | 0,5 GB | sim |
| 50.000 | 20 GB | **não** (T4 tem 16 GB) |
| 100.000 | 80 GB | **não** (e estoura a RAM da CPU) |

Além disso, `N*N` com `N=100000` **estoura o `int`** (overflow). Calculando as distâncias on-the-fly, a memória vira O(N·D) ≈ 3 MB em 100k. **É isso que nos permitiu escalar até 100.000 pontos.**

---

## 4. `benchmark.py` — o maestro

### 4.1. Configuração
```python
TAMANHOS = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 50000, 100000]
DIMENSOES = 4 ; CLUSTERS = 5 ; ENABLE_FLOAT = True
def reps_for(n): return 5 se n<=8000, 3 se n<=32000, senão 2
```
- `reps_for`: faz **menos repetições** para N grande (porque a CPU em 100k é lenta). Mais repetições em N pequeno (onde o tempo é ruidoso).
- **`--max N`**: filtro `TAMANHOS = [n for n in TAMANHOS if n <= N]`. Por isso `--max 8000` corta os grandes; **sem** `--max`, roda até 100000.

### 4.2. Geração de dados — `gerar_dataset_csv`
Usa `make_blobs` com `random_state=42` (sempre os mesmos dados → reprodutível) e grava no formato `.csv` descrito na seção 1.

### 4.3. Compilação — `compilar_cpp` / `compilar_cuda`
- CPU: `g++ -O3 -fopenmp`.
- GPU: `nvcc -O3 -arch=sm_60` (e uma segunda vez com `-DUSE_FLOAT`).
- Se `nvcc` não existir (sem GPU), segue só com a CPU.

### 4.4. Execução e medição — `rodar_uma_vez` / `medir`
- `rodar_uma_vez`: roda o binário, lê a saída `chave: valor` e devolve um dicionário. Para a CPU, define `OMP_NUM_THREADS` via variável de ambiente — é assim que controlamos "1 thread" vs "N threads" **com a mesma binária**.
- `medir`: repete `reps` vezes e devolve **média e desvio-padrão** do `Time_Total`.

### 4.5. Validações (executadas antes do benchmark)
- `validar_dunn_analitico`: roda o caso dos 4 pontos; o Dunn **tem** que dar 2.0 (na CPU e na GPU).
- `validar_contra_sklearn`: gera o Iris-like (N=150), roda nosso C++ e compara Silhueta/DB com o `scikit-learn`. Se a diferença passar de 1e-5, **aborta**.
- Mensagem: isso prova que **não trocamos exatidão por velocidade**.

### 4.6. Laço principal — os 3 motores
Para cada N: roda CPU-1 (`OMP_NUM_THREADS=1`), CPU-OMP (`OMP_NUM_THREADS=todos`) e GPU; confere `|CPU - GPU| < 1e-5` para Dunn/Silhueta/DB (coluna **Match**); calcula speed-up; imprime a linha.

### 4.7. Saída — `salvar_csv` + `gerar_graficos`
- Salva tudo em `resultados_benchmark.csv` (inclusive o breakdown `GPU_H2D/Dunn/Silhueta/DB`).
- Chama `gerar_graficos.py` para desenhar os gráficos a partir do CSV.

---

## 5. `gerar_graficos.py` — os gráficos finais

Lê `resultados_benchmark.csv` e gera 3 imagens (sem precisar de GPU):

1. **`bench_tempo.png`** (2 painéis): **linear** (mostra o tamanho real do abismo CPU×GPU, com o gap de 100k anotado) + **log-log** (mostra que ambos crescem como O(N²) — retas paralelas — e a distância vertical é o speed-up).
2. **`bench_speedup.png`**: speed-up com **eixo log em N** (espalha os N pequenos; sem o "pico falso"). Marca a **zona de overhead** (N pequeno, tempos < 1 ms, ruidosos) e a anomalia do OpenMP em N=250.
3. **`bench_breakdown.png`** (2 painéis): **linhas log-log** das 4 etapas (todas visíveis em qualquer N) + **barras 100% empilhadas** (composição: Dunn+Silhueta ≈ 99,9% do tempo).

> Pode rodar localmente: `python gerar_graficos.py` (só precisa de `pandas` e `matplotlib`).

---

## 6. Pipeline no Google Colab
1. Ativar a GPU T4 (*Ambiente de execução → Alterar o tipo*).
2. Clonar o repositório.
3. `python benchmark.py` (ou `--max 8000` para teste rápido).
4. Baixar `resultados_benchmark.csv` e os `bench_*.png`.

---

## 7. "Se perguntarem X, aponte para Y" (mapa rápido de defesa)

| Pergunta provável | Onde está a resposta |
|---|---|
| "Como vocês garantem que o resultado está certo?" | 3 camadas de validação (`benchmark.py`, seção 4.5) |
| "Por que conseguem rodar 100 mil pontos?" | matrix-free (seção 3.8) |
| "O speed-up não é injusto contra 1 thread?" | medimos **também** contra CPU OpenMP (seção 4.6) |
| "Onde a GPU gasta o tempo?" | breakdown (Dunn+Silhueta dominam; seção 5) |
| "Streams/overlap não ajudariam?" | H2D é ~0 no matrix-free → ganho irrelevante (breakdown) |
| "Por que `double` e não `float`?" | trade-off precisão×velocidade; suportamos os dois (seção 3.1) |
| "Por que a redução em árvore precisa de `__syncthreads`?" | evita condição de corrida entre passos (seção 3.4) |
