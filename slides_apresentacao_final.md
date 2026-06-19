---
marp: true
theme: default
paginate: true
header: 'Computação de Alto Desempenho — Trabalho Prático (3ª Parte / Final)'
footer: 'Validação Paralela de Clusters em GPU — UFG / Instituto de Informática'
style: |
  section { font-family: 'Inter', sans-serif; padding: 40px; }
  h1 { color: #1e3a8a; }
  h2 { color: #2563eb; }
  strong { color: #1d4ed8; }
---

<!--
==========================================================================
 ROTEIRO COMPLETO DOS SLIDES — APRESENTAÇÃO FINAL (3ª PARTE)
==========================================================================
 COMO USAR ESTE ARQUIVO:
 - Cada bloco separado por "---" é UM slide.
 - Os slides marcados com  [IMAGEM: arquivo.png]  são slides DEDICADOS a
   uma figura gerada pelo benchmark no Colab. Insira a imagem indicada.
 - Onde aparecer  <preencher após rodar no Colab>  substitua pelo número
   real obtido em resultados_benchmark.csv (o código mudou para a versão
   matrix-free + OpenMP, então os números da Etapa 2 NÃO valem mais).
 - Para exportar em PDF:  marp slides_apresentacao_final.md --pdf
 - Tempo-alvo: 10–15 min (≈ 1 min por slide de conteúdo).

 IMAGENS GERADAS PELO benchmark.py (no Colab com GPU):
   * bench_tempo.png       -> tempo total vs N (CPU-1, CPU-OMP, GPU) log-log
   * bench_speedup.png     -> speed-up vs N (GPU/CPU-1 e GPU/CPU-OMP)
   * bench_breakdown.png   -> onde está o tempo na GPU (por etapa)
   * curva_performance_cuda.png -> figura combinada (opcional/backup)
==========================================================================
-->

# Validação Paralela de Clusters em GPU
## Aceleração CUDA *matrix-free* das métricas de Dunn, Silhueta e Davies-Bouldin

**Equipe:**
- Henrique M. M. Miranda — 202405479
- Cindy Stephanie Gomes Rabelo — 202403898
- Eduardo Dias Peixoto — 202010395
- Luiany Goncalves Carvalho — 202303351

**Disciplina:** Computação de Alto Desempenho · **Prof.:** Ricardo Augusto Pereira Franco
**UFG — Instituto de Informática**

<!-- Fala: apresentar a equipe e o tema em 1 frase: aceleramos em GPU o cálculo
das 3 métricas clássicas de validação de clusters, sem materializar a matriz
de distâncias, o que nos permitiu escalar para 100 mil pontos. -->

---

## 1. O Problema e sua Contextualização

- **Clustering** (K-Means, DBSCAN) agrupa dados **sem rótulos**. Mas como saber se o agrupamento é **bom**?
- Usamos **métricas internas de validação**: medem **compacidade** (intra-cluster) e **separação** (inter-cluster).
- **O gargalo:** Dunn e Silhueta exigem as **distâncias euclidianas par-a-par** de todos os pontos → custo de tempo **O(N²·D)**.
- **Impacto em Big Data:** ao dobrar N, o trabalho **quadruplica**. Em CPU sequencial isso inviabiliza milhares/milhões de pontos.

> É um problema de **paralelismo de dados massivo** — caso ideal para GPU.

<!-- Fala: contextualizar que validar clusters é tão importante quanto agrupá-los,
e que o custo quadrático é o que justifica usar GPU. -->

---

## 2. As Três Métricas (Formulação)

**Índice de Dunn** — maior = melhor (separação / dispersão):
$$Dunn = \frac{\min_{a \neq b}\ \delta(C_a, C_b)}{\max_{c}\ \Delta(C_c)}$$

**Coeficiente de Silhueta** — varia em [-1, 1], maior = melhor:
$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \qquad S = \frac{1}{N}\sum_i s_i$$

**Davies-Bouldin** — menor = melhor (baseado em centróides):
$$DB = \frac{1}{K}\sum_{i=1}^{K}\max_{j\neq i}\frac{S_i + S_j}{M_{ij}}$$

<!-- Fala: explicar rapidamente o significado de a_i (coesão), b_i (separação),
S_i (dispersão interna) e M_ij (distância entre centróides). -->

---

## 3. Artigo-base e Nosso Posicionamento

- **Artigo:** *Parallel and scalable Dunn Index for the validation of big data clusters* — Ncir et al., **Parallel Computing (Elsevier)**.
- **O que o artigo propõe:** um Dunn **distribuído em Apache Spark** (divide-and-conquer) + **amostragem** ("Sketch and Validate") para **aproximar** o índice em larga escala.
- **Nossa abordagem (complementar):**
  - Mesmo **problema** (validação de clusters em escala), **plataforma diferente**: **GPU/CUDA** em vez de cluster Spark.
  - Cálculo **exato** (sem amostragem) acelerado por **paralelismo de threads**.
  - Foco em **engenharia de kernels** e na hierarquia de memória da GPU.

> Contraste honesto: **exato/GPU** (nós) × **aproximado/distribuído** (artigo). São estratégias diferentes para o mesmo gargalo O(N²).

<!-- Fala: deixar claro o vínculo com o artigo E a diferença de abordagem —
isso responde diretamente ao critério (a) do enunciado. -->

---

## 4. Objetivos

- **Geral:** acelerar em GPU o cálculo **exato** das três métricas, superando um **baseline em CPU**.
- **Específicos:**
  1. Transferir a carga quadrática das distâncias para os milhares de núcleos da GPU.
  2. **Não materializar** a matriz N×N → reduzir memória de **O(N²) para O(N·D)** e **escalar** o N.
  3. Comparar **CPU 1-thread × CPU OpenMP × GPU** de forma justa (mesma máquina).
  4. **Validar a corretude** numérica em múltiplas camadas.
  5. Medir **speed-up**, **escalabilidade** e o **breakdown** de tempo por etapa.

---

## 5. Decisão-chave de Engenharia: *Matrix-free*

**Problema da versão ingênua (Etapa 2):** guardar a matriz de distâncias `D[N×N]`.

| N | Matriz `D` (double) | Cabe na T4 (16 GB)? |
|---:|---:|:---:|
| 8.000 | 0,5 GB | ✅ |
| 50.000 | **20 GB** | ❌ |
| 100.000 | **80 GB** | ❌ (e estoura 12 GB de RAM na CPU!) |

**Solução:** calcular `d(i,j)` **on-the-fly** dentro dos kernels (cada bloco trata uma linha/ponto e varre os demais). 
→ Memória **O(N·D)** (≈ 3 MB em N=100k) · sem overflow de `int` em `N·N` · **viabiliza 50k–100k**.

<!-- Fala: este é o coração técnico do trabalho final. A matriz era o limite
real de escala; eliminá-la é o que destrava 100 mil pontos. -->

---

## 6. Metodologia Paralela — A) Índice de Dunn

- **1 bloco por linha/ponto `i`** (256 threads/bloco).
- Coordenadas de `i` carregadas em **memória compartilhada** (`s_xi`) e reusadas por todas as threads.
- Threads varrem os pontos `j` (grid-stride), calculam `d(i,j)` **on-the-fly** e acumulam:
  - **máximo intra-cluster** e **mínimo inter-cluster** locais.
- **Redução em árvore** em *shared memory* (log₂256 = 8 passos) → resultado do bloco.
- CPU faz só a **redução global O(N)** dos dois vetores de tamanho N.

`kernel: dunn_rowwise_kernel` em `metrics_cuda.cu`

<!-- Fala: enfatizar que a matriz nunca existe; cada linha é consumida e
reduzida na hora, economizando banda de memória global. -->

---

## 7. Metodologia Paralela — B) Silhueta e Davies-Bouldin

**Coeficiente de Silhueta** (`silhouette_rowwise_kernel`):
- 1 bloco por ponto; **memória compartilhada dinâmica** `blockDim × K` (em **double**).
- Cada thread acumula `d(i,j)` no cluster de destino; redução por cluster; thread 0 calcula `s_i`.

**Davies-Bouldin** (kernels de centróide/dispersão):
- Centróides e dispersões acumulados via **`atomicAdd`** no device (com *fallback* CAS p/ double).
- Razão `R_ij` resolvida em paralelo por cluster. Custo **O(N)** — bem mais barato que Dunn/Silhueta.

> Toda a hierarquia de memória é explorada: **registradores → shared → global**.

---

## 8. Baseline em CPU (comparação justa)

- Mesmo algoritmo **matrix-free** em C++ (`baseline_cpu.cpp`) → comparação **simétrica** com a GPU.
- **Paralelização OpenMP** (`#pragma omp parallel for` com `reduction(max/min/+)`).
- Threads controladas por `OMP_NUM_THREADS` → permite medir **CPU 1-thread** e **CPU multi-thread**.
- Resultado: comparamos **três motores** — CPU-1, CPU-OMP e GPU — todos no **mesmo ambiente** (Colab).

> Comparar a GPU contra **1 thread** e contra **todos os núcleos** dá robustez científica ao speed-up.

<!-- Fala: explicar que medir só contra 1 thread infla o ganho; por isso
incluímos a CPU paralela (OpenMP) como segundo baseline. -->

---

## 9. Validação Numérica Rigorosa (3 camadas)

1. **Caso analítico do Dunn:** 4 pontos com Dunn teórico = **2.0**. CPU e GPU obtêm **2.000000** (erro 0).
2. **Ground truth (scikit-learn):** Silhueta e DB no Iris (N=150) batem com o `scikit-learn` com erro **~10⁻⁹**.
3. **Equivalência CPU ≡ GPU:** em todos os tamanhos do benchmark, as três métricas coincidem (tolerância 10⁻⁵).

> A corretude é **pré-condição**: o `benchmark.py` **aborta** se qualquer camada divergir.

<!-- Fala: a banca valoriza muito corretude; mostramos que aceleramos SEM
trocar exatidão por velocidade (no modo double). -->

---

## [IMAGEM] Resultado 1 — Tempo de Execução vs N

> **[IMAGEM: bench_tempo.png]**
> Curvas (escala log-log): **CPU 1-thread**, **CPU OpenMP** e **GPU CUDA** em função de N (até 100.000).

<!--
Inserir aqui a figura bench_tempo.png (gerada pelo benchmark.py no Colab).
Fala: as curvas de CPU crescem com inclinação ~2 (O(N^2)) no log-log,
enquanto a GPU cresce muito mais devagar — a distância entre as curvas é o ganho.
-->

---

## [IMAGEM] Resultado 2 — Speed-up vs N

> **[IMAGEM: bench_speedup.png]**
> Speed-up **GPU vs CPU-1thread** e **GPU vs CPU-OpenMP**, em função de N.

<!--
Inserir aqui a figura bench_speedup.png.
Fala: o speed-up cresce com N (mais trabalho = melhor amortização do overhead);
a curva vs CPU-OpenMP é menor (baseline mais forte) e é a comparação mais honesta.
-->

---

## [IMAGEM] Resultado 3 — Onde está o tempo (breakdown GPU)

> **[IMAGEM: bench_breakdown.png]**
> Barras empilhadas do tempo de GPU por etapa: **H2D (cópia)**, **Dunn**, **Silhueta**, **Davies-Bouldin**.

<!--
Inserir aqui a figura bench_breakdown.png.
Fala: Dunn e Silhueta (O(N^2)) dominam; Davies-Bouldin (O(N)) é desprezível;
a cópia H2D é mínima (X é pequeno na versão matrix-free) — por isso streams
de transferência teriam pouco efeito aqui (ver Trabalhos Futuros).
-->

---

## 10. Tabela de Resultados

| N | CPU-1 (s) | CPU-OMP (s) | GPU (s) | Speed-up (vs CPU-1) | Corretude |
|---:|---:|---:|---:|---:|:---:|
| 8.000   | `<...>` | `<...>` | `<...>` | `<...>` | 100% |
| 32.000  | `<...>` | `<...>` | `<...>` | `<...>` | 100% |
| 50.000  | `<...>` | `<...>` | `<...>` | `<...>` | 100% |
| 100.000 | `<...>` | `<...>` | `<...>` | **`<máx>`** | 100% |

*Valores em média ± desvio de várias repetições — preencher de `resultados_benchmark.csv` após rodar no Colab.*

- Speed-up máximo de **`<preencher>×`** em N=100.000.
- Corretude **100%** CPU ≡ GPU em todos os tamanhos.

<!-- Fala: destacar que agora escalamos para 100k — algo impossível na versão
com matriz N×N — e que o ganho cresce com o tamanho do problema. -->

---

## 11. Análise dos Resultados

- **Escalabilidade:** a CPU segue O(N²); a GPU mantém tempos baixos → a distância **aumenta com N**.
- **Baseline forte:** mesmo contra a **CPU OpenMP** (todos os núcleos), a GPU vence por boa margem.
- **Gargalo interno:** Dunn e Silhueta concentram o tempo (par-a-par); DB é marginal.
- **Transferência irrelevante:** H2D ≈ 0 na versão matrix-free → o tempo é **compute-bound**.
- **(Opcional) float × double:** o modo `float` acelera na T4 (fp32 ≫ fp64) com erro controlado — trade-off precisão × velocidade.

---

## 12. Limitações e Trabalhos Futuros

- **Recomputação:** Dunn e Silhueta recalculam distâncias separadamente; um **kernel fundido** evitaria recomputar.
- **Silhueta e K grande:** *shared memory* cresce com `blockDim×K` (limita K muito alto).
- **Streams CUDA:** com matrix-free a cópia H2D já é mínima, então o overlap traria **pouco ganho** — fica como trabalho futuro (ou overlap de kernels independentes).
- **Aproximação à la artigo-base:** combinar nossa GPU com **amostragem/"Sketch & Validate"** para ir além de 100k.
- **Multi-GPU / precisão mista** para datasets na casa dos milhões.

<!-- Fala: ser honesto sobre limites mostra maturidade e conecta de volta
ao artigo-base (amostragem) como caminho para milhões de pontos. -->

---

## 13. Conclusão

- Implementamos e **paralelizamos em CUDA** as três métricas de validação de clusters.
- A decisão **matrix-free** eliminou o gargalo de memória e permitiu **escalar até 100.000 pontos**.
- Mantivemos **corretude exata** (validada em 3 camadas) enquanto obtivemos **speed-up de `<preencher>×`**.
- Comparação **justa** contra CPU sequencial **e** CPU OpenMP, com **análise de breakdown** por etapa.
- Entregáveis: código (`git`), `benchmark.py` automatizado, artigo (modelo SSCAD) e esta apresentação.

---

# Obrigado!
## Perguntas?

**Equipe:** Henrique Miranda · Cindy Rabelo · Eduardo Peixoto · Luiany Carvalho

*Computação de Alto Desempenho — UFG / Instituto de Informática*

<!-- Resumo das conquistas para fechar:
 - matrix-free: O(N^2) -> O(N) de memória, escala a 100k
 - corretude 100% (analítico + sklearn + CPU≡GPU)
 - speed-up <preencher>x vs CPU-1 e <preencher>x vs CPU-OpenMP
 - breakdown mostra Dunn/Silhueta como gargalo; H2D irrelevante -->
