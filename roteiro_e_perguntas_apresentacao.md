# Roteiro de Apresentação + Banco de Perguntas e Respostas

> **Para que serve:** roteiro do que falar em **cada slide** da apresentação final + **10 perguntas e respostas** por slide para preparar a banca.
> **Tempo-alvo:** ~12–15 min de fala (≈ 1 min por slide de conteúdo).
> **Equipe (3):** Henrique M. M. Miranda · Cindy Stephanie Gomes Rabelo · Luiany Goncalves Carvalho.
> **Números-chave (decore):** speed-up **24,9×** (vs CPU-1) e **24,5×** (vs OpenMP) em N=100.000; GPU **3,24 s** vs CPU **80,6 s**; memória **O(N²) → O(N·D)** (80 GB → ~3 MB); corretude **100%** validada em 3 camadas.

---

## Slide 1 — Capa: Validação Paralela de Clusters em GPU

**🎤 Roteiro:**
> "Boa noite, professor Ricardo e colegas. Somos a equipe Henrique Miranda, Cindy Rabelo e Luiany Carvalho, e vamos apresentar nosso trabalho final de Computação de Alto Desempenho: a **aceleração em GPU, com CUDA**, do cálculo de três métricas clássicas de validação de agrupamentos — Dunn, Silhueta e Davies-Bouldin. A ideia central, que vão ver ao longo da apresentação, é uma estratégia **matrix-free** que nos permitiu escalar para 100 mil pontos mantendo o resultado **exato**, com speed-up de até quase 25 vezes sobre a CPU."

**❓ Perguntas e Respostas:**
1. **P:** Em uma frase, qual é o trabalho? **R:** Acelerar em GPU/CUDA o cálculo exato dos índices de Dunn, Silhueta e Davies-Bouldin, usando uma abordagem matrix-free que escala até 100 mil pontos.
2. **P:** O que significa "matrix-free"? **R:** Não materializar (não armazenar) a matriz de distâncias N×N; as distâncias são recalculadas sob demanda dentro dos kernels.
3. **P:** Por que usar GPU para esse problema? **R:** Porque o cálculo de distâncias é "paralelismo de dados massivo" — milhares de continhas iguais e independentes, caso ideal para os milhares de núcleos da GPU.
4. **P:** Qual o ganho headline? **R:** Speed-up de até 24,9× sobre a CPU sequencial e 24,5× sobre a CPU OpenMP, em N=100.000.
5. **P:** O ganho de velocidade custou exatidão? **R:** Não — a corretude é exata e foi validada em três camadas independentes.
6. **P:** Quais as três métricas e por que três? **R:** Dunn, Silhueta e Davies-Bouldin; usamos as três porque medem a qualidade do agrupamento de formas complementares.
7. **P:** Qual a disciplina e o contexto? **R:** Computação de Alto Desempenho (UFG), trabalho prático final, com baseline em CPU a ser superado.
8. **P:** Que linguagem/plataforma? **R:** CUDA C++ na GPU, C++ com OpenMP no baseline de CPU, e Python para automação/validação.
9. **P:** Qual hardware foi usado? **R:** Google Colab com GPU NVIDIA Tesla T4 (16 GB) e CPU de 2 vCPUs.
10. **P:** O código está disponível? **R:** Sim, publicamente no GitHub (repositório Ricktheus/Trabalho-final-Cuda), com benchmark reprodutível.

---

## Slide 2 — Contextualização: O Problema

**🎤 Roteiro:**
> "Algoritmos como K-Means e DBSCAN **agrupam** dados sem rótulos. Mas surge a pergunta: o agrupamento ficou **bom**? Para responder sem rótulos de referência usamos **métricas internas de validação**, que medem **compacidade** (o quão apertados são os grupos) e **separação** (o quão distantes estão entre si). O problema é o **custo**: Dunn e Silhueta exigem as distâncias euclidianas **par-a-par** entre todos os pontos, o que dá complexidade de tempo **O(N²·D)**. Ao **dobrar** N, o trabalho **quadruplica** — então, para dezenas ou centenas de milhares de pontos, a CPU sequencial se torna inviável. É exatamente esse gargalo quadrático que justifica trazer a GPU."

**❓ Perguntas e Respostas:**
1. **P:** O que é clustering? **R:** Particionar dados em grupos de itens parecidos, sem rótulos prévios (aprendizado não supervisionado).
2. **P:** Por que validar um agrupamento? **R:** Porque o algoritmo sempre entrega *algum* agrupamento; validar dá uma "nota" objetiva de qualidade (coesão e separação).
3. **P:** O que é uma métrica *interna*? **R:** A que avalia a qualidade usando só os próprios dados e a partição, sem rótulos externos de referência.
4. **P:** Por que o custo é O(N²)? **R:** Porque Dunn e Silhueta precisam da distância de cada ponto a todos os outros — cerca de N²/2 pares.
5. **P:** O que é o "D" em O(N²·D)? **R:** A dimensionalidade (nº de features); cada distância euclidiana custa O(D).
6. **P:** Por que "dobrar N quadruplica o trabalho"? **R:** Porque o custo é proporcional a N²; (2N)² = 4N².
7. **P:** As três métricas têm o mesmo gargalo? **R:** Não — Dunn e Silhueta são O(N²); Davies-Bouldin usa centróides e é O(N·D + K²D), bem mais barato.
8. **P:** Por que não basta uma métrica? **R:** Cada uma enxerga a qualidade de um ângulo (extremos, por-ponto, centróides); juntas são mais robustas.
9. **P:** Por que K-Means/DBSCAN não resolvem isso? **R:** Eles **formam** os grupos; validar é uma etapa posterior, independente do algoritmo de agrupamento.
10. **P:** Por que GPU e não só otimizar a CPU? **R:** O cálculo de distâncias é massivamente paralelo e independente — encaixe perfeito para milhares de threads de GPU, não para poucos núcleos de CPU.

---

## Slide 3 — Formulação Matemática: As Três Métricas

**🎤 Roteiro:**
> "Formalmente: o **Dunn** é a razão entre a **menor separação** entre dois clusters e o **maior diâmetro** interno; quanto **maior**, melhor. A **Silhueta** de cada ponto compara o quão bem ele se encaixa no próprio grupo (a) com o grupo vizinho mais próximo (b): `s = (b−a)/max(a,b)`, variando de −1 a +1, e a nota global é a média; **maior é melhor**. O **Davies-Bouldin** usa **centróides**: para cada par de clusters soma as dispersões internas e divide pela distância entre centros, pegando o pior vizinho de cada cluster e tirando a média; aqui **menor é melhor**. As duas primeiras dependem de distâncias par-a-par; o DB não."

**❓ Perguntas e Respostas:**
1. **P:** Defina o Índice de Dunn. **R:** min separação inter-cluster / max diâmetro intra-cluster; maior = grupos compactos e bem separados.
2. **P:** O que são `a(i)` e `b(i)` na Silhueta? **R:** `a(i)` é a distância média de i aos pontos do próprio cluster (coesão); `b(i)` é a menor distância média a um cluster vizinho (separação).
3. **P:** Qual o intervalo da Silhueta e o que significam os extremos? **R:** [−1, +1]; perto de +1 bem agrupado, ~0 na fronteira, negativo provavelmente no cluster errado.
4. **P:** O que mede `S_i` e `M_ij` no Davies-Bouldin? **R:** `S_i` é a dispersão interna do cluster i (distância média ao centróide); `M_ij` é a distância entre os centróides i e j.
5. **P:** Por que no Dunn "maior é melhor" e no DB "menor é melhor"? **R:** Dunn é separação/dispersão (quer numerador grande); DB é dispersão/separação (quer denominador grande → razão pequena).
6. **P:** Por que o Dunn é sensível a outliers? **R:** Ele usa mínimo e máximo (extremos); um único ponto fora do lugar muda o diâmetro ou a separação.
7. **P:** A Silhueta global é o quê? **R:** A média das silhuetas de todos os N pontos.
8. **P:** Qual métrica NÃO precisa de distâncias par-a-par? **R:** Davies-Bouldin — ela trabalha com centróides e dispersões.
9. **P:** Que distância vocês usam? **R:** Euclidiana, d(x_i,x_j) = raiz da soma dos quadrados das diferenças por dimensão.
10. **P:** O resultado das métricas depende da implementação (CPU/GPU)? **R:** Não deveria — e provamos que não: CPU e GPU dão valores idênticos (equivalência validada).

---

## Slide 4 — Posicionamento: Artigo-base

**🎤 Roteiro:**
> "Nosso trabalho parte do artigo *Parallel and scalable Dunn Index for the validation of big data clusters*, de Ncir e colegas, publicado na **Parallel Computing (Elsevier)**. Eles atacam o mesmo gargalo, mas com estratégia **diferente**: distribuem o cálculo em um cluster de máquinas com **Apache Spark** e usam **amostragem** — a técnica *Sketch and Validate* — para **aproximar** o índice em larga escala. Nossa abordagem é **complementar**: em vez de aproximar de forma distribuída, calculamos de forma **exata**, acelerando numa **única GPU**. O contraste é honesto: **exato/GPU** (nós) versus **aproximado/distribuído** (artigo) — duas respostas diferentes para o mesmo problema O(N²)."

**❓ Perguntas e Respostas:**
1. **P:** Qual é o artigo-base? **R:** Ncir, Hamza & Bouaguel, *Parallel and scalable Dunn Index...*, Parallel Computing, v.102, 2021.
2. **P:** Qual a estratégia do artigo? **R:** Dunn distribuído em Apache Spark (divide-and-conquer) + amostragem "Sketch and Validate" para aproximar o índice.
3. **P:** Como vocês diferem dele? **R:** Cálculo exato (sem amostragem) acelerado por GPU/CUDA numa única máquina, em vez de aproximado e distribuído.
4. **P:** Por que dizem "complementar" e não "melhor"? **R:** São trade-offs diferentes: exatidão numa GPU vs. escala distribuída aproximada; podem até ser combinados.
5. **P:** Quando a abordagem deles seria preferível? **R:** Para volumes na casa dos milhões/bilhões, onde uma única GPU não cabe e uma aproximação distribuída é aceitável.
6. **P:** E quando a de vocês é preferível? **R:** Quando se quer o valor **exato** do índice e os dados cabem no processamento de uma GPU (até ~100k+ no nosso caso).
7. **P:** Poderiam combinar as duas ideias? **R:** Sim — usar amostragem para reduzir o N e GPU para acelerar o cálculo exato de cada amostra; citamos isso como trabalho futuro.
8. **P:** O artigo calcula as três métricas? **R:** O foco dele é o Dunn; nós estendemos para Dunn, Silhueta e Davies-Bouldin.
9. **P:** O que é "Sketch and Validate"? **R:** Uma técnica de amostragem que estima o índice a partir de um subconjunto representativo, em vez de todos os pares.
10. **P:** Por que não reimplementaram o método do artigo para comparar? **R:** São plataformas diferentes (Spark distribuído vs. GPU); o enunciado pede superar um baseline de CPU, e o contraste conceitual já é o posicionamento científico.

---

## Slide 5 — Objetivos do Projeto

**🎤 Roteiro:**
> "Nosso objetivo geral é **acelerar em GPU o cálculo exato** das três métricas, superando um baseline em CPU. Os objetivos específicos são cinco: primeiro, **transferir a carga quadrática** das distâncias para os milhares de núcleos da GPU; segundo, **não materializar a matriz N×N**, reduzindo a memória de O(N²) para O(N·D) e escalando o N; terceiro, fazer uma **comparação justa** — CPU 1-thread, CPU OpenMP e GPU na mesma máquina; quarto, **validar a corretude** numérica em múltiplas camadas; e quinto, **medir** speed-up, escalabilidade e o breakdown de tempo por etapa."

**❓ Perguntas e Respostas:**
1. **P:** Qual é o objetivo geral? **R:** Acelerar em GPU o cálculo exato das três métricas, superando um baseline em CPU.
2. **P:** Por que "exato" é um objetivo explícito? **R:** Para diferenciar do artigo-base (aproximado) e garantir que a aceleração não troca velocidade por precisão.
3. **P:** O que significa "comparação justa"? **R:** CPU e GPU na mesma máquina, mesmos dados, e comparar a GPU também contra a CPU paralela (OpenMP), não só contra 1 thread.
4. **P:** Por que reduzir memória é um objetivo? **R:** Porque a matriz N×N (O(N²)) é o que impedia escalar; eliminá-la destrava N grande.
5. **P:** O que é "breakdown de tempo por etapa"? **R:** Decompor o tempo de GPU entre as etapas (H2D, Dunn, Silhueta, DB) para saber onde o tempo é gasto.
6. **P:** Qual hierarquia de memória vocês exploram? **R:** Registradores, memória compartilhada e memória global da GPU.
7. **P:** Qual o teto de N visado? **R:** Até 10⁵ (100.000) pontos.
8. **P:** Como garantem a corretude (objetivo 4)? **R:** Validação multicamada: caso analítico, scikit-learn e equivalência CPU≡GPU.
9. **P:** Por que medir escalabilidade? **R:** Para mostrar como o speed-up evolui com N e confirmar o comportamento O(N²) das curvas.
10. **P:** Superar a CPU era suficiente ou foram além? **R:** Fomos além: comparamos contra a melhor configuração de CPU disponível (OpenMP) e analisamos o porquê dos ganhos.

---

## Slide 6 — Decisão-chave: Matrix-free

**🎤 Roteiro:**
> "Esta é a **decisão técnica central**. A versão ingênua guardaria a matriz de distâncias N×N. Em precisão dupla, isso é N²×8 bytes: para 50 mil pontos já são **20 GB** — que **não cabem** nos 16 GB da T4 nem nos ~12 GB de RAM do Colab; para 100 mil, seriam **80 GB**, totalmente inviável, e ainda estouraria o limite de inteiro de 32 bits no índice N·N. Nossa solução é **matrix-free**: cada bloco trata um ponto e varre os demais, calculando `d(i,j)` **na hora**. Com isso a memória cai para **O(N·D)** — cerca de **3 MB** em 100 mil pontos. O preço é recalcular distâncias, mas é essa troca que **viabiliza a escala**."

**❓ Perguntas e Respostas:**
1. **P:** O que a versão ingênua faria de errado? **R:** Armazenaria a matriz N×N inteira (O(N²) de memória), o que não cabe para N grande.
2. **P:** Quanto ocuparia a matriz em N=100.000? **R:** 100.000² × 8 bytes = 80 GB.
3. **P:** Por que 50.000 já é problema? **R:** São 20 GB — excedem tanto os 16 GB da GPU quanto os ~12 GB de RAM do ambiente.
4. **P:** O que é o overflow de inteiro de 32 bits? **R:** N·N para N≥~46.341 ultrapassa 2³¹; um índice linear em `int` estouraria — no código usamos `size_t` para evitar.
5. **P:** Para quanto cai a memória com matrix-free? **R:** Para O(N·D) — só guardamos os pontos (N×D), ~3 MB em 100k com D=4 em double.
6. **P:** Qual é o "preço" da matrix-free? **R:** Recalcular distâncias sob demanda (algumas mais de uma vez), gastando mais FLOPs para economizar memória.
7. **P:** Por que essa troca vale a pena? **R:** A GPU é abundante em poder de cálculo, mas limitada em memória; trocar memória por recomputação é o que permite escalar.
8. **P:** A tabela é da memória real usada? **R:** Não — é a memória que a matriz *custaria* se fosse armazenada; na prática nunca a alocamos.
9. **P:** Matrix-free muda o resultado numérico? **R:** Não, o resultado é idêntico; muda só *como* as distâncias são obtidas (recomputadas vs. lidas de uma tabela).
10. **P:** O que de fato limita o N agora? **R:** Não é mais a memória, e sim o **tempo** da CPU de referência (para a comparação de speed-up).

---

## Slide 7 — Metodologia Paralela: Índice de Dunn

**🎤 Roteiro:**
> "No kernel do Dunn, `dunn_rowwise_kernel`, lançamos **um bloco por ponto** `i`, com **256 threads** por bloco. As coordenadas de `i` são carregadas em **memória compartilhada** e reutilizadas por todas as threads do bloco. As threads varrem os demais pontos `j` em **grid-stride**, calculam `d(i,j)` **on-the-fly** e acumulam, localmente, o **máximo intra-cluster** e o **mínimo inter-cluster**. Em seguida uma **redução em árvore** na memória compartilhada — `log₂256 = 8` passos sincronizados por `__syncthreads()` — consolida o resultado do bloco. A CPU faz apenas a **redução global final**, de custo O(N), sobre dois vetores de tamanho N — sem nunca transferir a matriz inteira."

**❓ Perguntas e Respostas:**
1. **P:** Qual o mapeamento de paralelismo do Dunn? **R:** Um bloco por ponto i, 256 threads por bloco varrendo os outros pontos j.
2. **P:** Por que carregar `i` em memória compartilhada? **R:** Para reusá-lo nas 256 threads sem cada uma buscar na memória global (lenta), economizando banda.
3. **P:** O que é grid-stride? **R:** Um laço em que cada thread processa índices j, j+blockDim, j+2·blockDim..., cobrindo todos os pontos mesmo com mais pontos que threads.
4. **P:** Por que `log₂256 = 8` passos? **R:** A redução em árvore combina 256 valores em 8 rodadas (256→128→...→1), em vez de 255 comparações sequenciais.
5. **P:** Para que serve `__syncthreads()`? **R:** Sincroniza as threads do bloco entre as rodadas da redução, garantindo que todas terminem antes da próxima.
6. **P:** O que cada bloco acumula? **R:** O máximo de distância intra-cluster (para o diâmetro) e o mínimo inter-cluster (para a separação), localmente.
7. **P:** Por que a CPU ainda faz uma redução final? **R:** Para combinar os N resultados parciais por bloco num único Dunn — é O(N), barato, e evita transferir a matriz.
8. **P:** Por que 256 threads (e não 1024)? **R:** É potência de 2 (boa para a redução), com bom equilíbrio entre ocupação e uso de memória compartilhada/registradores.
9. **P:** A matriz N×N aparece em algum momento? **R:** Não — cada distância é calculada, usada e descartada; só trafegam dois vetores de tamanho N.
10. **P:** Por que o Dunn ainda é O(N²) na GPU? **R:** Cada um dos N blocos varre N pontos; a GPU não reduz a complexidade, ela paraleliza o trabalho (faz muito ao mesmo tempo).

---

## Slide 8 — Metodologia Paralela: Silhueta e Davies-Bouldin

**🎤 Roteiro:**
> "Na **Silhueta**, kernel `silhouette_rowwise_kernel`, também usamos **um bloco por ponto**, mas com **memória compartilhada dinâmica** de tamanho `blockDim × K` (em double): cada thread acumula `d(i,j)` no cluster de destino, fazemos uma redução por cluster e a thread 0 calcula `a(i)`, `b(i)` e a silhueta local. No **Davies-Bouldin**, como ele não depende de distâncias par-a-par, usamos kernels de centróide e dispersão: acumulamos via `atomicAdd` no device — com *fallback* por compare-and-swap para double em GPUs antigas — e resolvemos a razão `R_ij` em paralelo por cluster. O custo é **O(N·D + K²D)**, linear em N, muito mais barato que Dunn e Silhueta. Exploramos toda a hierarquia: **registradores → compartilhada → global**."

**❓ Perguntas e Respostas:**
1. **P:** Por que a Silhueta usa shared memory `blockDim × K`? **R:** Para acumular, por cluster, a soma das distâncias do ponto i a cada cluster — daí o tamanho proporcional a K.
2. **P:** Como obtêm `a(i)` e `b(i)` no kernel? **R:** Após reduzir as somas por cluster, a thread 0 usa a soma do próprio cluster para `a(i)` e a menor média entre os outros clusters para `b(i)`.
3. **P:** Por que o DB usa centróides em vez de pares? **R:** Porque a definição do DB é baseada em dispersões em torno de centróides e distâncias entre centros — não exige todos os pares.
4. **P:** O que é `atomicAdd` e por que é preciso? **R:** Uma soma atômica ("um de cada vez") usada quando muitas threads escrevem no mesmo acumulador (centróide), evitando condição de corrida.
5. **P:** O que é o *fallback* por compare-and-swap? **R:** Em GPUs anteriores a sm_60, não há `atomicAdd` nativo para double; implementamos via `atomicCAS` em laço. A T4 (sm_75) já tem o nativo.
6. **P:** Qual o custo do Davies-Bouldin? **R:** O(N·D + K²D) — linear em N (com D e K pequenos, ≈ O(N)).
7. **P:** Por que o DB é "marginal" no tempo total? **R:** Por ser O(N) e não O(N²), seu tempo é desprezível frente a Dunn e Silhueta.
8. **P:** Que hierarquia de memória é explorada? **R:** Registradores (variáveis locais), memória compartilhada (acumuladores/redução) e memória global (os pontos).
9. **P:** A Silhueta tem alguma limitação de K? **R:** Sim — a shared memory cresce com `blockDim × K`, então K muito grande exigiria outra estratégia.
10. **P:** Por que acumular em double mesmo no modo float? **R:** Para preservar precisão nas somas/reduções, que são onde o erro de ponto flutuante se acumula.

---

## Slide 9 — Baseline em CPU: Comparação Justa

**🎤 Roteiro:**
> "Para uma comparação **justa**, o baseline de CPU implementa **o mesmo algoritmo matrix-free** em C++, garantindo simetria com a GPU. Os laços externos de Dunn e Silhueta são paralelizados com **OpenMP** (`#pragma omp parallel for` com cláusulas de redução), e o número de threads é controlado por `OMP_NUM_THREADS` — então, com o **mesmo binário**, medimos tanto a execução sequencial (1 thread) quanto a paralela. Comparamos **três motores**: CPU 1-thread, CPU OpenMP e GPU, todos na **mesma máquina** do Colab. Uma ressalva honesta: as 2 vCPUs do Colab dividem **um único núcleo físico**, então o OpenMP acelera pouco em N grande — voltaremos a isso na análise."

**❓ Perguntas e Respostas:**
1. **P:** Por que o baseline também é matrix-free? **R:** Para a comparação ser simétrica — mesma estratégia algorítmica na CPU e na GPU, isolando o efeito do hardware/paralelismo.
2. **P:** Como o OpenMP paraleliza? **R:** Com `#pragma omp parallel for` nos laços externos de Dunn e Silhueta, usando `reduction(max/min/+)` para evitar condições de corrida.
3. **P:** Por que comparar contra 1 thread E contra OpenMP? **R:** Comparar só contra 1 thread infla o ganho; incluir a CPU paralela mostra a GPU vs. a melhor CPU disponível.
4. **P:** As cláusulas de redução servem para quê? **R:** Para combinar com segurança os resultados parciais de cada thread (máximo, mínimo, soma) sem corrida de dados.
5. **P:** Por que CPU e GPU na mesma máquina? **R:** Para o speed-up ser justo — mesmo ambiente, mesmos dados, sem viés de hardware externo.
6. **P:** O OpenMP acelerou bastante? **R:** Não em N grande: as 2 vCPUs do Colab são hyperthreads de 1 núcleo físico, então o ganho é só ~1,02× sobre 1 thread.
7. **P:** Isso enfraquece o resultado? **R:** Não — significa que mesmo o **melhor** caso de CPU da plataforma fica ~24× atrás da GPU; só não medimos escalabilidade forte de OpenMP.
8. **P:** Como mediram sequencial e paralelo com um binário só? **R:** Variando `OMP_NUM_THREADS` (1 = sequencial; todos = paralelo) em tempo de execução.
9. **P:** Como seria o OpenMP numa CPU melhor? **R:** Com 8–16 núcleos físicos, o baseline OpenMP seria bem mais forte e o speed-up da GPU, proporcionalmente menor.
10. **P:** Por que não compararam contra uma CPU mais potente? **R:** O objetivo é um speed-up justo na **mesma** máquina; trocar a CPU quebraria a simetria do experimento.

---

## Slide 10 — Validação Numérica: 3 Camadas

**🎤 Roteiro:**
> "Tratamos a corretude como **pré-condição**: antes de medir desempenho, validamos em **três camadas independentes**. A primeira é um **caso analítico** do Dunn, com 4 pontos cujo valor teórico é exatamente **2,0** — CPU e GPU dão `2.000000`, erro zero. A segunda compara Silhueta e Davies-Bouldin contra o **scikit-learn** num conjunto `make_blobs(N=150)` — batem com erro da ordem de **10⁻⁹**, no limite da precisão de ponto flutuante. A terceira é a **equivalência CPU≡GPU** em todos os tamanhos. O `benchmark.py` **aborta** se qualquer camada divergir além de 10⁻⁵ — ou seja, o speed-up só é medido sobre resultados comprovadamente corretos."

**❓ Perguntas e Respostas:**
1. **P:** Quais são as três camadas? **R:** (1) caso analítico do Dunn, (2) ground truth com scikit-learn, (3) equivalência CPU≡GPU.
2. **P:** Por que um caso analítico para o Dunn? **R:** Porque o scikit-learn não implementa Dunn nativamente; criamos 4 pontos com resultado conhecido (2,0) para checar.
3. **P:** Por que make_blobs e não Iris? **R:** Usamos `make_blobs(n_samples=150)` — dados sintéticos controlados e reprodutíveis; é o que está no `benchmark.py`.
4. **P:** Qual o erro vs. scikit-learn? **R:** ~2,3×10⁻⁹ na Silhueta e ~3,2×10⁻⁹ no DB — no limite da precisão de double.
5. **P:** O que é a equivalência CPU≡GPU? **R:** Em todos os tamanhos do benchmark, GPU e CPU produzem os mesmos valores das três métricas (100% de match).
6. **P:** Qual a tolerância usada? **R:** 10⁻⁵; se qualquer camada divergir além disso, o benchmark aborta.
7. **P:** Por que validar antes de medir tempo? **R:** Para garantir que o ganho de velocidade não vem às custas de um resultado errado — velocidade sobre resultado incorreto não vale nada.
8. **P:** Por que o erro não é exatamente zero vs. scikit-learn? **R:** Porque ordens de soma diferentes em ponto flutuante geram resíduos minúsculos (~10⁻⁹), esperados e inofensivos.
9. **P:** O caso analítico deu erro zero mesmo? **R:** Sim — 2,000000 em CPU e GPU, porque é um valor exato e bem-condicionado.
10. **P:** Como sabem que não há "bug espelhado" (mesmo erro nos dois)? **R:** O caso analítico e o scikit-learn são referências **externas** independentes; concordar com elas descarta erro comum a CPU/GPU.

---

## Slide 11 — Resultado: Tempo de Execução vs N

**🎤 Roteiro:**
> "Este gráfico mostra o tempo das três estratégias em função de N, em escala log-log. As curvas de CPU — 1-thread e OpenMP — crescem com **inclinação ≈ 2**, confirmando o comportamento **O(N²)**, enquanto a GPU se mantém **muito mais baixa**. Em **N=100.000**, a CPU sequencial leva **80,6 s** e a OpenMP **79,2 s**, contra apenas **3,24 s** da GPU. A distância vertical entre as curvas é exatamente o speed-up. Repare que as curvas de CPU 1-thread e OpenMP ficam quase **coladas** — reflexo das 2 vCPUs num só núcleo físico."

**❓ Perguntas e Respostas:**
1. **P:** Por que escala log-log? **R:** Porque uma lei de potência O(N^k) vira uma **reta de inclinação k** em log-log, facilitando ver o expoente.
2. **P:** O que a inclinação ≈ 2 confirma? **R:** O comportamento quadrático O(N²) de Dunn e Silhueta.
3. **P:** A GPU também é O(N²)? **R:** Sim — a inclinação dela também é ~2; ela faz o mesmo trabalho, só muito mais rápido (curva deslocada para baixo).
4. **P:** Quais os tempos em N=100.000? **R:** CPU-1: 80,6 s; CPU-OpenMP: 79,2 s; GPU: 3,24 s.
5. **P:** Por que as curvas de CPU-1 e OpenMP quase coincidem? **R:** Porque o OpenMP quase não acelera (2 vCPUs = 1 núcleo físico), então os tempos ficam parecidos.
6. **P:** O que representa a distância vertical entre curvas? **R:** O speed-up (razão de tempos) para aquele N.
7. **P:** Por que a vantagem da GPU cresce com N? **R:** Em N pequeno o overhead fixo domina; quando o trabalho aumenta, esse custo é amortizado e o paralelismo brilha.
8. **P:** O tempo medido inclui transferências e leitura de dados? **R:** O total inclui a leitura do dataset no host; as transferências H2D são mínimas no matrix-free.
9. **P:** Como garantem que os tempos são estáveis? **R:** Cada ponto é a média de repetições (5 até 8k, 3 até 32k, 2 nos maiores); o desvio acompanha o código.
10. **P:** Em N pequeno a GPU compensa? **R:** Pouco — em N=250 o ganho é só ~1,6×, pois o overhead domina; a GPU compensa em N grande.

---

## Slide 12 — Resultado: Speed-up vs N

**🎤 Roteiro:**
> "Aqui isolamos o **speed-up** em função de N. Ele **cresce monotonicamente** com o tamanho do problema: começa modesto e atinge o **máximo de 24,9×** sobre a CPU 1-thread e **24,5×** sobre a CPU OpenMP em **N=100.000**. A explicação é a amortização do custo fixo: em N pequeno, inicialização e transferências dominam; conforme o trabalho aumenta, esse overhead vira ruído e o paralelismo massivo da GPU domina. Há uma pequena anomalia em N=250 na curva do OpenMP, causada pelo overhead de criação de threads em problemas minúsculos."

**❓ Perguntas e Respostas:**
1. **P:** Qual o speed-up máximo? **R:** 24,9× vs. CPU-1thread e 24,5× vs. CPU-OpenMP, em N=100.000.
2. **P:** Por que o speed-up cresce com N? **R:** O custo fixo (overhead) é amortizado quando há mais trabalho; a fração útil paralelizável aumenta.
3. **P:** Por que as duas curvas de speed-up são tão próximas? **R:** Porque CPU-1 e CPU-OpenMP têm tempos parecidos (OpenMP acelera pouco), então a GPU tem ganho semelhante sobre ambas.
4. **P:** O que causa a anomalia em N=250 (OpenMP)? **R:** O overhead de criar/gerenciar threads supera o trabalho útil em problemas muito pequenos, deixando o OpenMP mais lento que o sequencial ali.
5. **P:** O speed-up é "real" se o OpenMP é fraco? **R:** Sim — a GPU também vence a CPU sequencial por ~24,9×; o OpenMP só mostra que mesmo a melhor CPU disponível fica para trás.
6. **P:** O speed-up satura? **R:** Até 100k ainda cresce; saturaria quando a GPU ficasse limitada por seus próprios recursos, o que não ocorre nessa faixa.
7. **P:** Por que não passar de 100k para mostrar speed-up maior? **R:** O limite é o tempo da CPU de referência (80 s já em 100k); medir mais seria caro, não impossível.
8. **P:** Esse ganho é otimista ou conservador? **R:** Conservador — a T4 é fraca em double; uma GPU melhor (A100/H100) daria speed-up bem maior.
9. **P:** O que é "speed-up" exatamente? **R:** Tempo da CPU dividido pelo tempo da GPU para o mesmo N e mesmos dados.
10. **P:** O speed-up seria o mesmo em float? **R:** Não — em float a T4 é muito mais rápida, então o speed-up tende a ser ainda maior (com erro controlado).

---

## Slide 13 — Resultado: Breakdown de Tempo da GPU

**🎤 Roteiro:**
> "Agora **onde** o tempo da GPU é gasto. Decompondo as quatro etapas instrumentadas — H2D, Dunn, Silhueta e Davies-Bouldin — vemos que **Dunn e Silhueta concentram ~99,9% do tempo de kernel** (as duas métricas par-a-par, O(N²)). A cópia Host-to-Device e o Davies-Bouldin são **desprezíveis**, abaixo de um milissegundo. Isso tem uma implicação prática direta: como as transferências são mínimas na formulação matrix-free, técnicas de sobreposição de cópia e cálculo, como CUDA streams, trariam ganho irrelevante aqui — o problema é **compute-bound**, não memory-bound."

**❓ Perguntas e Respostas:**
1. **P:** Quais são as quatro etapas do breakdown? **R:** H2D (cópia host→device), Dunn, Silhueta e Davies-Bouldin.
2. **P:** Quem domina o tempo? **R:** Dunn e Silhueta, com ~99,9% — são as métricas par-a-par O(N²).
3. **P:** Os 99,9% são de quê exatamente? **R:** Das **4 etapas medidas** (tempo de kernel), não do tempo total da tabela.
4. **P:** Então a soma das etapas bate com o "GPU (s)" da Tabela 2? **R:** Quase: as 4 etapas somam ~3,11 s e o total é 3,24 s; a diferença (~0,12 s) é a **leitura do dataset no host**, que não é etapa de GPU.
5. **P:** Por que H2D é desprezível? **R:** Porque, sendo matrix-free, só copiamos os pontos (O(N·D), ~3 MB), não uma matriz de 80 GB.
6. **P:** O que significa "compute-bound"? **R:** O gargalo é o **cálculo** (as distâncias), não a transferência de dados nem a memória.
7. **P:** Por que CUDA streams não ajudariam? **R:** Streams sobrepõem cópia e cálculo; como a cópia já é ~0, não há o que sobrepor — ganho irrelevante.
8. **P:** Por que o Davies-Bouldin é tão barato? **R:** É O(N), baseado em centróides, sem o custo quadrático dos pares.
9. **P:** Como mediram o tempo de cada kernel? **R:** Com `cudaEvent` em torno de cada kernel (e `chrono` para H2D/leitura), de forma instrumentada.
10. **P:** Qual otimização faria sentido então? **R:** Reduzir o cálculo em si — por exemplo, um **kernel fundido** que calcula a distância uma vez e alimenta Dunn e Silhueta juntos.

---

## Slide 14 — Tabela de Resultados

**🎤 Roteiro:**
> "Esta tabela consolida tudo. Para cada N, mostramos os tempos de CPU-1, CPU-OpenMP e GPU, o speed-up e a corretude. O destaque é a última linha: em **N=100.000**, a GPU faz em **3,24 s** o que a CPU faz em ~80 s — **24,9×** mais rápido (24,5× vs OpenMP). A coluna de corretude é **100% em todos os tamanhos**: a GPU é sempre idêntica à CPU. Os tempos são médias de repetições — 5 até 8 mil, 3 até 32 mil, 2 nos maiores — e o erro contra o scikit-learn fica em ~10⁻⁹. Ou seja, escalamos para 100 mil pontos, algo **impossível** na versão com matriz, e o ganho **cresce** com o tamanho do problema."

**❓ Perguntas e Respostas:**
1. **P:** Qual a linha mais importante? **R:** N=100.000: GPU 3,24 s vs CPU 80,6 s → 24,9× (e 24,5× vs OpenMP), corretude 100%.
2. **P:** O que significa "Corretude 100%"? **R:** Em todos os N testados, as três métricas da GPU coincidem com as da CPU (e com as referências externas).
3. **P:** Como os tempos foram agregados? **R:** Média de repetições: 5 (N≤8k), 3 (até 32k) e 2 (50k e 100k).
4. **P:** Onde está o desvio-padrão? **R:** Não é exibido na tabela por legibilidade, mas é calculado e acompanha o código-fonte (`resultados_benchmark.csv`).
5. **P:** Por que o speed-up vs OpenMP é quase igual ao vs 1-thread? **R:** Porque o OpenMP, nas 2 vCPUs do Colab (1 núcleo físico), quase não acelera (~1,02×).
6. **P:** A versão antiga (com matriz) chegaria a 100k? **R:** Não — precisaria de 80 GB; a matrix-free é o que viabiliza essa escala.
7. **P:** O ganho é constante ao longo de N? **R:** Não — cresce: de ~16× em 8k para ~25× em 100k.
8. **P:** Qual o erro vs. scikit-learn citado no rodapé? **R:** ~10⁻⁹, no limite da precisão de ponto flutuante em double.
9. **P:** Esses números são em double ou float? **R:** Em double (modo padrão, validado); o float seria ainda mais rápido na T4.
10. **P:** Os dados de teste são quais? **R:** Sintéticos via `make_blobs` (D=4, K=5, semente fixa) para garantir reprodutibilidade.

---

## Slide 15 — Análise dos Resultados

**🎤 Roteiro:**
> "Para fechar a análise, cinco pontos. **Escalabilidade:** a CPU segue O(N²) e a GPU mantém tempos baixos, então a vantagem **aumenta com N**. **Baseline honesto:** as 2 vCPUs do Colab dividem 1 núcleo físico, então o OpenMP quase não acelera (~1,02×) — ainda assim, mesmo o melhor caso de CPU fica ~24× atrás da GPU, então o ganho **não é trivial**. **Gargalo interno:** Dunn e Silhueta concentram o tempo; Davies-Bouldin é marginal. **Transferência irrelevante:** o problema é compute-bound, e a decisão matrix-free foi acertada. E o **float × double:** o modo float acelera ainda mais na T4 com erro controlado, disponível como opção — um trade-off precisão × velocidade."

**❓ Perguntas e Respostas:**
1. **P:** Resuma a conclusão sobre escalabilidade. **R:** Como CPU é O(N²) e GPU cresce muito mais devagar, a vantagem da GPU aumenta com N.
2. **P:** Por que chamam o baseline de "honesto"? **R:** Porque admitimos que o OpenMP do Colab quase não paraleliza (2 vCPUs/1 núcleo), em vez de vender um ganho inflado.
3. **P:** Se o OpenMP é fraco, o resultado vale? **R:** Sim — a GPU vence inclusive a CPU sequencial por 24,9×; o ponto robusto é bater o melhor caso de CPU disponível.
4. **P:** Por que o problema é compute-bound? **R:** Porque ~99,9% do tempo está no cálculo (Dunn/Silhueta) e a transferência H2D é ~0 no matrix-free.
5. **P:** A decisão matrix-free se justificou? **R:** Sim — eliminou o gargalo de memória, viabilizou 100k e tornou a transferência irrelevante (compute-bound).
6. **P:** Qual o trade-off do float? **R:** Mais velocidade na T4 (fp32 ≫ fp64) por um pouco de precisão; aceitável para validar clusters, e oferecido como opção.
7. **P:** Por que os ganhos são "conservadores"? **R:** A T4 é fraca em double; numa GPU melhor (A100/H100) o speed-up seria substancialmente maior.
8. **P:** Qual a principal limitação reconhecida? **R:** A recomputação de distâncias (Dunn e Silhueta separadamente) e o teto de ~100k imposto pelo tempo da CPU.
9. **P:** Quais os trabalhos futuros? **R:** Kernel fundido, múltiplas GPUs/precisão mista para milhões, combinação com amostragem do artigo-base, e datasets reais.
10. **P:** Qual a mensagem final de uma frase? **R:** Aceleramos de forma exata as três métricas em GPU, escalando para 100 mil pontos com até ~25× de ganho — sem abrir mão da corretude.

---

## 🎯 Perguntas gerais de fechamento (bônus)

1. **P:** Se tivessem que destacar UMA contribuição, qual seria? **R:** A estratégia matrix-free, que eliminou o gargalo O(N²) de memória e destravou a escala de 100 mil pontos.
2. **P:** O que vocês fariam diferente com mais tempo? **R:** Um kernel fundido (uma distância alimentando Dunn e Silhueta), multi-GPU, e testes em datasets reais.
3. **P:** Por que não usaram uma biblioteca pronta (cuML/cuBLAS)? **R:** O objetivo didático é **implementar e otimizar os kernels** nós mesmos, explorando a hierarquia de memória — e calcular Dunn, que não é padrão nessas libs.
4. **P:** O método funciona para clusters de formato irregular? **R:** A corretude sim (as fórmulas independem do formato); só não testamos com dados reais/irregulares — fica como limitação.
5. **P:** Como garantem reprodutibilidade? **R:** Dados sintéticos com semente fixa, código público, e um `benchmark.py` que compila, valida e mede automaticamente.
6. **P:** Qual o papel do Python no trabalho? **R:** Orquestração: gerar dados, compilar CPU/GPU, validar contra scikit-learn, medir tempos e plotar — não está no caminho crítico de desempenho.
7. **P:** O speed-up incluiria a geração dos dados? **R:** Não — medimos o cálculo das métricas; a geração/leitura dos dados é contabilizada à parte (e some no breakdown de GPU).
8. **P:** É possível ir além de 100 mil pontos? **R:** Sim, em memória (matrix-free), mas o tempo da CPU de referência cresce muito; para milhões usaríamos multi-GPU ou amostragem.
9. **P:** Por que três métricas e não só o Dunn do artigo? **R:** Para um panorama mais completo de validação e para mostrar que a abordagem em GPU generaliza para diferentes padrões de acesso.
10. **P:** Qual a relevância prática disso? **R:** Permite validar agrupamentos em datasets grandes em segundos em vez de minutos, tornando viável testar muitas partições/parâmetros.
