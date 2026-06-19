// ======================================================================
//  metrics_cuda.cu  -  Versao MATRIX-FREE (sem materializar a matriz NxN)
// ----------------------------------------------------------------------
//  Calculo paralelo em GPU (CUDA) de tres metricas de validacao de
//  clusters: Indice de Dunn, Coeficiente de Silhueta e Davies-Bouldin.
//
//  Diferenca para a versao anterior:
//    - NAO alocamos mais a matriz de distancias D[N*N] na VRAM.
//    - As distancias par-a-par sao recalculadas "on-the-fly" dentro dos
//      kernels de Dunn e Silhueta (cada bloco trata uma linha/ponto i e
//      varre os demais pontos j calculando d(i,j) a partir de X).
//    - Memoria passa de O(N^2) para O(N*D): viabiliza N=50.000/100.000.
//    - Remove o overflow de int em (N*N) que ocorria para N grande.
//
//  Precisao (trade-off velocidade x exatidao):
//    - Compile com  -DUSE_FLOAT  para usar float nas coordenadas e no
//      calculo das distancias (mais rapido na T4, que e fraca em fp64).
//    - Sem a flag, usa double (padrao, validado contra a CPU/sklearn).
//    - As reducoes/somatorios acumulam sempre em double para preservar
//      a exatidao mesmo no modo float.
// ======================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <chrono>
#include <set>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------
// Tipo de precisao para coordenadas e distancias (float opcional)
// ----------------------------------------------------------------------
#ifdef USE_FLOAT
  typedef float real_t;
  #define DEV_SQRT(x) sqrtf(x)
  #define PRECISION_LABEL 0
#else
  typedef double real_t;
  #define DEV_SQRT(x) sqrt(x)
  #define PRECISION_LABEL 1
#endif

#define MAX_D 128          // dimensao maxima suportada (cabe em shared estatica)
#define BLOCK_SIZE 256     // threads por bloco (potencia de 2 p/ reducao)

// Estrutura para armazenar o dataset na CPU
struct Dataset {
    int N;
    int D;
    int K;
    std::vector<real_t> X;
    std::vector<int> labels;
    std::vector<int> unique_labels;
};

// Carrega o dataset (mesmo formato do baseline CPU: "N D K" + N linhas)
Dataset load_dataset(const std::string& filepath) {
    Dataset ds;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo: " << filepath << std::endl;
        std::exit(1);
    }

    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        ss >> ds.N >> ds.D >> ds.K;
    }

    ds.X.resize((size_t)ds.N * ds.D);
    ds.labels.resize(ds.N);

    std::set<int> label_set;
    for (int i = 0; i < ds.N; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Erro de leitura na linha " << i + 2 << std::endl;
            std::exit(1);
        }
        std::stringstream ss(line);
        double v;
        for (int d = 0; d < ds.D; ++d) {
            ss >> v;
            ds.X[(size_t)i * ds.D + d] = (real_t)v;
        }
        ss >> ds.labels[i];
        label_set.insert(ds.labels[i]);
    }

    ds.unique_labels.assign(label_set.begin(), label_set.end());
    for (int i = 0; i < ds.N; ++i) {
        auto it = std::find(ds.unique_labels.begin(), ds.unique_labels.end(), ds.labels[i]);
        ds.labels[i] = std::distance(ds.unique_labels.begin(), it);
    }

    return ds;
}

// Macro para tratar erros de CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(1); \
        } \
    } while (0)

// ----------------------------------------------------------------------
// Suporte a atomicAdd para double em arquiteturas antigas (< sm_60)
// ----------------------------------------------------------------------
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
#if defined(__CUDA_ARCH__)
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
#endif
#endif

// Distancia euclidiana on-the-fly entre o ponto i (em shared) e o ponto j (global).
// O somatorio e feito em real_t (rapido em float); o resultado retorna em double.
__device__ __forceinline__ double dev_dist(const real_t* xi, const real_t* X, int j, int D) {
    real_t sum = (real_t)0;
    const real_t* xj = X + (size_t)j * D;
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        real_t diff = xi[d] - xj[d];
        sum += diff * diff;
    }
    return (double)DEV_SQRT(sum);
}

// ----------------------------------------------------------------------
// KERNELS CUDA (MATRIX-FREE)
// ----------------------------------------------------------------------

// 1. Indice de Dunn — um bloco por linha i; distancias calculadas on-the-fly.
//    Cada bloco encontra o maior intra-cluster e o menor inter-cluster da linha.
__global__ void dunn_rowwise_kernel(const real_t* X, const int* labels,
                                    double* row_max_intra, double* row_min_inter,
                                    int N, int D) {
    int i = blockIdx.x;            // um bloco por ponto/linha
    int tx = threadIdx.x;

    __shared__ real_t s_xi[MAX_D]; // coordenadas do ponto i (reuso por todas as threads)
    __shared__ double s_max[BLOCK_SIZE];
    __shared__ double s_min[BLOCK_SIZE];

    for (int d = tx; d < D; d += blockDim.x) s_xi[d] = X[(size_t)i * D + d];
    __syncthreads();

    double local_max = 0.0;
    double local_min = 1e18;       // "infinito"
    int own_label = labels[i];

    for (int j = tx; j < N; j += blockDim.x) {
        if (j == i) continue;
        double dist = dev_dist(s_xi, X, j, D);
        if (labels[j] == own_label) {
            if (dist > local_max) local_max = dist;
        } else {
            if (dist < local_min) local_min = dist;
        }
    }

    s_max[tx] = local_max;
    s_min[tx] = local_min;
    __syncthreads();

    // Reducao em arvore em memoria compartilhada
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            if (s_max[tx + stride] > s_max[tx]) s_max[tx] = s_max[tx + stride];
            if (s_min[tx + stride] < s_min[tx]) s_min[tx] = s_min[tx + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        row_max_intra[i] = s_max[0];
        row_min_inter[i] = s_min[0];
    }
}

// 2. Coeficiente de Silhueta — um bloco por ponto i; distancias on-the-fly.
//    shared_sums (dinamica) acumula em double a soma de d(i,j) por cluster.
__global__ void silhouette_rowwise_kernel(const real_t* X, const int* labels,
                                          const int* cluster_sizes, double* s,
                                          int N, int K, int D) {
    int i = blockIdx.x;
    int tx = threadIdx.x;

    extern __shared__ double shared_sums[]; // tamanho: blockDim.x * K
    __shared__ real_t s_xi[MAX_D];

    for (int d = tx; d < D; d += blockDim.x) s_xi[d] = X[(size_t)i * D + d];
    for (int c = 0; c < K; ++c) shared_sums[tx * K + c] = 0.0;
    __syncthreads();

    int own_cluster = labels[i];

    for (int j = tx; j < N; j += blockDim.x) {
        if (j == i) continue;
        int c_j = labels[j];
        if (c_j >= 0 && c_j < K) {
            shared_sums[tx * K + c_j] += dev_dist(s_xi, X, j, D);
        }
    }
    __syncthreads();

    // Reducao entre as threads do bloco para cada um dos K clusters
    if (tx < K) {
        double total_sum = 0.0;
        for (int t = 0; t < blockDim.x; ++t) {
            total_sum += shared_sums[t * K + tx];
        }
        shared_sums[tx] = total_sum; // reutiliza as K primeiras posicoes
    }
    __syncthreads();

    if (tx == 0) {
        if (cluster_sizes[own_cluster] <= 1) {
            s[i] = 0.0;
        } else {
            double a = shared_sums[own_cluster] / (cluster_sizes[own_cluster] - 1);
            double b = 1e18;
            for (int c = 0; c < K; ++c) {
                if (c == own_cluster) continue;
                if (cluster_sizes[c] == 0) continue;
                double avg_dist = shared_sums[c] / cluster_sizes[c];
                if (avg_dist < b) b = avg_dist;
            }
            s[i] = (b - a) / fmax(a, b);
        }
    }
}

// 3. Davies-Bouldin (ja era matrix-free: baseado em centroides)
__global__ void compute_centroids_kernel(const real_t* X, const int* labels,
                                         double* centroids, int* cluster_sizes, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int c = labels[i];
        for (int d = 0; d < D; ++d) {
            atomicAdd(&centroids[c * D + d], (double)X[(size_t)i * D + d]);
        }
        atomicAdd(&cluster_sizes[c], 1);
    }
}

__global__ void divide_centroids_kernel(double* centroids, const int* cluster_sizes, int K, int D) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K) {
        int size = cluster_sizes[c];
        if (size > 0) {
            for (int d = 0; d < D; ++d) centroids[c * D + d] /= size;
        }
    }
}

__global__ void compute_dispersion_kernel(const real_t* X, const int* labels,
                                          const double* centroids, double* S, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int c = labels[i];
        double sum_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = (double)X[(size_t)i * D + d] - centroids[c * D + d];
            sum_sq += diff * diff;
        }
        atomicAdd(&S[c], sqrt(sum_sq));
    }
}

__global__ void divide_dispersion_kernel(double* S, const int* cluster_sizes, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K) {
        int size = cluster_sizes[c];
        if (size > 0) S[c] /= size;
    }
}

__global__ void compute_db_kernel(const double* S, const double* centroids,
                                  double* db_ratios, int K, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) {
        double max_ratio = 0.0;
        for (int j = 0; j < K; ++j) {
            if (i == j) continue;
            double sum_sq = 0.0;
            for (int d = 0; d < D; ++d) {
                double diff = centroids[i * D + d] - centroids[j * D + d];
                sum_sq += diff * diff;
            }
            double M_ij = sqrt(sum_sq);
            if (M_ij > 0.0) {
                double ratio = (S[i] + S[j]) / M_ij;
                if (ratio > max_ratio) max_ratio = ratio;
            }
        }
        db_ratios[i] = max_ratio;
    }
}

// ----------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <caminho_do_dataset.csv> [apenas_dunn: 0 ou 1]" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    bool run_all = true;
    if (argc >= 3) run_all = (std::stoi(argv[2]) == 0);

    // 1. Carrega o dataset na CPU
    auto t_load_start = std::chrono::high_resolution_clock::now();
    Dataset ds = load_dataset(filepath);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double cpu_load_time = std::chrono::duration<double>(t_load_end - t_load_start).count();

    if (ds.D > MAX_D) {
        std::cerr << "Erro: D=" << ds.D << " excede MAX_D=" << MAX_D << std::endl;
        return 1;
    }

    // 2. Alocacao e copia Host -> Device (apenas X e labels: O(N*D), nao mais O(N^2))
    real_t* d_X;
    int* d_labels;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)ds.N * ds.D * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)ds.N * sizeof(int)));

    auto t_h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_X, ds.X.data(), (size_t)ds.N * ds.D * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, ds.labels.data(), (size_t)ds.N * sizeof(int), cudaMemcpyHostToDevice));
    auto t_h2d_end = std::chrono::high_resolution_clock::now();
    double h2d_time = std::chrono::duration<double>(t_h2d_end - t_h2d_start).count();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ------------------------------------------------------------------
    // A. Indice de Dunn (distancias on-the-fly)
    // ------------------------------------------------------------------
    double* d_row_max;
    double* d_row_min;
    CUDA_CHECK(cudaMalloc(&d_row_max, (size_t)ds.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_row_min, (size_t)ds.N * sizeof(double)));

    CUDA_CHECK(cudaEventRecord(start));
    dunn_rowwise_kernel<<<ds.N, BLOCK_SIZE>>>(d_X, d_labels, d_row_max, d_row_min, ds.N, ds.D);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float dunn_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&dunn_ms, start, stop));

    std::vector<double> h_row_max(ds.N), h_row_min(ds.N);
    CUDA_CHECK(cudaMemcpy(h_row_max.data(), d_row_max, ds.N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row_min.data(), d_row_min, ds.N * sizeof(double), cudaMemcpyDeviceToHost));

    double max_intra = 0.0;
    double min_inter = std::numeric_limits<double>::infinity();
    for (int i = 0; i < ds.N; ++i) {
        if (h_row_max[i] > max_intra) max_intra = h_row_max[i];
        if (h_row_min[i] < min_inter) min_inter = h_row_min[i];
    }
    double dunn_score = (max_intra == 0.0) ? 0.0 : (min_inter / max_intra);
    double dunn_time = dunn_ms / 1000.0;

    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_min));

    // ------------------------------------------------------------------
    // B. Coeficiente de Silhueta (distancias on-the-fly)
    // ------------------------------------------------------------------
    double sil_score = -2.0;
    double sil_time = 0.0;

    if (run_all) {
        std::vector<int> h_cluster_sizes(ds.K, 0);
        for (int i = 0; i < ds.N; ++i) h_cluster_sizes[ds.labels[i]]++;

        int* d_cluster_sizes;
        double* d_s;
        CUDA_CHECK(cudaMalloc(&d_cluster_sizes, ds.K * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_s, (size_t)ds.N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_cluster_sizes, h_cluster_sizes.data(), ds.K * sizeof(int), cudaMemcpyHostToDevice));

        size_t shared_mem_size = (size_t)BLOCK_SIZE * ds.K * sizeof(double);

        CUDA_CHECK(cudaEventRecord(start));
        silhouette_rowwise_kernel<<<ds.N, BLOCK_SIZE, shared_mem_size>>>(d_X, d_labels, d_cluster_sizes, d_s, ds.N, ds.K, ds.D);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float sil_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&sil_ms, start, stop));

        std::vector<double> h_s(ds.N);
        CUDA_CHECK(cudaMemcpy(h_s.data(), d_s, ds.N * sizeof(double), cudaMemcpyDeviceToHost));

        double sil_sum = 0.0;
        for (int i = 0; i < ds.N; ++i) sil_sum += h_s[i];
        sil_score = sil_sum / ds.N;
        sil_time = sil_ms / 1000.0;

        CUDA_CHECK(cudaFree(d_cluster_sizes));
        CUDA_CHECK(cudaFree(d_s));
    }

    // ------------------------------------------------------------------
    // C. Davies-Bouldin
    // ------------------------------------------------------------------
    double db_score = -1.0;
    double db_time = 0.0;

    if (run_all) {
        double* d_centroids;
        int* d_db_cluster_sizes;
        double* d_S;
        double* d_db_ratios;

        CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)ds.K * ds.D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_db_cluster_sizes, ds.K * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_S, ds.K * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_db_ratios, ds.K * sizeof(double)));

        CUDA_CHECK(cudaMemset(d_centroids, 0, (size_t)ds.K * ds.D * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_db_cluster_sizes, 0, ds.K * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_S, 0, ds.K * sizeof(double)));

        int blocksForPoints = (ds.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blocksForClusters = (ds.K + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CUDA_CHECK(cudaEventRecord(start));
        compute_centroids_kernel<<<blocksForPoints, BLOCK_SIZE>>>(d_X, d_labels, d_centroids, d_db_cluster_sizes, ds.N, ds.D);
        divide_centroids_kernel<<<blocksForClusters, BLOCK_SIZE>>>(d_centroids, d_db_cluster_sizes, ds.K, ds.D);
        compute_dispersion_kernel<<<blocksForPoints, BLOCK_SIZE>>>(d_X, d_labels, d_centroids, d_S, ds.N, ds.D);
        divide_dispersion_kernel<<<blocksForClusters, BLOCK_SIZE>>>(d_S, d_db_cluster_sizes, ds.K);
        compute_db_kernel<<<blocksForClusters, BLOCK_SIZE>>>(d_S, d_centroids, d_db_ratios, ds.K, ds.D);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float db_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&db_ms, start, stop));

        std::vector<double> h_db_ratios(ds.K);
        CUDA_CHECK(cudaMemcpy(h_db_ratios.data(), d_db_ratios, ds.K * sizeof(double), cudaMemcpyDeviceToHost));

        double db_sum = 0.0;
        for (int i = 0; i < ds.K; ++i) db_sum += h_db_ratios[i];
        db_score = db_sum / ds.K;
        db_time = db_ms / 1000.0;

        CUDA_CHECK(cudaFree(d_centroids));
        CUDA_CHECK(cudaFree(d_db_cluster_sizes));
        CUDA_CHECK(cudaFree(d_S));
        CUDA_CHECK(cudaFree(d_db_ratios));
    }

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Saida formatada para o script Python
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "--- RESULTS ---" << std::endl;
    std::cout << "N: " << ds.N << std::endl;
    std::cout << "D: " << ds.D << std::endl;
    std::cout << "K: " << ds.K << std::endl;
    std::cout << "Precision: " << PRECISION_LABEL << std::endl;  // 1=double, 0=float
    std::cout << "Dunn: " << dunn_score << std::endl;
    std::cout << "Silhouette: " << sil_score << std::endl;
    std::cout << "DB: " << db_score << std::endl;
    std::cout << "Time_Load: " << cpu_load_time << std::endl;
    std::cout << "Time_H2D: " << h2d_time << std::endl;
    std::cout << "Time_Dunn: " << dunn_time << std::endl;
    std::cout << "Time_Silhouette: " << sil_time << std::endl;
    std::cout << "Time_DB: " << db_time << std::endl;
    std::cout << "Time_Total: " << (cpu_load_time + h2d_time + dunn_time + sil_time + db_time) << std::endl;

    return 0;
}
