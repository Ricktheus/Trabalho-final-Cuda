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

// Estrutura para armazenar o dataset na CPU
struct Dataset {
    int N;
    int D;
    int K;
    std::vector<double> X;
    std::vector<int> labels;
    std::vector<int> unique_labels;
};

// Carrega o dataset (mesmo formato do baseline CPU)
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

    ds.X.resize(ds.N * ds.D);
    ds.labels.resize(ds.N);

    std::set<int> label_set;
    for (int i = 0; i < ds.N; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Erro de leitura na linha " << i + 2 << std::endl;
            std::exit(1);
        }
        std::stringstream ss(line);
        for (int d = 0; d < ds.D; ++d) {
            ss >> ds.X[i * ds.D + d];
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

// ----------------------------------------------------------------------
// KERNELS CUDA
// ----------------------------------------------------------------------

// 1. Cálculo da matriz de distâncias par-a-par O(N^2)
__global__ void pairwise_distances_kernel(const double* X, double* D, int N, int Dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Indice de coluna (muda rapido)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Indice de linha (muda devagar)

    if (i < N && j < N) {
        if (i == j) {
            D[i * N + j] = 0.0;
            return;
        }
        double sum = 0.0;
        for (int d = 0; d < Dim; ++d) {
            double diff = X[i * Dim + d] - X[j * Dim + d];
            sum += diff * diff;
        }
        D[i * N + j] = sqrt(sum);
    }
}

// 2. Redução paralela para o Índice de Dunn (encontra max_intra e min_inter de cada linha)
__global__ void dunn_reduction_kernel(const double* D, const int* labels, double* row_max_intra, double* row_min_inter, int N) {
    int i = blockIdx.x; // Um bloco por linha da matriz
    int tx = threadIdx.x;

    __shared__ double s_max[256];
    __shared__ double s_min[256];

    double local_max = 0.0;
    double local_min = 1e15; // Infinito

    int own_label = labels[i];

    for (int j = tx; j < N; j += blockDim.x) {
        if (i == j) continue;
        double dist = D[i * N + j];
        if (labels[j] == own_label) {
            if (dist > local_max) local_max = dist;
        } else {
            if (dist < local_min) local_min = dist;
        }
    }

    s_max[tx] = local_max;
    s_min[tx] = local_min;
    __syncthreads();

    // Redução em memória compartilhada
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

// 3. Coeficiente de Silhueta
// Usa memória compartilhada dinâmica de tamanho blockDim.x * K doubles
__global__ void silhouette_kernel(const double* D, const int* labels, const int* cluster_sizes, double* s, int N, int K) {
    int i = blockIdx.x; // Um bloco por ponto
    int tx = threadIdx.x;

    extern __shared__ double shared_sums[]; // Tamanho: blockDim.x * K

    // Inicializa
    for (int c = 0; c < K; ++c) {
        shared_sums[tx * K + c] = 0.0;
    }
    __syncthreads();

    int own_cluster = labels[i];

    // Acumula as somas das distâncias localmente
    for (int j = tx; j < N; j += blockDim.x) {
        if (i == j) continue;
        int c_j = labels[j];
        if (c_j >= 0 && c_j < K) {
            shared_sums[tx * K + c_j] += D[i * N + j];
        }
    }
    __syncthreads();

    // Redução entre as threads do bloco para cada um dos K clusters
    if (tx < K) {
        double total_sum = 0.0;
        for (int t = 0; t < blockDim.x; ++t) {
            total_sum += shared_sums[t * K + tx];
        }
        shared_sums[tx] = total_sum; // Reutiliza as K primeiras posições
    }
    __syncthreads();

    if (tx == 0) {
        if (cluster_sizes[own_cluster] <= 1) {
            s[i] = 0.0;
        } else {
            double a = shared_sums[own_cluster] / (cluster_sizes[own_cluster] - 1);
            double b = 1e15; // Infinito
            for (int c = 0; c < K; ++c) {
                if (c == own_cluster) continue;
                if (cluster_sizes[c] == 0) continue;
                double avg_dist = shared_sums[c] / cluster_sizes[c];
                if (avg_dist < b) {
                    b = avg_dist;
                }
            }
            s[i] = (b - a) / fmax(a, b);
        }
    }
}

// 4. Davies-Bouldin Kernels
__global__ void compute_centroids_kernel(const double* X, const int* labels, double* centroids, int* cluster_sizes, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int c = labels[i];
        for (int d = 0; d < D; ++d) {
            atomicAdd(&centroids[c * D + d], X[i * D + d]);
        }
        atomicAdd(&cluster_sizes[c], 1);
    }
}

__global__ void divide_centroids_kernel(double* centroids, const int* cluster_sizes, int K, int D) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K) {
        int size = cluster_sizes[c];
        if (size > 0) {
            for (int d = 0; d < D; ++d) {
                centroids[c * D + d] /= size;
            }
        }
    }
}

__global__ void compute_dispersion_kernel(const double* X, const int* labels, const double* centroids, double* S, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int c = labels[i];
        double sum_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = X[i * D + d] - centroids[c * D + d];
            sum_sq += diff * diff;
        }
        double dist = sqrt(sum_sq);
        atomicAdd(&S[c], dist);
    }
}

__global__ void divide_dispersion_kernel(double* S, const int* cluster_sizes, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K) {
        int size = cluster_sizes[c];
        if (size > 0) {
            S[c] /= size;
        }
    }
}

__global__ void compute_db_kernel(const double* S, const double* centroids, double* db_ratios, int K, int D) {
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
                if (ratio > max_ratio) {
                    max_ratio = ratio;
                }
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
    if (argc >= 3) {
        run_all = (std::stoi(argv[2]) == 0);
    }

    // 1. Carrega o dataset na CPU
    auto t_load_start = std::chrono::high_resolution_clock::now();
    Dataset ds = load_dataset(filepath);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double cpu_load_time = std::chrono::duration<double>(t_load_end - t_load_start).count();

    // 2. Alocação e cópia de memória Host -> Device
    double* d_X;
    int* d_labels;
    double* d_D;
    
    CUDA_CHECK(cudaMalloc(&d_X, ds.N * ds.D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_labels, ds.N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_D, ds.N * ds.N * sizeof(double)));

    auto t_copy_h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_X, ds.X.data(), ds.N * ds.D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, ds.labels.data(), ds.N * sizeof(int), cudaMemcpyHostToDevice));
    auto t_copy_h2d_end = std::chrono::high_resolution_clock::now();
    double copy_h2d_time = std::chrono::duration<double>(t_copy_h2d_end - t_copy_h2d_start).count();

    // Inicialização do profiler de CUDA
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ------------------------------------------------------------------
    // A. Execução da Matriz de Distâncias par-a-par
    // ------------------------------------------------------------------
    dim3 threadsPerBlock2D(16, 16);
    dim3 numBlocks2D((ds.N + 15) / 16, (ds.N + 15) / 16);

    CUDA_CHECK(cudaEventRecord(start));
    pairwise_distances_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_X, d_D, ds.N, ds.D);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float dist_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&dist_ms, start, stop));
    double dist_time = dist_ms / 1000.0;

    // ------------------------------------------------------------------
    // B. Execução do Índice de Dunn
    // ------------------------------------------------------------------
    double* d_row_max;
    double* d_row_min;
    CUDA_CHECK(cudaMalloc(&d_row_max, ds.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_row_min, ds.N * sizeof(double)));

    CUDA_CHECK(cudaEventRecord(start));
    dunn_reduction_kernel<<<ds.N, 256>>>(d_D, d_labels, d_row_max, d_row_min, ds.N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float dunn_reduction_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&dunn_reduction_ms, start, stop));
    double dunn_gpu_calc_time = dunn_reduction_ms / 1000.0;

    // Copia os vetores reduzidos de linha de volta para a CPU para a redução global final
    std::vector<double> h_row_max(ds.N);
    std::vector<double> h_row_min(ds.N);
    
    auto t_copy_d2h_dunn_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_row_max.data(), d_row_max, ds.N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row_min.data(), d_row_min, ds.N * sizeof(double), cudaMemcpyDeviceToHost));
    auto t_copy_d2h_dunn_end = std::chrono::high_resolution_clock::now();
    double copy_d2h_dunn_time = std::chrono::duration<double>(t_copy_d2h_dunn_end - t_copy_d2h_dunn_start).count();

    auto t_dunn_final_start = std::chrono::high_resolution_clock::now();
    double max_intra = 0.0;
    double min_inter = std::numeric_limits<double>::infinity();
    for (int i = 0; i < ds.N; ++i) {
        if (h_row_max[i] > max_intra) max_intra = h_row_max[i];
        if (h_row_min[i] < min_inter) min_inter = h_row_min[i];
    }
    double dunn_score = (max_intra == 0.0) ? 0.0 : (min_inter / max_intra);
    auto t_dunn_final_end = std::chrono::high_resolution_clock::now();
    double dunn_final_cpu_time = std::chrono::duration<double>(t_dunn_final_end - t_dunn_final_start).count();

    double dunn_time = dunn_gpu_calc_time + copy_d2h_dunn_time + dunn_final_cpu_time;

    // Libera Dunn auxiliares
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_min));

    // ------------------------------------------------------------------
    // C. Execução do Coeficiente de Silhueta
    // ------------------------------------------------------------------
    double sil_score = -2.0;
    double sil_time = 0.0;

    if (run_all) {
        // Precisamos dos tamanhos dos clusters na GPU
        int* d_cluster_sizes;
        CUDA_CHECK(cudaMalloc(&d_cluster_sizes, ds.K * sizeof(int)));
        
        // Vamos calcular os tamanhos na CPU e copiar para a GPU para poupar tempo
        std::vector<int> h_cluster_sizes(ds.K, 0);
        for (int i = 0; i < ds.N; ++i) {
            h_cluster_sizes[ds.labels[i]]++;
        }
        CUDA_CHECK(cudaMemcpy(d_cluster_sizes, h_cluster_sizes.data(), ds.K * sizeof(int), cudaMemcpyHostToDevice));

        double* d_s;
        CUDA_CHECK(cudaMalloc(&d_s, ds.N * sizeof(double)));

        // Tamanho da memória compartilhada dinâmica: blockDim.x * K * sizeof(double)
        size_t shared_mem_size = 256 * ds.K * sizeof(double);

        CUDA_CHECK(cudaEventRecord(start));
        silhouette_kernel<<<ds.N, 256, shared_mem_size>>>(d_D, d_labels, d_cluster_sizes, d_s, ds.N, ds.K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float sil_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&sil_ms, start, stop));
        double sil_gpu_time = sil_ms / 1000.0;

        std::vector<double> h_s(ds.N);
        auto t_copy_sil_start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(h_s.data(), d_s, ds.N * sizeof(double), cudaMemcpyDeviceToHost));
        auto t_copy_sil_end = std::chrono::high_resolution_clock::now();
        double copy_sil_time = std::chrono::duration<double>(t_copy_sil_end - t_copy_sil_start).count();

        auto t_sil_final_start = std::chrono::high_resolution_clock::now();
        double sil_sum = 0.0;
        for (int i = 0; i < ds.N; ++i) {
            sil_sum += h_s[i];
        }
        sil_score = sil_sum / ds.N;
        auto t_sil_final_end = std::chrono::high_resolution_clock::now();
        double sil_final_cpu_time = std::chrono::duration<double>(t_sil_final_end - t_sil_final_start).count();

        sil_time = sil_gpu_time + copy_sil_time + sil_final_cpu_time;

        CUDA_CHECK(cudaFree(d_cluster_sizes));
        CUDA_CHECK(cudaFree(d_s));
    }

    // ------------------------------------------------------------------
    // D. Execução do Davies-Bouldin
    // ------------------------------------------------------------------
    double db_score = -1.0;
    double db_time = 0.0;

    if (run_all) {
        double* d_centroids;
        int* d_db_cluster_sizes;
        double* d_S;
        double* d_db_ratios;

        CUDA_CHECK(cudaMalloc(&d_centroids, ds.K * ds.D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_db_cluster_sizes, ds.K * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_S, ds.K * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_db_ratios, ds.K * sizeof(double)));

        CUDA_CHECK(cudaMemset(d_centroids, 0, ds.K * ds.D * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_db_cluster_sizes, 0, ds.K * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_S, 0, ds.K * sizeof(double)));

        int threadsPerBlock = 256;
        int blocksForPoints = (ds.N + threadsPerBlock - 1) / threadsPerBlock;
        int blocksForClusters = (ds.K + threadsPerBlock - 1) / threadsPerBlock;

        CUDA_CHECK(cudaEventRecord(start));
        
        // 1. Somar coordenadas para centroides
        compute_centroids_kernel<<<blocksForPoints, threadsPerBlock>>>(d_X, d_labels, d_centroids, d_db_cluster_sizes, ds.N, ds.D);
        
        // 2. Dividir para obter a média
        divide_centroids_kernel<<<blocksForClusters, threadsPerBlock>>>(d_centroids, d_db_cluster_sizes, ds.K, ds.D);
        
        // 3. Somar distâncias para dispersão
        compute_dispersion_kernel<<<blocksForPoints, threadsPerBlock>>>(d_X, d_labels, d_centroids, d_S, ds.N, ds.D);
        
        // 4. Dividir dispersão pelo tamanho
        divide_dispersion_kernel<<<blocksForClusters, threadsPerBlock>>>(d_S, d_db_cluster_sizes, ds.K);
        
        // 5. Calcular a pior razão para cada cluster
        compute_db_kernel<<<blocksForClusters, threadsPerBlock>>>(d_S, d_centroids, d_db_ratios, ds.K, ds.D);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float db_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&db_ms, start, stop));
        double db_gpu_time = db_ms / 1000.0;

        std::vector<double> h_db_ratios(ds.K);
        auto t_copy_db_start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(h_db_ratios.data(), d_db_ratios, ds.K * sizeof(double), cudaMemcpyDeviceToHost));
        auto t_copy_db_end = std::chrono::high_resolution_clock::now();
        double copy_db_time = std::chrono::duration<double>(t_copy_db_end - t_copy_db_start).count();

        auto t_db_final_start = std::chrono::high_resolution_clock::now();
        double db_sum = 0.0;
        for (int i = 0; i < ds.K; ++i) {
            db_sum += h_db_ratios[i];
        }
        db_score = db_sum / ds.K;
        auto t_db_final_end = std::chrono::high_resolution_clock::now();
        double db_final_cpu_time = std::chrono::duration<double>(t_db_final_end - t_db_final_start).count();

        db_time = db_gpu_time + copy_db_time + db_final_cpu_time;

        CUDA_CHECK(cudaFree(d_centroids));
        CUDA_CHECK(cudaFree(d_db_cluster_sizes));
        CUDA_CHECK(cudaFree(d_S));
        CUDA_CHECK(cudaFree(d_db_ratios));
    }

    // Libera memória global principal da GPU
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_D));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Output formatado para leitura pelo script Python
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "--- RESULTS ---" << std::endl;
    std::cout << "N: " << ds.N << std::endl;
    std::cout << "D: " << ds.D << std::endl;
    std::cout << "K: " << ds.K << std::endl;
    std::cout << "Dunn: " << dunn_score << std::endl;
    std::cout << "Silhouette: " << sil_score << std::endl;
    std::cout << "DB: " << db_score << std::endl;
    std::cout << "Time_Load: " << cpu_load_time << std::endl;
    std::cout << "Time_Distances: " << dist_time << std::endl;
    std::cout << "Time_Dunn: " << dunn_time << std::endl;
    std::cout << "Time_Silhouette: " << sil_time << std::endl;
    std::cout << "Time_DB: " << db_time << std::endl;
    std::cout << "Time_Total: " << (cpu_load_time + copy_h2d_time + dist_time + dunn_time + sil_time + db_time) << std::endl;

    return 0;
}
