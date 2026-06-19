// ======================================================================
//  baseline_cpu.cpp  -  Versao MATRIX-FREE (sem materializar a matriz NxN)
// ----------------------------------------------------------------------
//  Baseline sequencial/paralelo (CPU) das tres metricas de validacao de
//  clusters: Indice de Dunn, Coeficiente de Silhueta e Davies-Bouldin.
//
//  Mudancas em relacao a versao anterior:
//    - NAO aloca mais a matriz de distancias D[N*N] (eram 80 GB em N=100k
//      e havia overflow de int em N*N). As distancias sao recalculadas
//      on-the-fly -> memoria O(N*D), viabiliza N=50.000/100.000.
//    - Paralelizacao opcional com OpenMP (compile com -fopenmp). O numero
//      de threads e controlado pela variavel de ambiente OMP_NUM_THREADS,
//      permitindo medir CPU 1-thread vs CPU multi-thread vs GPU.
//
//  Compilacao:
//    g++ -O3 -fopenmp baseline_cpu.cpp -o baseline_cpu     (multi-thread)
//    g++ -O3            baseline_cpu.cpp -o baseline_cpu     (sequencial)
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
#ifdef _OPENMP
#include <omp.h>
#endif

struct Dataset {
    int N;
    int D;
    int K;
    std::vector<double> X;
    std::vector<int> labels;
    std::vector<int> unique_labels;
};

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
        for (int d = 0; d < ds.D; ++d) {
            ss >> ds.X[(size_t)i * ds.D + d];
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

// Distancia euclidiana on-the-fly entre os pontos i e j
static inline double dist_ij(const Dataset& ds, int i, int j) {
    double sum = 0.0;
    const double* xi = &ds.X[(size_t)i * ds.D];
    const double* xj = &ds.X[(size_t)j * ds.D];
    for (int d = 0; d < ds.D; ++d) {
        double diff = xi[d] - xj[d];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// 1) Indice de Dunn (matrix-free)
double compute_dunn_index(const Dataset& ds) {
    double max_intra = 0.0;
    double min_inter = std::numeric_limits<double>::infinity();

    #pragma omp parallel for schedule(dynamic, 64) \
            reduction(max:max_intra) reduction(min:min_inter)
    for (int i = 0; i < ds.N; ++i) {
        int li = ds.labels[i];
        for (int j = i + 1; j < ds.N; ++j) {
            double dist = dist_ij(ds, i, j);
            if (li == ds.labels[j]) {
                if (dist > max_intra) max_intra = dist;
            } else {
                if (dist < min_inter) min_inter = dist;
            }
        }
    }

    if (max_intra == 0.0) return 0.0;
    return min_inter / max_intra;
}

// 2) Coeficiente de Silhueta (matrix-free)
double compute_silhouette_index(const Dataset& ds) {
    std::vector<int> cluster_sizes(ds.K, 0);
    for (int i = 0; i < ds.N; ++i) cluster_sizes[ds.labels[i]]++;

    double silhouette_sum = 0.0;

    #pragma omp parallel for schedule(dynamic, 64) reduction(+:silhouette_sum)
    for (int i = 0; i < ds.N; ++i) {
        int own_cluster = ds.labels[i];
        if (cluster_sizes[own_cluster] <= 1) continue;

        std::vector<double> dist_sum(ds.K, 0.0);
        for (int j = 0; j < ds.N; ++j) {
            if (j == i) continue;
            dist_sum[ds.labels[j]] += dist_ij(ds, i, j);
        }

        double a = dist_sum[own_cluster] / (cluster_sizes[own_cluster] - 1);
        double b = std::numeric_limits<double>::infinity();
        for (int c = 0; c < ds.K; ++c) {
            if (c == own_cluster) continue;
            if (cluster_sizes[c] == 0) continue;
            double avg_dist = dist_sum[c] / cluster_sizes[c];
            if (avg_dist < b) b = avg_dist;
        }
        silhouette_sum += (b - a) / std::max(a, b);
    }

    return silhouette_sum / ds.N;
}

// 3) Indice Davies-Bouldin (baseado em centroides; ja era O(N))
double compute_davies_bouldin_index(const Dataset& ds) {
    std::vector<double> centroids((size_t)ds.K * ds.D, 0.0);
    std::vector<int> cluster_sizes(ds.K, 0);

    for (int i = 0; i < ds.N; ++i) {
        int c = ds.labels[i];
        cluster_sizes[c]++;
        for (int d = 0; d < ds.D; ++d) {
            centroids[(size_t)c * ds.D + d] += ds.X[(size_t)i * ds.D + d];
        }
    }
    for (int c = 0; c < ds.K; ++c) {
        if (cluster_sizes[c] > 0) {
            for (int d = 0; d < ds.D; ++d) centroids[(size_t)c * ds.D + d] /= cluster_sizes[c];
        }
    }

    std::vector<double> S(ds.K, 0.0);
    for (int i = 0; i < ds.N; ++i) {
        int c = ds.labels[i];
        double sum_sq = 0.0;
        for (int d = 0; d < ds.D; ++d) {
            double diff = ds.X[(size_t)i * ds.D + d] - centroids[(size_t)c * ds.D + d];
            sum_sq += diff * diff;
        }
        S[c] += std::sqrt(sum_sq);
    }
    for (int c = 0; c < ds.K; ++c) {
        if (cluster_sizes[c] > 0) S[c] /= cluster_sizes[c];
    }

    double db_sum = 0.0;
    for (int i = 0; i < ds.K; ++i) {
        if (cluster_sizes[i] == 0) continue;
        double max_ratio = 0.0;
        for (int j = 0; j < ds.K; ++j) {
            if (i == j || cluster_sizes[j] == 0) continue;
            double sum_sq = 0.0;
            for (int d = 0; d < ds.D; ++d) {
                double diff = centroids[(size_t)i * ds.D + d] - centroids[(size_t)j * ds.D + d];
                sum_sq += diff * diff;
            }
            double M_ij = std::sqrt(sum_sq);
            if (M_ij > 0.0) {
                double ratio = (S[i] + S[j]) / M_ij;
                if (ratio > max_ratio) max_ratio = ratio;
            }
        }
        db_sum += max_ratio;
    }

    return db_sum / ds.K;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <caminho_do_dataset.csv> [apenas_dunn: 0 ou 1]" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    bool run_all = true;
    if (argc >= 3) run_all = (std::stoi(argv[2]) == 0);

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    auto t_start = std::chrono::high_resolution_clock::now();
    Dataset ds = load_dataset(filepath);
    auto t_load = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(t_load - t_start).count();

    // 1. Dunn
    auto t_dunn_start = std::chrono::high_resolution_clock::now();
    double dunn = compute_dunn_index(ds);
    auto t_dunn_end = std::chrono::high_resolution_clock::now();
    double dunn_time = std::chrono::duration<double>(t_dunn_end - t_dunn_start).count();

    double sil = -2.0, sil_time = 0.0, db = -1.0, db_time = 0.0;

    if (run_all) {
        auto t_sil_start = std::chrono::high_resolution_clock::now();
        sil = compute_silhouette_index(ds);
        auto t_sil_end = std::chrono::high_resolution_clock::now();
        sil_time = std::chrono::duration<double>(t_sil_end - t_sil_start).count();

        auto t_db_start = std::chrono::high_resolution_clock::now();
        db = compute_davies_bouldin_index(ds);
        auto t_db_end = std::chrono::high_resolution_clock::now();
        db_time = std::chrono::duration<double>(t_db_end - t_db_start).count();
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_total_end - t_start).count();

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "--- RESULTS ---" << std::endl;
    std::cout << "N: " << ds.N << std::endl;
    std::cout << "D: " << ds.D << std::endl;
    std::cout << "K: " << ds.K << std::endl;
    std::cout << "Threads: " << n_threads << std::endl;
    std::cout << "Dunn: " << dunn << std::endl;
    std::cout << "Silhouette: " << sil << std::endl;
    std::cout << "DB: " << db << std::endl;
    std::cout << "Time_Load: " << load_time << std::endl;
    std::cout << "Time_Dunn: " << dunn_time << std::endl;
    std::cout << "Time_Silhouette: " << sil_time << std::endl;
    std::cout << "Time_DB: " << db_time << std::endl;
    std::cout << "Time_Total: " << total_time << std::endl;

    return 0;
}
