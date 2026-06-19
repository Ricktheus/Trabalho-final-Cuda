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

// Estrutura para armazenar os dados lidos do CSV
struct Dataset {
    int N; // Número de pontos
    int D; // Dimensões por ponto
    int K; // Número de clusters únicos
    std::vector<double> X; // Coordenadas linearizadas (N * D)
    std::vector<int> labels; // Rótulos dos pontos (N)
    std::vector<int> unique_labels; // Rótulos únicos ordenados
};

// Carrega o dataset de um arquivo formatado
Dataset load_dataset(const std::string& filepath) {
    Dataset ds;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo: " << filepath << std::endl;
        std::exit(1);
    }

    std::string line;
    // Primeira linha: N D K
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
    // Mapeia os labels para 0..K-1
    for (int i = 0; i < ds.N; ++i) {
        auto it = std::find(ds.unique_labels.begin(), ds.unique_labels.end(), ds.labels[i]);
        ds.labels[i] = std::distance(ds.unique_labels.begin(), it);
    }

    return ds;
}

// Calcula a matriz de distâncias euclidianas par-a-par O(N^2)
std::vector<double> compute_pairwise_distances(const Dataset& ds) {
    std::vector<double> D_mat(ds.N * ds.N, 0.0);
    for (int i = 0; i < ds.N; ++i) {
        for (int j = i; j < ds.N; ++j) {
            if (i == j) {
                D_mat[i * ds.N + j] = 0.0;
                continue;
            }
            double sum = 0.0;
            for (int d = 0; d < ds.D; ++d) {
                double diff = ds.X[i * ds.D + d] - ds.X[j * ds.D + d];
                sum += diff * diff;
            }
            double dist = std::sqrt(sum);
            D_mat[i * ds.N + j] = dist;
            D_mat[j * ds.N + i] = dist; // Simétrica
        }
    }
    return D_mat;
}

// 1) Índice de Dunn
double compute_dunn_index(const Dataset& ds, const std::vector<double>& D_mat) {
    double max_intra = 0.0;
    double min_inter = std::numeric_limits<double>::infinity();

    for (int i = 0; i < ds.N; ++i) {
        for (int j = i + 1; j < ds.N; ++j) {
            double dist = D_mat[i * ds.N + j];
            if (ds.labels[i] == ds.labels[j]) {
                if (dist > max_intra) {
                    max_intra = dist;
                }
            } else {
                if (dist < min_inter) {
                    min_inter = dist;
                }
            }
        }
    }

    if (max_intra == 0.0) return 0.0;
    return min_inter / max_intra;
}

// 2) Coeficiente de Silhueta
double compute_silhouette_index(const Dataset& ds, const std::vector<double>& D_mat) {
    std::vector<int> cluster_sizes(ds.K, 0);
    for (int i = 0; i < ds.N; ++i) {
        cluster_sizes[ds.labels[i]]++;
    }

    double silhouette_sum = 0.0;

    for (int i = 0; i < ds.N; ++i) {
        int own_cluster = ds.labels[i];
        if (cluster_sizes[own_cluster] <= 1) {
            // Se o cluster possui apenas 1 elemento, a silhueta desse ponto é 0 por definição
            continue;
        }

        // Calcula a soma de distâncias para cada cluster
        std::vector<double> dist_sum(ds.K, 0.0);
        for (int j = 0; j < ds.N; ++j) {
            dist_sum[ds.labels[j]] += D_mat[i * ds.N + j];
        }

        // Distância média intra-cluster (a_i)
        // Nota: subtrai a distância de i para ele mesmo (que é 0), e divide por size - 1
        double a = dist_sum[own_cluster] / (cluster_sizes[own_cluster] - 1);

        // Menor distância média inter-cluster (b_i)
        double b = std::numeric_limits<double>::infinity();
        for (int c = 0; c < ds.K; ++c) {
            if (c == own_cluster) continue;
            if (cluster_sizes[c] == 0) continue;
            double avg_dist = dist_sum[c] / cluster_sizes[c];
            if (avg_dist < b) {
                b = avg_dist;
            }
        }

        double s_i = (b - a) / std::max(a, b);
        silhouette_sum += s_i;
    }

    return silhouette_sum / ds.N;
}

// 3) Índice Davies-Bouldin
double compute_davies_bouldin_index(const Dataset& ds) {
    // 1. Calcular centroides
    std::vector<double> centroids(ds.K * ds.D, 0.0);
    std::vector<int> cluster_sizes(ds.K, 0);

    for (int i = 0; i < ds.N; ++i) {
        int c = ds.labels[i];
        cluster_sizes[c]++;
        for (int d = 0; d < ds.D; ++d) {
            centroids[c * ds.D + d] += ds.X[i * ds.D + d];
        }
    }

    for (int c = 0; c < ds.K; ++c) {
        if (cluster_sizes[c] > 0) {
            for (int d = 0; d < ds.D; ++d) {
                centroids[c * ds.D + d] /= cluster_sizes[c];
            }
        }
    }

    // 2. Calcular dispersão interna S_i (distância média dos pontos ao centroide do cluster)
    std::vector<double> S(ds.K, 0.0);
    for (int i = 0; i < ds.N; ++i) {
        int c = ds.labels[i];
        double sum_sq = 0.0;
        for (int d = 0; d < ds.D; ++d) {
            double diff = ds.X[i * ds.D + d] - centroids[c * ds.D + d];
            sum_sq += diff * diff;
        }
        S[c] += std::sqrt(sum_sq);
    }

    for (int c = 0; c < ds.K; ++c) {
        if (cluster_sizes[c] > 0) {
            S[c] /= cluster_sizes[c];
        }
    }

    // 3. Calcular a razão R_ij e encontrar o pior caso para cada cluster i
    double db_sum = 0.0;
    for (int i = 0; i < ds.K; ++i) {
        if (cluster_sizes[i] == 0) continue;
        double max_ratio = 0.0;
        for (int j = 0; j < ds.K; ++j) {
            if (i == j || cluster_sizes[j] == 0) continue;

            // Distância entre centroides i e j
            double sum_sq = 0.0;
            for (int d = 0; d < ds.D; ++d) {
                double diff = centroids[i * ds.D + d] - centroids[j * ds.D + d];
                sum_sq += diff * diff;
            }
            double M_ij = std::sqrt(sum_sq);

            if (M_ij > 0.0) {
                double ratio = (S[i] + S[j]) / M_ij;
                if (ratio > max_ratio) {
                    max_ratio = ratio;
                }
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
    if (argc >= 3) {
        run_all = (std::stoi(argv[2]) == 0);
    }

    // Medição de tempo total incluindo carregamento e computação
    auto t_start = std::chrono::high_resolution_clock::now();

    Dataset ds = load_dataset(filepath);

    auto t_load = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(t_load - t_start).count();

    // 1. Matriz de distâncias
    auto t_dist_start = std::chrono::high_resolution_clock::now();
    std::vector<double> D_mat = compute_pairwise_distances(ds);
    auto t_dist_end = std::chrono::high_resolution_clock::now();
    double dist_time = std::chrono::duration<double>(t_dist_end - t_dist_start).count();

    // 2. Dunn Index
    auto t_dunn_start = std::chrono::high_resolution_clock::now();
    double dunn = compute_dunn_index(ds, D_mat);
    auto t_dunn_end = std::chrono::high_resolution_clock::now();
    double dunn_time = std::chrono::duration<double>(t_dunn_end - t_dunn_start).count();

    double sil = -2.0;
    double sil_time = 0.0;
    double db = -1.0;
    double db_time = 0.0;

    if (run_all) {
        // 3. Silhouette Index
        auto t_sil_start = std::chrono::high_resolution_clock::now();
        sil = compute_silhouette_index(ds, D_mat);
        auto t_sil_end = std::chrono::high_resolution_clock::now();
        sil_time = std::chrono::duration<double>(t_sil_end - t_sil_start).count();

        // 4. Davies-Bouldin Index
        auto t_db_start = std::chrono::high_resolution_clock::now();
        db = compute_davies_bouldin_index(ds);
        auto t_db_end = std::chrono::high_resolution_clock::now();
        db_time = std::chrono::duration<double>(t_db_end - t_db_start).count();
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_total_end - t_start).count();

    // Output formatado para ser lido facilmente pelo Python script
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "--- RESULTS ---" << std::endl;
    std::cout << "N: " << ds.N << std::endl;
    std::cout << "D: " << ds.D << std::endl;
    std::cout << "K: " << ds.K << std::endl;
    std::cout << "Dunn: " << dunn << std::endl;
    std::cout << "Silhouette: " << sil << std::endl;
    std::cout << "DB: " << db << std::endl;
    std::cout << "Time_Load: " << load_time << std::endl;
    std::cout << "Time_Distances: " << dist_time << std::endl;
    std::cout << "Time_Dunn: " << dunn_time << std::endl;
    std::cout << "Time_Silhouette: " << sil_time << std::endl;
    std::cout << "Time_DB: " << db_time << std::endl;
    std::cout << "Time_Total: " << total_time << std::endl;

    return 0;
}
