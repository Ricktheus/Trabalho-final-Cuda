# ======================================================================
#  benchmark.py  -  Orquestracao do experimento final (matrix-free)
# ----------------------------------------------------------------------
#  - Compila o baseline CPU (g++ -fopenmp) e o codigo GPU (nvcc).
#  - Valida a corretude (caso analitico do Dunn + scikit-learn).
#  - Mede tempos com REPETICOES -> reporta media +/- desvio-padrao.
#  - Compara TRES motores: CPU 1-thread, CPU multi-thread (OpenMP) e GPU.
#  - Escala ate N=100.000 (viavel gracas a versao matrix-free).
#  - Gera graficos CONSISTENTES (todos no mesmo ambiente/execucao):
#       * bench_tempo.png      -> tempo total vs N (3 curvas, log-log)
#       * bench_speedup.png    -> speed-up vs N (GPU/CPU-1 e GPU/CPU-OMP)
#       * bench_breakdown.png  -> onde esta o tempo na GPU (por kernel)
#       * curva_performance_cuda.png -> figura combinada (compatibilidade)
#  - Salva a tabela final em resultados_benchmark.csv para o artigo.
#
#  Uso:   python benchmark.py            (sweep completo, ate 100k)
#         python benchmark.py --max 8000 (limita N; util p/ testes rapidos)
# ======================================================================

import os
import sys
import subprocess
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# --------------------------- Configuracao -----------------------------
TAMANHOS = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 50000, 100000]
DIMENSOES = 4
CLUSTERS = 5
RANDOM_STATE = 42
ENABLE_FLOAT = True   # compila tambem a versao GPU em float (trade-off precisao x velocidade)

def reps_for(n):
    """Menos repeticoes para N grande (CPU 1-thread fica lenta em O(N^2))."""
    if n <= 8000:
        return 5
    if n <= 32000:
        return 3
    return 2

# Permite limitar o N maximo pela linha de comando (--max N)
if "--max" in sys.argv:
    try:
        limite = int(sys.argv[sys.argv.index("--max") + 1])
        TAMANHOS = [n for n in TAMANHOS if n <= limite]
        print(f"[CONFIG] Limitando N <= {limite}: {TAMANHOS}")
    except (ValueError, IndexError):
        print("[CONFIG] Uso: --max <N>. Ignorando.")


# --------------------------- Geracao de dados -------------------------
def gerar_dataset_csv(filename, n_samples):
    X, y = make_blobs(n_samples=n_samples, n_features=DIMENSOES, centers=CLUSTERS,
                      cluster_std=1.0, random_state=RANDOM_STATE)
    X = X.astype(np.float64)
    y = y.astype(np.int32)
    with open(filename, "w") as f:
        f.write(f"{n_samples} {DIMENSOES} {CLUSTERS}\n")
        for i in range(n_samples):
            coords = " ".join(f"{val:.8f}" for val in X[i])
            f.write(f"{coords} {y[i]}\n")


# --------------------------- Compilacao -------------------------------
def compilar_cpp():
    print("Compilando baseline_cpu.cpp (g++ -O3 -fopenmp)...")
    out_file = "baseline_cpu.exe" if os.name == "nt" else "./baseline_cpu"
    cmd = ["g++", "-O3", "-fopenmp", "baseline_cpu.cpp", "-o", out_file]
    try:
        subprocess.run(cmd, check=True)
        print("Compilacao da CPU concluida com sucesso.")
        return out_file
    except subprocess.CalledProcessError as e:
        print(f"Erro na compilacao da CPU: {e}")
        return None

def compilar_cuda(use_float=False):
    label = "float" if use_float else "double"
    print(f"Verificando nvcc e compilando metrics_cuda.cu ({label})...")
    try:
        subprocess.run(["nvcc", "--version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        print("Compilador nvcc NAO encontrado. Se estiver no Colab, ative a GPU (T4).")
        return None

    if os.name == "nt":
        out_file = f"metrics_cuda_{label}.exe"
    else:
        out_file = f"./metrics_cuda_{label}"
    cmd = ["nvcc", "-O3", "-arch=sm_60", "metrics_cuda.cu", "-o", out_file]
    if use_float:
        cmd.insert(2, "-DUSE_FLOAT")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Compilacao da GPU ({label}) concluida com sucesso.")
        return out_file
    except subprocess.CalledProcessError as e:
        print(f"FALHA NA COMPILACAO DO CODIGO CUDA ({label}):")
        print(e.stderr)
        return None


# --------------------------- Execucao ---------------------------------
def rodar_uma_vez(executable, csv_path, n_threads=None):
    """Executa o binario uma vez e faz o parse das linhas 'chave: valor'."""
    env = os.environ.copy()
    if n_threads is not None:
        env["OMP_NUM_THREADS"] = str(n_threads)
    try:
        res = subprocess.run([executable, csv_path, "0"], capture_output=True,
                             text=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {executable}: {e.stderr}")
        return None
    parsed = {}
    for line in res.stdout.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                parsed[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return parsed

def medir(executable, csv_path, reps, n_threads=None):
    """Roda 'reps' vezes; retorna (resultado_referencia, media, desvio) do Time_Total."""
    tempos = []
    ref = None
    for _ in range(reps):
        r = rodar_uma_vez(executable, csv_path, n_threads)
        if r is None:
            return None, None, None
        if ref is None:
            ref = r
        tempos.append(r["Time_Total"])
    media = statistics.mean(tempos)
    desvio = statistics.stdev(tempos) if len(tempos) > 1 else 0.0
    return ref, media, desvio


# --------------------------- Validacoes -------------------------------
def validar_contra_sklearn(cpu_exec):
    print("\n" + "=" * 64)
    print("VALIDACAO DE CORRETUDE CONTRA SCIKIT-LEARN")
    print("=" * 64)
    X, y = make_blobs(n_samples=150, n_features=4, centers=3, random_state=42)
    temp = "temp_val.csv"
    with open(temp, "w") as f:
        f.write("150 4 3\n")
        for i in range(150):
            f.write(" ".join(f"{v:.8f}" for v in X[i]) + f" {y[i]}\n")
    res = rodar_uma_vez(cpu_exec, temp, n_threads=1)
    if os.path.exists(temp):
        os.remove(temp)
    if not res:
        print("[ERRO] Falha ao rodar a CPU na validacao.")
        return False
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    sil_ref = float(silhouette_score(X, y))
    db_ref = float(davies_bouldin_score(X, y))
    dif_sil = abs(res["Silhouette"] - sil_ref)
    dif_db = abs(res["DB"] - db_ref)
    print(f"  Silhueta  : C++={res['Silhouette']:.6f} | Sklearn={sil_ref:.6f} | dif={dif_sil:.2e}")
    print(f"  Davies-B. : C++={res['DB']:.6f} | Sklearn={db_ref:.6f} | dif={dif_db:.2e}")
    if dif_sil >= 1e-5 or dif_db >= 1e-5:
        print("[FALHA] Divergencia numerica contra scikit-learn!")
        return False
    print("Validacao contra scikit-learn: SUCESSO (100% Match)")
    return True

def validar_dunn_analitico(cpu_exec, gpu_exec):
    print("\n" + "=" * 64)
    print("VALIDACAO DO INDICE DE DUNN CONTRA CASO ANALITICO")
    print("=" * 64)
    X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [5.0, 2.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    temp = "temp_dunn.csv"
    with open(temp, "w") as f:
        f.write("4 2 2\n")
        for i in range(4):
            f.write(" ".join(f"{v:.8f}" for v in X[i]) + f" {y[i]}\n")
    dunn_ref = 2.0
    res_cpu = rodar_uma_vez(cpu_exec, temp, n_threads=1)
    if not res_cpu:
        if os.path.exists(temp):
            os.remove(temp)
        print("[ERRO] Falha ao rodar a CPU no teste analitico.")
        return False
    dif_cpu = abs(res_cpu["Dunn"] - dunn_ref)
    print(f"  Dunn CPU : C++={res_cpu['Dunn']:.6f} | Esperado={dunn_ref:.6f} | dif={dif_cpu:.2e}")
    gpu_ok = True
    if gpu_exec:
        res_gpu = rodar_uma_vez(gpu_exec, temp)
        if res_gpu:
            dif_gpu = abs(res_gpu["Dunn"] - dunn_ref)
            print(f"  Dunn GPU : CUDA={res_gpu['Dunn']:.6f} | Esperado={dunn_ref:.6f} | dif={dif_gpu:.2e}")
            gpu_ok = dif_gpu < 1e-5
        else:
            gpu_ok = False
    if os.path.exists(temp):
        os.remove(temp)
    if dif_cpu >= 1e-5 or not gpu_ok:
        print("[FALHA] Divergencia no caso analitico de Dunn!")
        return False
    print("Validacao analitica do Dunn: SUCESSO (100% Match)")
    return True


# --------------------------- Graficos ---------------------------------
def gerar_graficos(dados, tem_gpu):
    ns = [d["N"] for d in dados]

    # 1) Tempo vs N (log-log)
    plt.figure(figsize=(7, 5))
    plt.plot(ns, [d["cpu1_mean"] for d in dados], "o-", color="crimson", label="CPU 1-thread (C++)")
    plt.plot(ns, [d["cpuN_mean"] for d in dados], "D-", color="darkorange", label=f"CPU OpenMP ({dados[0]['threads']} threads)")
    if tem_gpu:
        plt.plot(ns, [d["gpu_mean"] for d in dados], "s-", color="teal", label="GPU CUDA (double)")
        if dados[0].get("gpuf_mean"):
            plt.plot(ns, [d["gpuf_mean"] for d in dados], "^--", color="seagreen", label="GPU CUDA (float)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Numero de Pontos (N)"); plt.ylabel("Tempo de Execucao (s)")
    plt.title("Tempo de Execucao vs Tamanho do Dataset (log-log)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig("bench_tempo.png", dpi=150); plt.close()

    if tem_gpu:
        # 2) Speed-up
        plt.figure(figsize=(7, 5))
        plt.plot(ns, [d["cpu1_mean"] / d["gpu_mean"] for d in dados], "^-", color="purple", label="GPU vs CPU 1-thread")
        plt.plot(ns, [d["cpuN_mean"] / d["gpu_mean"] for d in dados], "v-", color="navy", label="GPU vs CPU OpenMP")
        plt.axhline(y=1.0, color="gray", linestyle="--")
        plt.xlabel("Numero de Pontos (N)"); plt.ylabel("Speed-up (x)")
        plt.title("Speed-up Total")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig("bench_speedup.png", dpi=150); plt.close()

        # 3) Breakdown por kernel (GPU) - barras empilhadas
        plt.figure(figsize=(8, 5))
        labels = [str(n) for n in ns]
        h2d = np.array([d["gpu_h2d"] for d in dados])
        dunn = np.array([d["gpu_dunn"] for d in dados])
        sil = np.array([d["gpu_sil"] for d in dados])
        dbk = np.array([d["gpu_db"] for d in dados])
        plt.bar(labels, h2d, label="H2D (copia)", color="#bdbdbd")
        plt.bar(labels, dunn, bottom=h2d, label="Dunn", color="#1f77b4")
        plt.bar(labels, sil, bottom=h2d + dunn, label="Silhueta", color="#ff7f0e")
        plt.bar(labels, dbk, bottom=h2d + dunn + sil, label="Davies-Bouldin", color="#2ca02c")
        plt.xlabel("Numero de Pontos (N)"); plt.ylabel("Tempo na GPU (s)")
        plt.title("Onde esta o tempo na GPU (breakdown por etapa)")
        plt.legend(); plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig("bench_breakdown.png", dpi=150); plt.close()

        # 4) Figura combinada (compatibilidade com nome antigo)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(ns, [d["cpu1_mean"] for d in dados], "o-", color="crimson", label="CPU 1-thread")
        ax1.plot(ns, [d["cpuN_mean"] for d in dados], "D-", color="darkorange", label="CPU OpenMP")
        ax1.plot(ns, [d["gpu_mean"] for d in dados], "s-", color="teal", label="GPU CUDA")
        ax1.set_xlabel("N"); ax1.set_ylabel("Tempo (s)"); ax1.set_title("Tempo CPU vs GPU")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(ns, [d["cpu1_mean"] / d["gpu_mean"] for d in dados], "^-", color="purple", label="GPU vs CPU 1-thread")
        ax2.axhline(y=1.0, color="gray", linestyle="--")
        ax2.set_xlabel("N"); ax2.set_ylabel("Speed-up (x)"); ax2.set_title("Speed-up Total")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig("curva_performance_cuda.png", dpi=150); plt.close()
        print("\n[SUCESSO] Graficos gerados: bench_tempo.png, bench_speedup.png, bench_breakdown.png, curva_performance_cuda.png")
    else:
        print("\n[SUCESSO] Grafico gerado: bench_tempo.png (apenas CPU; rode no Colab com GPU para o resto)")


def salvar_csv(dados, tem_gpu):
    with open("resultados_benchmark.csv", "w") as f:
        cab = "N,CPU1_s,CPU1_std,CPUomp_s,CPUomp_std,Threads,GPU_s,GPU_std,SpeedupGPUvsCPU1,SpeedupGPUvsCPUomp,GPU_H2D,GPU_Dunn,GPU_Silhueta,GPU_DB,Dunn,Silhueta,DB,Match\n"
        f.write(cab)
        for d in dados:
            if tem_gpu:
                su1 = d["cpu1_mean"] / d["gpu_mean"]
                suN = d["cpuN_mean"] / d["gpu_mean"]
                f.write(f"{d['N']},{d['cpu1_mean']:.6f},{d['cpu1_std']:.6f},"
                        f"{d['cpuN_mean']:.6f},{d['cpuN_std']:.6f},{d['threads']},"
                        f"{d['gpu_mean']:.6f},{d['gpu_std']:.6f},{su1:.2f},{suN:.2f},"
                        f"{d['gpu_h2d']:.6f},{d['gpu_dunn']:.6f},{d['gpu_sil']:.6f},{d['gpu_db']:.6f},"
                        f"{d['Dunn']:.6f},{d['Silhueta']:.6f},{d['DB']:.6f},{d['match']}\n")
            else:
                f.write(f"{d['N']},{d['cpu1_mean']:.6f},{d['cpu1_std']:.6f},"
                        f"{d['cpuN_mean']:.6f},{d['cpuN_std']:.6f},{d['threads']},"
                        f"N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,"
                        f"{d['Dunn']:.6f},{d['Silhueta']:.6f},{d['DB']:.6f},N/A\n")
    print("[SUCESSO] Tabela salva em resultados_benchmark.csv")


# --------------------------- Principal --------------------------------
def main():
    print("=" * 64)
    cpu_exec = compilar_cpp()
    gpu_exec = compilar_cuda(use_float=False)
    gpu_exec_f = compilar_cuda(use_float=True) if (ENABLE_FLOAT and gpu_exec) else None

    if not cpu_exec:
        print("Erro: nao foi possivel compilar o baseline da CPU. Abortando.")
        return

    if not validar_contra_sklearn(cpu_exec):
        print("Abortando: falha na validacao contra scikit-learn.")
        return
    if not validar_dunn_analitico(cpu_exec, gpu_exec):
        print("Abortando: falha na validacao analitica do Dunn.")
        return

    # numero de threads disponiveis (para a curva OpenMP)
    n_threads = os.cpu_count() or 4

    dados = []
    tem_gpu = gpu_exec is not None

    print("\nIniciando benchmark (media de varias repeticoes)...")
    print(f"{'N':>7} | {'CPU-1 (s)':>11} | {'CPU-OMP (s)':>12} | {'GPU (s)':>10} | {'Speedup':>8} | {'Match':>10}")
    print("-" * 80)

    for n in TAMANHOS:
        reps = reps_for(n)
        csv_file = f"temp_data_{n}.csv"
        gerar_dataset_csv(csv_file, n)

        ref_c1, m_c1, s_c1 = medir(cpu_exec, csv_file, reps, n_threads=1)
        ref_cN, m_cN, s_cN = medir(cpu_exec, csv_file, reps, n_threads=n_threads)

        registro = {
            "N": n, "threads": n_threads,
            "cpu1_mean": m_c1, "cpu1_std": s_c1,
            "cpuN_mean": m_cN, "cpuN_std": s_cN,
            "Dunn": ref_c1["Dunn"], "Silhueta": ref_c1["Silhouette"], "DB": ref_c1["DB"],
            "match": "N/A",
        }

        speed_str = "N/A"
        match_str = "N/A"
        if tem_gpu:
            ref_g, m_g, s_g = medir(gpu_exec, csv_file, reps)
            if ref_g:
                registro.update({
                    "gpu_mean": m_g, "gpu_std": s_g,
                    "gpu_h2d": ref_g.get("Time_H2D", 0.0),
                    "gpu_dunn": ref_g.get("Time_Dunn", 0.0),
                    "gpu_sil": ref_g.get("Time_Silhouette", 0.0),
                    "gpu_db": ref_g.get("Time_DB", 0.0),
                })
                # corretude CPU vs GPU (double)
                dif = max(abs(ref_c1["Dunn"] - ref_g["Dunn"]),
                          abs(ref_c1["Silhouette"] - ref_g["Silhouette"]),
                          abs(ref_c1["DB"] - ref_g["DB"]))
                match_str = "SIM (100%)" if dif < 1e-5 else f"DIF {dif:.1e}"
                registro["match"] = match_str
                speed_str = f"{m_c1 / m_g:6.2f}x"
            # versao float (opcional)
            if gpu_exec_f:
                ref_gf, m_gf, s_gf = medir(gpu_exec_f, csv_file, reps)
                if ref_gf:
                    registro["gpuf_mean"] = m_gf

        dados.append(registro)
        if os.path.exists(csv_file):
            os.remove(csv_file)

        gpu_show = f"{registro.get('gpu_mean', float('nan')):.4f}" if tem_gpu else "N/A"
        print(f"{n:>7} | {m_c1:>9.4f}+-{s_c1:>4.2f} | {m_cN:>9.4f}+-{s_cN:>4.2f} | "
              f"{gpu_show:>10} | {speed_str:>8} | {match_str:>10}")

    gerar_graficos(dados, tem_gpu)
    salvar_csv(dados, tem_gpu)


if __name__ == "__main__":
    main()
