import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Configurações do Benchmark
TAMANHOS = [250, 500, 1000, 2000, 4000, 8000]
DIMENSOES = 4
CLUSTERS = 5
RANDOM_STATE = 42

def gerar_dataset_csv(filename, n_samples):
    """Gera blobs sintéticos e salva em arquivo CSV compatível com C++/CUDA."""
    X, y = make_blobs(n_samples=n_samples, n_features=DIMENSOES, centers=CLUSTERS, 
                      cluster_std=1.0, random_state=RANDOM_STATE)
    X = X.astype(np.float64)
    y = y.astype(np.int32)
    
    with open(filename, 'w') as f:
        f.write(f"{n_samples} {DIMENSOES} {CLUSTERS}\n")
        for i in range(n_samples):
            coords = " ".join(f"{val:.8f}" for val in X[i])
            f.write(f"{coords} {y[i]}\n")

def compilar_cpp():
    """Compila o baseline CPU em C++."""
    print("Compilando baseline_cpu.cpp...")
    compiler = "g++"
    if os.name == 'nt':
        out_file = "baseline_cpu.exe"
    else:
        out_file = "./baseline_cpu"
        
    cmd = [compiler, "-O3", "baseline_cpu.cpp", "-o", out_file]
    try:
        subprocess.run(cmd, check=True)
        print("Compilação da CPU concluída com sucesso.")
        return out_file
    except subprocess.CalledProcessError as e:
        print(f"Erro na compilação: {e}")
        return None

def compilar_cuda():
    """Tenta compilar o código CUDA se o nvcc estiver disponível."""
    print("Verificando compilador CUDA (nvcc)...")
    try:
        subprocess.run(["nvcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        print("Compilador nvcc NÃO encontrado no sistema. Se você estiver no Colab, verifique se ativou a GPU (T4) nas configurações do ambiente.")
        return None
        
    print("Compilador nvcc detectado. Compilando metrics_cuda.cu...")
    if os.name == 'nt':
        out_file = "metrics_cuda.exe"
    else:
        out_file = "./metrics_cuda"
        
    cmd = ["nvcc", "-O3", "-arch=sm_60", "metrics_cuda.cu", "-o", out_file]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Compilação da GPU (CUDA) concluída com sucesso.")
        return out_file
    except subprocess.CalledProcessError as e:
        print("FALHA NA COMPILAÇÃO DO CÓDIGO CUDA:")
        print(e.stderr)
        print(e.stdout)
        return None

def rodar_experimento(executable, csv_path):
    """Executa o executável e faz o parse dos resultados formatados."""
    try:
        # Passa 0 para executar todas as três métricas
        cmd = [executable, csv_path, "0"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        parsed = {}
        for line in res.stdout.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                parsed[k.strip()] = float(v.strip())
        return parsed
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {executable}: {e.stderr}")
        return None

def validar_contra_sklearn(cpu_exec):
    print("\n" + "=" * 64)
    print("VALIDACAO DE CORRETUDE CONTRA SCRIPT SCIKIT-LEARN")
    print("=" * 64)
    X, y = make_blobs(n_samples=150, n_features=4, centers=3, random_state=42)
    temp_val_file = "temp_val.csv"
    with open(temp_val_file, 'w') as f:
        f.write("150 4 3\n")
        for i in range(150):
            coords = " ".join(f"{val:.8f}" for val in X[i])
            f.write(f"{coords} {y[i]}\n")
    
    res = rodar_experimento(cpu_exec, temp_val_file)
    if os.path.exists(temp_val_file):
        os.remove(temp_val_file)
        
    if not res:
        print("[ERRO] Falha ao rodar executavel CPU na validacao.")
        return False
        
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    sil_ref = float(silhouette_score(X, y))
    db_ref = float(davies_bouldin_score(X, y))
    
    dif_sil = abs(res["Silhouette"] - sil_ref)
    dif_db = abs(res["DB"] - db_ref)
    
    print(f"  Silhueta  : C++={res['Silhouette']:.6f} | Sklearn={sil_ref:.6f} | dif={dif_sil:.2e}")
    print(f"  Davies-B. : C++={res['DB']:.6f} | Sklearn={db_ref:.6f} | dif={dif_db:.2e}")
    
    if dif_sil >= 1e-5 or dif_db >= 1e-5:
        print("[FALHA] Divergencia numerica contra scikit-learn detectada!")
        return False
        
    print("Validacao contra scikit-learn: SUCESSO (100% Match)")
    return True

def validar_dunn_analitico(cpu_exec, gpu_exec):
    print("\n" + "=" * 64)
    print("VALIDACAO DO INDICE DE DUNN CONTRA CASO ANALITICO")
    print("=" * 64)
    X = np.array([[0.0, 0.0], [1.0, 0.0],
                  [5.0, 0.0], [5.0, 2.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    
    temp_dunn_file = "temp_dunn.csv"
    with open(temp_dunn_file, 'w') as f:
        f.write("4 2 2\n")
        for i in range(4):
            coords = " ".join(f"{val:.8f}" for val in X[i])
            f.write(f"{coords} {y[i]}\n")
            
    res_cpu = rodar_experimento(cpu_exec, temp_dunn_file)
    dunn_ref = 2.0
    
    if not res_cpu:
        print("[ERRO] Falha ao rodar executavel CPU no teste analitico.")
        if os.path.exists(temp_dunn_file):
            os.remove(temp_dunn_file)
        return False
        
    dif_cpu = abs(res_cpu["Dunn"] - dunn_ref)
    print(f"  Dunn CPU: C++={res_cpu['Dunn']:.6f} | Esperado={dunn_ref:.6f} | dif={dif_cpu:.2e}")
    
    gpu_success = True
    if gpu_exec:
        res_gpu = rodar_experimento(gpu_exec, temp_dunn_file)
        if res_gpu:
            dif_gpu = abs(res_gpu["Dunn"] - dunn_ref)
            print(f"  Dunn GPU: CUDA={res_gpu['Dunn']:.6f} | Esperado={dunn_ref:.6f} | dif={dif_gpu:.2e}")
            if dif_gpu >= 1e-5:
                gpu_success = False
        else:
            gpu_success = False
            
    if os.path.exists(temp_dunn_file):
        os.remove(temp_dunn_file)
        
    if dif_cpu >= 1e-5 or not gpu_success:
        print("[FALHA] Divergencia no caso analitico de Dunn detectada!")
        return False
        
    print("Validacao analitica do Dunn: SUCESSO (100% Match)")
    return True

def main():
    print("=" * 64)
    # Compila os códigos
    cpu_exec = compilar_cpp()
    gpu_exec = compilar_cuda()
    
    if not cpu_exec:
        print("Erro: Não foi possível compilar o baseline da CPU. Abortando.")
        return

    # Validacoes de corretude
    if not validar_contra_sklearn(cpu_exec):
        print("Abortando devido a falha na validacao contra scikit-learn.")
        return
    if not validar_dunn_analitico(cpu_exec, gpu_exec):
        print("Abortando devido a falha na validacao analitica do Dunn.")
        return

    cpu_results = []
    gpu_results = []
    
    print("\nIniciando simulações de benchmark...")
    print(f"{'N':>6} | {'T_CPU (s)':>10} | {'T_GPU (s)':>10} | {'Speed-up':>8} | {'Dunn Match?':>11}")
    print("-" * 64)
    
    for n in TAMANHOS:
        csv_file = f"temp_data_{n}.csv"
        gerar_dataset_csv(csv_file, n)
        
        # CPU
        res_cpu = rodar_experimento(cpu_exec, csv_file)
        if res_cpu:
            cpu_results.append(res_cpu)
            t_cpu = res_cpu["Time_Total"]
        else:
            t_cpu = None
            
        # GPU
        t_gpu = None
        match_str = "N/A"
        if gpu_exec:
            res_gpu = rodar_experimento(gpu_exec, csv_file)
            if res_gpu:
                gpu_results.append(res_gpu)
                t_gpu = res_gpu["Time_Total"]
                # Validação matemática de corretude
                dif_dunn = abs(res_cpu["Dunn"] - res_gpu["Dunn"])
                dif_sil = abs(res_cpu["Silhouette"] - res_gpu["Silhouette"])
                dif_db = abs(res_cpu["DB"] - res_gpu["DB"])
                
                if dif_dunn < 1e-5 and dif_sil < 1e-5 and dif_db < 1e-5:
                    match_str = "SIM (100%)"
                else:
                    match_str = "ERRO"
                    print(f"\n[AVISO] Diferença numérica detectada em N={n}:")
                    print(f"  Dunn CPU: {res_cpu['Dunn']:.6f} | GPU: {res_gpu['Dunn']:.6f}")
                    print(f"  Silh CPU: {res_cpu['Silhouette']:.6f} | GPU: {res_gpu['Silhouette']:.6f}")
                    print(f"  DB   CPU: {res_cpu['DB']:.6f} | GPU: {res_gpu['DB']:.6f}")
        
        # Limpa arquivo temporário
        if os.path.exists(csv_file):
            os.remove(csv_file)
            
        if t_cpu is not None:
            if t_gpu is not None:
                speedup = t_cpu / t_gpu
                print(f"{n:>6} | {t_cpu:>10.4f} | {t_gpu:>10.4f} | {speedup:>7.2f}x | {match_str:>11}")
            else:
                print(f"{n:>6} | {t_cpu:>10.4f} | {'N/A':>10} | {'N/A':>8} | {match_str:>11}")
        else:
            print(f"{n:>6} | {'Erro':>10} | {'N/A':>10} | {'N/A':>8} | {match_str:>11}")

    # Plotar gráficos se tivermos GPU
    if len(gpu_results) > 0 and len(cpu_results) == len(gpu_results):
        ns = np.array(TAMANHOS)
        cpu_times = np.array([r["Time_Total"] for r in cpu_results])
        gpu_times = np.array([r["Time_Total"] for r in gpu_results])
        
        # 1. Gráfico de Tempos
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(ns, cpu_times, 'o-', color='crimson', label='CPU Sequencial (C++)')
        plt.plot(ns, gpu_times, 's-', color='teal', label='GPU Paralela (CUDA)')
        plt.xlabel('Número de Pontos (N)')
        plt.ylabel('Tempo de Execução (s)')
        plt.title('Tempo de Execução vs Tamanho do Dataset')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Gráfico de Speed-up
        plt.subplot(1, 2, 2)
        speedups = cpu_times / gpu_times
        plt.plot(ns, speedups, '^-', color='purple', label='Speed-up Real')
        # Curva de speed-up ideal/referência
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.xlabel('Número de Pontos (N)')
        plt.ylabel('Ganho de Velocidade (x)')
        plt.title('Speed-up Total (CPU / GPU)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('curva_performance_cuda.png', dpi=150)
        print("\n[SUCESSO] Gráfico 'curva_performance_cuda.png' gerado e salvo.")
    else:
        print("\nPara rodar os testes da GPU e gerar os gráficos de speed-up, envie os arquivos ")
        print("`baseline_cpu.cpp`, `metrics_cuda.cu` e `benchmark.py` para um ambiente com GPU NVIDIA ")
        print("(como o Google Colab) e execute `python benchmark.py` lá.")
        
        # Plota apenas a curva CPU
        ns = np.array(TAMANHOS)
        cpu_times = np.array([r["Time_Total"] for r in cpu_results])
        plt.figure(figsize=(6, 4))
        plt.plot(ns, cpu_times, 'o-', color='crimson', label='CPU Sequencial (C++)')
        plt.xlabel('Número de Pontos (N)')
        plt.ylabel('Tempo de Execução (s)')
        plt.title('Curva de Complexidade CPU baseline - O(N²)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('curva_complexidade_cpu.png', dpi=150)
        print("[AVISO] Apenas gráfico 'curva_complexidade_cpu.png' foi gerado.")

if __name__ == "__main__":
    main()
