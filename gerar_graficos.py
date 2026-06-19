# ======================================================================
#  gerar_graficos.py  -  Gera os graficos finais a partir do CSV
# ----------------------------------------------------------------------
#  Le 'resultados_benchmark.csv' (produzido pelo benchmark.py) e gera
#  versoes melhoradas dos graficos. NAO precisa de GPU nem de sklearn:
#  roda em qualquer lugar com matplotlib/pandas (inclusive local).
#
#  Uso:  python gerar_graficos.py  [caminho_do_csv]
#
#  Saidas:
#    bench_tempo.png      -> painel duplo: LINEAR (impacto) + LOG-LOG (escala)
#    bench_speedup.png    -> speed-up com eixo log em N (sem "pico" em N pequeno)
#    bench_breakdown.png  -> painel duplo: linhas log-log + barras 100% empilhadas
# ======================================================================

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "resultados_benchmark.csv"

# Cores consistentes em todos os graficos
C_CPU1 = "#d62728"   # vermelho
C_CPUO = "#ff7f0e"   # laranja
C_GPU  = "#17becf"   # ciano/teal
C_DUNN = "#1f77b4"   # azul
C_SILH = "#ff7f0e"   # laranja
C_H2D  = "#7f7f7f"   # cinza
C_DB   = "#2ca02c"   # verde


def carregar(csv):
    df = pd.read_csv(csv)
    # garante ordenacao por N
    return df.sort_values("N").reset_index(drop=True)


# ----------------------------------------------------------------------
# 1) TEMPO: painel LINEAR (impacto) + LOG-LOG (escalabilidade)
# ----------------------------------------------------------------------
def grafico_tempo(df):
    N = df["N"].values
    cpu1, cpu1e = df["CPU1_s"].values, df["CPU1_std"].values
    cpuo, cpuoe = df["CPUomp_s"].values, df["CPUomp_std"].values
    gpu, gpue = df["GPU_s"].values, df["GPU_std"].values
    thr = int(df["Threads"].iloc[0])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- ESQUERDA: linear (impacto visual do tamanho do gap) ---
    axL.errorbar(N, cpu1, yerr=cpu1e, fmt="o-", color=C_CPU1, capsize=3, label="CPU 1-thread (C++)")
    axL.errorbar(N, cpuo, yerr=cpuoe, fmt="D-", color=C_CPUO, capsize=3, label=f"CPU OpenMP ({thr} threads)")
    axL.errorbar(N, gpu, yerr=gpue, fmt="s-", color=C_GPU, capsize=3, label="GPU CUDA (double)")
    axL.set_xlabel("Numero de Pontos (N)")
    axL.set_ylabel("Tempo de Execucao (s)")
    axL.set_title("Impacto real — escala LINEAR")
    axL.legend(loc="upper left")
    axL.grid(True, alpha=0.3)

    # anota o gap no maior N
    i = int(np.argmax(N))
    su = cpu1[i] / gpu[i]
    axL.annotate(
        f"N={N[i]:,}\nCPU: {cpu1[i]:.1f} s\nGPU: {gpu[i]:.2f} s\n→ {su:.1f}× mais rapido".replace(",", "."),
        xy=(N[i], gpu[i]), xytext=(N[i] * 0.45, cpu1[i] * 0.55),
        fontsize=10, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#e0a800"),
        arrowprops=dict(arrowstyle="->", color="#333"))
    # barra dupla CPU vs GPU no maior N (reforco visual)
    axL.annotate("", xy=(N[i], cpu1[i]), xytext=(N[i], gpu[i]),
                 arrowprops=dict(arrowstyle="<->", color="#888", lw=1.2))

    # --- DIREITA: log-log (escalabilidade O(N^2)) ---
    axR.loglog(N, cpu1, "o-", color=C_CPU1, label="CPU 1-thread")
    axR.loglog(N, cpuo, "D-", color=C_CPUO, label="CPU OpenMP")
    axR.loglog(N, gpu, "s-", color=C_GPU, label="GPU CUDA")
    # reta-guia de inclinacao 2 (O(N^2)) ancorada na CPU
    Nref = np.array([N[0], N[-1]], dtype=float)
    guia = cpu1[0] * (Nref / N[0]) ** 2
    axR.loglog(Nref, guia, ":", color="black", alpha=0.6, label="referencia $O(N^2)$")
    axR.set_xlabel("Numero de Pontos (N)")
    axR.set_ylabel("Tempo de Execucao (s)")
    axR.set_title("Escalabilidade — escala LOG-LOG")
    axR.legend(loc="upper left")
    axR.grid(True, which="both", alpha=0.3)
    axR.text(0.97, 0.05,
             "Curvas paralelas e com inclinacao ~2:\nmesmo crescimento O(N^2);\na distancia vertical = speed-up",
             transform=axR.transAxes, fontsize=9, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#eef", ec="#99c"))

    plt.tight_layout()
    plt.savefig("bench_tempo.png", dpi=150)
    plt.close()
    print("[OK] bench_tempo.png")


# ----------------------------------------------------------------------
# 2) SPEED-UP: eixo log em N (resolve o "pico" em N pequeno)
# ----------------------------------------------------------------------
def grafico_speedup(df):
    N = df["N"].values
    su1 = df["SpeedupGPUvsCPU1"].values
    suo = df["SpeedupGPUvsCPUomp"].values

    fig, ax = plt.subplots(figsize=(9.5, 6))

    # zona de overhead (N pequeno, tempos sub-ms e ruidosos)
    ax.axvspan(N.min() * 0.8, 1000, color="gray", alpha=0.10)
    ax.text(np.sqrt(N.min() * 1000), ax.get_ylim()[1] * 0.92 if False else 23,
            "regiao dominada por overhead\n(tempos < 1 ms, ruidosos)",
            fontsize=8.5, ha="center", va="top", color="#555")

    ax.semilogx(N, su1, "^-", color="#7b2d8e", lw=2, ms=8, label="GPU vs CPU 1-thread")
    ax.semilogx(N, suo, "v-", color="#1f3a93", lw=2, ms=8, label="GPU vs CPU OpenMP")
    ax.axhline(y=1.0, color="gray", linestyle="--", label="sem ganho (1×)")

    # marca o maximo
    i = int(np.argmax(N))
    ax.annotate(f"{su1[i]:.1f}×", xy=(N[i], su1[i]), xytext=(N[i] * 0.6, su1[i] + 1.5),
                fontsize=12, fontweight="bold", color="#7b2d8e",
                arrowprops=dict(arrowstyle="->", color="#7b2d8e"))
    # marca a anomalia do OpenMP em N=250
    ax.annotate("overhead de criar\nthreads OpenMP\n(N muito pequeno)",
                xy=(N[0], suo[0]), xytext=(N[0] * 1.3, suo[0] + 2.0),
                fontsize=8.5, color="#1f3a93",
                arrowprops=dict(arrowstyle="->", color="#1f3a93"))

    ax.set_xlabel("Numero de Pontos (N) — escala log")
    ax.set_ylabel("Speed-up (x)")
    ax.set_title("Speed-up Total da GPU (cresce com o tamanho do problema)")
    ax.legend(loc="center right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("bench_speedup.png", dpi=150)
    plt.close()
    print("[OK] bench_speedup.png")


# ----------------------------------------------------------------------
# 3) BREAKDOWN: linhas log-log (todas visiveis) + barras 100% empilhadas
# ----------------------------------------------------------------------
def grafico_breakdown(df):
    N = df["N"].values
    h2d = df["GPU_H2D"].values
    dunn = df["GPU_Dunn"].values
    silh = df["GPU_Silhueta"].values
    dbk = df["GPU_DB"].values

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- ESQUERDA: linhas log-log (cada etapa visivel em TODO N) ---
    axL.loglog(N, dunn, "o-", color=C_DUNN, label="Dunn")
    axL.loglog(N, silh, "s-", color=C_SILH, label="Silhueta")
    axL.loglog(N, h2d, "^--", color=C_H2D, label="H2D (copia)")
    axL.loglog(N, dbk, "v--", color=C_DB, label="Davies-Bouldin")
    axL.set_xlabel("Numero de Pontos (N)")
    axL.set_ylabel("Tempo na GPU (s)")
    axL.set_title("Tempo por etapa — LOG-LOG (todas as etapas visiveis)")
    axL.legend(loc="upper left")
    axL.grid(True, which="both", alpha=0.3)
    axL.text(0.97, 0.05,
             "Dunn e Silhueta crescem ~O(N^2)\nH2D e DB ficam baixos e ~constantes",
             transform=axL.transAxes, fontsize=9, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#eef", ec="#99c"))

    # --- DIREITA: composicao 100% empilhada (visivel em todo N) ---
    total = h2d + dunn + silh + dbk
    p_h2d = 100 * h2d / total
    p_dunn = 100 * dunn / total
    p_silh = 100 * silh / total
    p_db = 100 * dbk / total
    x = np.arange(len(N))
    axR.bar(x, p_dunn, color=C_DUNN, label="Dunn")
    axR.bar(x, p_silh, bottom=p_dunn, color=C_SILH, label="Silhueta")
    axR.bar(x, p_h2d, bottom=p_dunn + p_silh, color=C_H2D, label="H2D (copia)")
    axR.bar(x, p_db, bottom=p_dunn + p_silh + p_h2d, color=C_DB, label="Davies-Bouldin")
    axR.set_xticks(x)
    axR.set_xticklabels([f"{n//1000}k" if n >= 1000 else str(n) for n in N], rotation=45)
    axR.set_xlabel("Numero de Pontos (N)")
    axR.set_ylabel("Porcentagem do tempo de GPU (%)")
    axR.set_title("Composicao do tempo — 100% empilhado")
    axR.legend(loc="lower center", ncol=2, fontsize=8)
    axR.set_ylim(0, 113)  # folga acima dos 100% para o callout (evita corte na borda)
    # callout do percentual Dunn+Silhueta no maior N: acima das barras, com seta,
    # ancorado para o interior para nunca cortar na borda direita
    i = len(N) - 1
    axR.annotate(f"Dunn + Silhueta\n$\\approx$ {p_dunn[i] + p_silh[i]:.1f}% do tempo",
                 xy=(x[i], 100), xytext=(x[i] - 2.6, 108),
                 ha="center", va="center", fontsize=8.5, fontweight="bold",
                 color="#1f3a5f",
                 arrowprops=dict(arrowstyle="->", color="#1f3a5f", lw=1.2),
                 annotation_clip=False)

    plt.tight_layout()
    plt.savefig("bench_breakdown.png", dpi=150)
    plt.close()
    print("[OK] bench_breakdown.png")


def main():
    df = carregar(CSV)
    grafico_tempo(df)
    grafico_speedup(df)
    grafico_breakdown(df)
    print("\nGraficos gerados a partir de", CSV)


if __name__ == "__main__":
    main()
