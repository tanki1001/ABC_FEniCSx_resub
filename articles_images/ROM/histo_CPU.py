
import numpy as np
import matplotlib.pyplot as plt
from articles_images.plot_lib import import_CPU_time
from pathlib import Path
plt.rcParams.update({
    "text.usetex": True
})

# --- Données ---
#series_names = ["Frequency-dependent \n\n" +r"$N = 5$", "Frequency-dependent\n\n" +r"$N = 10$", "Frequency-dependent \n\n"+r"$N = 20$", "Frequency-independent \n\n"+r"$\tilde{N} = 10$", "Frequency-independent \n\n"+r"$\tilde{N} = 20$", "Frequency-independent \n\n"+r"$\tilde{N} = 40$"]
series_names = [r"$N = 5$", r"$N = 10$" + "\n\nFrequency-dependent", r"$N = 20$", r"$\tilde{N} = 10$", r"$\tilde{N} = 20$" + " \n\nFrequency-independent",r"$\tilde{N} = 40$"]
#actions = ["Computing derivatives", "WCAWE algorithm", r"$Assembling \mathbf{C}(\text{j}k)$", r"$Split V_n$"]
actions = ["Derivative computation", "WCAWE procedure", r"Assembling $\mathbf{C}$", r"Splitting of $V$ into $\tilde{V}$", "Computing reduced problem"]

filename = Path(__file__).parent / "new_broken_cubic_small_b2p_tang_freqDep_0.01_4_4_5_1000Hz.json"
CPU_time_freqDep_N_5 = import_CPU_time(filename)
filename = Path(__file__).parent / "new_broken_cubic_small_b2p_tang_freqDep_0.01_4_4_10_1000Hz.json"
CPU_time_freqDep_N_10 = import_CPU_time(filename)
filename = Path(__file__).parent / "new_broken_cubic_small_b2p_tang_freqDep_0.01_4_4_20_1000Hz.json"
CPU_time_freqDep_N_20 = import_CPU_time(filename)

filename = Path(__file__).parent / "new_broken_cubic_small_b2p_tang_0.01_4_4_5_1000Hz.json"
CPU_time_N_5 = import_CPU_time(filename)
filename = Path(__file__).parent / "new_broken_cubic_small_b2p_tang_0.01_4_4_10_1000Hz.json"
CPU_time_N_10 = import_CPU_time(filename)
filename = Path(__file__).parent / "new_broken_cubic_small_b2p_tang_0.01_4_4_20_1000Hz.json"
CPU_time_N_20 = import_CPU_time(filename)

# Données par série (en secondes)
print()
CPU_data = [
    [CPU_time_freqDep_N_5["derivatives"], CPU_time_freqDep_N_5["buildingVn"], CPU_time_freqDep_N_5["assembling_C"], CPU_time_freqDep_N_5["spliting_Vn"], CPU_time_freqDep_N_5["solvingMOR"]],   # Série A (votre exemple)
    [CPU_time_freqDep_N_10["derivatives"], CPU_time_freqDep_N_10["buildingVn"], CPU_time_freqDep_N_10["assembling_C"], CPU_time_freqDep_N_10["spliting_Vn"], CPU_time_freqDep_N_10["solvingMOR"]],   # Série B (exemple)
    [CPU_time_freqDep_N_20["derivatives"], CPU_time_freqDep_N_20["buildingVn"], CPU_time_freqDep_N_20["assembling_C"], CPU_time_freqDep_N_20["spliting_Vn"], CPU_time_freqDep_N_20["solvingMOR"]],   # Série C (exemple)
    [CPU_time_N_5["derivatives"], CPU_time_N_5["buildingVn"], CPU_time_N_5["assembling_C"], CPU_time_N_5["spliting_Vn"], CPU_time_N_5["solvingMOR"]],   # Série D (exemple)
    [CPU_time_N_10["derivatives"], CPU_time_N_10["buildingVn"], CPU_time_N_10["assembling_C"], CPU_time_N_10["spliting_Vn"], CPU_time_N_10["solvingMOR"]],   # Série D (exemple)
    [CPU_time_N_20["derivatives"], CPU_time_N_20["buildingVn"], CPU_time_N_20["assembling_C"], CPU_time_N_20["spliting_Vn"], CPU_time_N_20["solvingMOR"]],   # Série D (exemple)
]

data = np.array(CPU_data)  # plus pratique pour manipuler

# --- Construction du graphique ---
x = np.arange(len(series_names))  # positions sur l'axe des séries

colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]

fig, ax = plt.subplots(figsize=(16, 9))

# Première couche (Action 1)
bottom = np.zeros(len(series_names))

for i_action in range(len(actions)):
    ax.bar(
        x,
        data[:, i_action],
        bottom=bottom,
        width = 0.2,
        #label=actions[i_action],
        label=actions[i_action],
        color=colors[i_action],
        edgecolor="black"
    )
    # Mise à jour de la base pour l'empilement suivant
    bottom += data[:, i_action]


# --- Mise en forme ---
#ax.set_title("Durée (secondes) par action – Histogramme empilé", fontsize=14)
ax.set_ylabel("Computational time (s)", fontsize=18)
ax.tick_params(labelsize=16)

ax.set_xticks(x)
#ax.set_xticklabels(series_names, rotation=45, ha="right")
ax.set_xticklabels(series_names, fontsize=16)

ax.yaxis.grid(True, alpha=0.25)

# Séparation visuelle entre les 3 premières et 3 dernières séries
ax.axvline(
    x=2.5,
    color="black",
    linewidth=1.2,
    linestyle="--",
    alpha=0.7
)



# --- Légendes ---
handles, labels = ax.get_legend_handles_labels()
label_C = r"Assembling $\mathbf{C}$"

handle_C = None
if label_C in labels:
    idx_C = labels.index(label_C)
    handle_C = handles[idx_C]

# --- Légendes ---
label_split = r"Splitting of $V$ into $\tilde{V}$"

handle_split = None
if label_split in labels:
    idx_split = labels.index(label_split)
    handle_split = handles[idx_split]

handles_out = [h for h, l in zip(handles, labels) if (l != label_C and l != label_split)]
labels_out = [l for l in labels if (l != label_C and l != label_split)]

legend_out = ax.legend(
    handles_out,
    labels_out,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=18,
    columnspacing=1.5,
    handlelength=1.8
)
ax.add_artist(legend_out)

if handle_C is not None:
    legend_C = ax.legend(
        [handle_C],
        [label_C],
        loc="upper left",
        fontsize=18,
        frameon=True
    )
ax.add_artist(legend_C)

if handle_split is not None:
    legend_split = ax.legend(
        [handle_split],
        [label_split],
        loc="upper right",
        fontsize=18,
        frameon=True
    )

plt.tight_layout(rect=(0, 0, 1, 0.9))
save_path = Path(__file__).parent / f"histo_CPU.pdf"
plt.savefig(save_path, bbox_inches="tight", bbox_extra_artists=[legend_out])
plt.show()
