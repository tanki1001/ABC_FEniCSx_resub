import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from pathlib import Path
from articles_images.plot_lib import (compute_analytical_radiation_factor, plot_analytical_result_sigma, import_radiation_factor, import_COMSOL_result,
                                      plot_new_old_comsol, plot_new_old_comsol_woErr, plot_MOR,
                                      radius, c0)
plt.rcParams.update({
    "text.usetex": True
})


s1 = "new_broken_cubic_small_b2p_tang_freqDep_0.01_4_4.json"
file_path1 = Path(__file__).parent / f"{s1}"
s2 = "new_broken_cubic_small_b2p_tang_0.01_4_4.json"
file_path2 = Path(__file__).parent / f"{s2}"

freqvec1, z_center1 = import_radiation_factor(file_path1)
freqvec2, z_center2 = import_radiation_factor(file_path2)
f0 = 1000
N = 5
s_ROM1 = f"new_broken_cubic_small_b2p_tang_freqDep_0.01_4_4_{N}_{f0}Hz.json"
file_path_ROM1 = Path(__file__).parent / f"{s_ROM1}"
s_ROM2 = f"new_broken_cubic_small_b2p_tang_0.01_4_4_{N}_{f0}Hz.json"
file_path_ROM2 = Path(__file__).parent / f"{s_ROM2}"

freqvec_ROM1, z_center_ROM1 = import_radiation_factor(file_path_ROM1)
freqvec_ROM2, z_center_ROM2 = import_radiation_factor(file_path_ROM2)

freq_zcent1 = [freqvec1, z_center1, z_center_ROM1]
freq_zcent2 = [freqvec1, z_center2, z_center_ROM2]
plot_MOR([freq_zcent1, freq_zcent2], N)


plt.tight_layout()
save_path = Path(__file__).parent / f"MOR_new_broken_cubic_{N}_{f0}Hz.pdf"
plt.savefig(save_path)
plt.show()
