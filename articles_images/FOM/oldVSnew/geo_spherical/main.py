import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from pathlib import Path
from articles_images.plot_lib import (compute_analytical_radiation_factor, plot_analytical_result_sigma, import_radiation_factor, import_COMSOL_result,
                                      plot_new_old_comsol, plot_new_old_comsol_woErr,
                                      radius, c0)
plt.rcParams.update({
    "text.usetex": True
})


case = "small"
if case == "large":
    lc = 2.5e-2
elif case == "small":
    lc = 1e-2

s1 = f"spherical_{case}_b2p_tang_freqDep_{lc}_4_4.json"
file_path1 = Path(__file__).parent / f"{s1}"
s2 = f"spherical_{case}_b2p_tang_{lc}_4_4.json"
file_path2 = Path(__file__).parent / f"{s2}"

s3 = f"{case}_spherical_sphericalABC.txt"
file_path3 = Path(__file__).parent / f"{s3}"

freqvec1, z_center1 = import_radiation_factor(file_path1)
freqvec2, z_center2 = import_radiation_factor(file_path2)
freqvec3, z_center3 = import_COMSOL_result(file_path3)

Z_analytical = compute_analytical_radiation_factor(freqvec1, radius)

freq_zcent1 = [freqvec1, z_center1]
freq_zcent2 = [freqvec2, z_center2]
freq_zcent3 = [freqvec3, z_center3]
wo_err = True
if wo_err:
    plot_new_old_comsol_woErr(freq_zcent1 ,freq_zcent2, freq_zcent3)
else:
    plot_new_old_comsol(freq_zcent1 ,freq_zcent2, freq_zcent3)

plt.tight_layout()
if wo_err:
    save_file = f"FEniCSx_OldVSNewSpherical_{case}_woErr.pdf"
else:
    save_file = f"FEniCSx_OldVSNewSpherical_{case}.pdf"
save_path = Path(__file__).parent / save_file
plt.savefig(save_path)
if __name__ == "__out__":
    print(f"PDF geo spherical saved : {save_file}")
    plt.show()
if __name__ == "__main__":
    plt.show()
