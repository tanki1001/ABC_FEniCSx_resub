from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from operators_POO import compute_analytical_radiation_factor


import json
folder = 'raw_results'
folder = Path(folder)
json_files = sorted(folder.glob('*.json'))
lc_b2p = []
rmse_b2p = []

lc_b2p_tang = []
rmse_b2p_tang = []

geometry1 = "spherical"



for fp in json_files:
    with open(fp, 'r') as f:
        data = json.load(f)
    if geometry1 != data['geometry1']:
        continue
    lc = data['lc']
    Z_center_real = np.array(data['Z_center']['real'])
    freqvec = np.array(data['frequencies'])
    Z_ana = compute_analytical_radiation_factor(freqvec, radius=0.1)
    #rmse = np.sqrt(np.mean(np.abs(Z_center_real - Z_ana.real)**2))
    rmse = np.linalg.norm(Z_center_real - Z_ana.real) / np.linalg.norm(Z_ana.real)    


    if "tang_freqDep" in data["ope"] :
        lc_b2p_tang.append(lc)
        rmse_b2p_tang.append(rmse)
    else :
        lc_b2p.append(lc)
        rmse_b2p.append(rmse)
    
lc_b2p_tang, rmse_b2p_tang = zip(*sorted(zip(lc_b2p_tang, rmse_b2p_tang)))
#lc_b2p, rmse_b2p = zip(*sorted(zip(lc_b2p, rmse_b2p)))

fig, ax1 = plt.subplots()
ax1.plot([1/lc for lc in lc_b2p_tang], rmse_b2p_tang, color = "blue")
#ax2 = ax1.twinx()
#ax2.plot([1/lc for lc in lc_b2p], rmse_b2p, color = "orange")

ax1.set_yscale('log')
#ax2.set_yscale('log')

ax1.set_xscale('log')
#ax2.set_xscale('log')

#fig1, ax= plt.subplots()
#ax.plot([1/lc for lc in lc_b2p_tang], rmse_b2p_tang, color = "blue")
#ax.plot([1/lc for lc in lc_b2p], rmse_b2p, color = "orange")
#ax.set_yscale('log')
#ax.set_xscale('log')
plt.show()