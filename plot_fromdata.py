from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from operators_POO import compute_analytical_radiation_factor
import json

lc = 0.025
filename1 = f'raw_results/spherical_small_b2p_freqDep_{lc}_4_4.json'

filename2 = f'raw_results/spherical_small_b2p_tang_freqDep_{lc}_4_4.json'

with open(filename1, 'r') as f:
        data1 = json.load(f)

with open(filename2, 'r') as f:
        data2 = json.load(f)

fig, ax = plt.subplots()
freqvec = np.array(data1['frequencies'])
Z_ana = compute_analytical_radiation_factor(freqvec, radius=0.1)
Z_b2p_freqDep = np.array(data1['Z_center']['real'])
Z_b2p_tang_freqDep = np.array(data2['Z_center']['real'])

ax.plot(freqvec, Z_ana, label='Analytical', color='black')
ax.plot(freqvec, Z_b2p_freqDep, label='B2p freqDep', linestyle='--', color='blue')
ax.plot(freqvec, Z_b2p_tang_freqDep, label='B2p tang freqDep', linestyle='--', color='orange')
ax.legend()
plt.show()