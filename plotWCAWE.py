from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from operators_POO import compute_analytical_radiation_factor
import json
from cycler import cycler

def smoothing(error, freqvec, f0):
    error_metric_left  = []
    error_metric_right = []

    f0_index = np.where(freqvec == f0)[0][0]

    error_metric_right.append(error[f0_index])

    j = 1 # index in the list of the right part 

    for _ in range(f0_index + 1, len(error)):
        sum_error = 0
        for ii in range(j + 1):
                #Ei = np.log10(error_newBSP_5[f0_index + ii])
                Ei = error[f0_index + ii]
                sum_error+=10**(Ei/10)
        value = 10*np.log10(1/(j + 1) * sum_error)
        error_metric_right.append(value)
        j+=1

    j = 1
    for _ in range(f0_index):
        sum_error = 0
        for ii in range(0, j + 1):
            Ei = (error[f0_index - ii])
            sum_error+=10**(Ei/10)
        #print(f'sum_error : {sum_error}')    
        value = 10*np.log10(1/(j+1) * sum_error)
        error_metric_left.append(value)
        j+=1
    #print(f'error_newBSP_5_metric_left : {error_newBSP_5_metric_left}')
    error_metric_left.reverse()
    error_metric = np.concatenate([error_metric_left, error_metric_right])

    return error_metric

def plot_MOR(freqvec, z_center_FOM, z_center_ROM, N, f0, ax1, ax2):


    label = f'N = {N}'
    #ax1.plot(freqvec, z_center_ROM.real, label=label, alpha = 0.3)
    ax1.plot(freqvec, z_center_ROM.real, label=label)

    ax1.set_ylim(0, 1.3)
    
    error  = np.abs(z_center_ROM.real - z_center_FOM.real)**2
    error_metric       = smoothing(error, freqvec, f0)

    #ax2.plot(freqvec, error_metric, label = label, alpha = 0.3)
    ax2.plot(freqvec, error_metric, label = label)

    

    
    ax1.set_ylabel(r'Radiation coefficient', fontsize=14)
    #ax1.set_xticklabels([])
    legend1 = ax1.legend(loc='upper left', frameon=True, fontsize = 5, ncol=3)  
    ax1.grid(linestyle='--')
    ax1.tick_params(labelsize=16)

    ax2.set_ylim(2e-11, 1e-3)
    ax2.set_yscale('log')
    ax2.set_xticklabels([])
    ax2.set_xlabel(r'Frequency [Hz]', fontsize=16)
    ax2.set_ylabel(r'Error (log scale)', fontsize=16)
    #legend2 = ax2.legend(loc='lower left', frameon=True, fontsize = 5, ncol=2)
    ax2.grid(which='major', linestyle='--')
    ax2.minorticks_off()
    ax2.tick_params(labelsize=16)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [2, 1]})

filenameFOM = f'new_broken_cubic_small_b2p_tang_0.01_4_4.json'
filenameFOM = Path(__file__).parent / f"raw_results/{filenameFOM}"
with open(filenameFOM, 'r') as f:
        dataFOM = json.load(f)
freqvec = np.array(dataFOM['frequencies'])
Z_b2p_freqDepFOM = np.array(dataFOM['Z_center']['real'])
ax1.plot(freqvec, Z_b2p_freqDepFOM, c ="black")


colors = plt.get_cmap("tab20").colors  # 20 couleurs
linestyles = ['-', '--', '-.', ':']

# Crée un cycler combinant couleur et style
#ax1.set_prop_cycle(cycler(color=colors) * cycler(linestyle=linestyles))
#ax2.set_prop_cycle(cycler(color=colors) * cycler(linestyle=linestyles))

N = 50
f0 = 1000
for N in [5, 10, 15, 20, 25, 30]:
    filenameROM = f'new_broken_cubic_small_b2p_tang_0.01_4_4_{N}_{f0}Hz.json'
    filenameROM = Path(__file__).parent / f"raw_results/MOR/{filenameROM}"

    with open(filenameROM, 'r') as f:
            dataROM = json.load(f)



    freqvec = np.array(dataROM['frequencies'])
    Z_b2p_freqDepROM = np.array(dataROM['Z_center']['real'])



    plot_MOR(freqvec=freqvec, z_center_FOM=Z_b2p_freqDepFOM, z_center_ROM=Z_b2p_freqDepROM, N = N, f0 = f0, ax1=ax1, ax2=ax2)

plt.show()

