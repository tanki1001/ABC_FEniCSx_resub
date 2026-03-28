import numpy as np
from scipy import special
import json
import matplotlib.pyplot as plt



c0 = 343.8  # Speed of sound in air
radius = 0.1

def import_COMSOL_result(s):
    
    with open(s, "r") as f:
        frequency = list()
        results = list()
        for line in f:
            if "%" in line:
                # on saute la ligne
                continue
            data = line.split()
            frequency.append(data[0])
            results.append(data[1])
            frequency = [float(element) for element in frequency]
            results = [float(element) for element in results]
    frequency = np.array(frequency)
    results = np.array(results)

    return frequency, results

def import_radiation_factor(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    freqvec = np.array(data['frequencies'])
    Z_center_real = np.array(data['Z_center']['real'])
    return freqvec, Z_center_real

def import_CPU_time(filename):
    with open(filename) as f:
        data = json.load(f)
    return data["CPU_time"]

def compute_analytical_radiation_factor(freqvec, radius):
    k_output = 2 * np.pi * freqvec / c0
    Z_analytical = (1 - 2 * special.jv(1, 2 * k_output * radius) / (2 * k_output * radius)
                    + 1j * 2 * special.struve(1, 2 * k_output * radius) / (2 * k_output * radius))
    return Z_analytical


def plot_analytical_result_sigma(ax, freqvec, radius):
    Z_analytical = compute_analytical_radiation_factor(freqvec, radius)
    ax.plot(freqvec, Z_analytical.real, label='Ground truth', c='black', linewidth = 2)
    ax.legend()

def plot_new_old_comsol(freq_zcent1, freq_zcent2, freq_zcent3):
    freqvec1, z_center1 = freq_zcent1[0], freq_zcent1[1]
    freqvec2, z_center2 = freq_zcent2[0], freq_zcent2[1]
    freqvec3, z_center3 = freq_zcent3[0], freq_zcent3[1]

    Z_analytical = compute_analytical_radiation_factor(freqvec1, radius)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [2, 1]})
    
    plot_analytical_result_sigma(ax1, freqvec1, radius)
    #ax1.plot(freqvec1, z_center1.real, label=r'\textbf{Original ABC}', c = 'C3', linestyle = "-.", marker='o', markersize=5, linewidth = 2, markerfacecolor='white')
    #ax1.plot(freqvec2, z_center2.real, label=r'\textbf{Proposed ABC}', c = 'C2', linestyle = "--", marker='*', markersize=5, linewidth = 2)
    #ax1.plot(freqvec3, z_center3.real, label=r'\textbf{COMSOL ABC}',   c = 'C4', linestyle = ":", marker='*', markersize=5, linewidth = 1)
    ax1.plot(freqvec1, z_center1.real, label=r'Original ABC', c = 'C3', linestyle = "-.", linewidth = 2)
    ax1.plot(freqvec2, z_center2.real, label=r'Proposed ABC', c = 'C2', linestyle = "--", linewidth = 2)
    ax1.plot(freqvec3, z_center3.real, label=r'COMSOL ABC',   c = 'C4', linestyle = (0, (1, 1)), linewidth = 2)
    ax1.set_ylim(0, 1.7)
    #ax1.set_ylabel(r'\textbf{Radiation coefficient}', fontsize=14)
    ax1.set_ylabel(r'Radiation coefficient', fontsize=14)
    ax1.tick_params(labelsize=16)
    legend1 = ax1.legend(loc='upper right', frameon=True, fontsize = 12)
    #legend1.get_frame().set_edgecolor('black')  
    #legend1.get_frame().set_linewidth(1.5)  
    ax1.grid(linestyle=':')



    error_old    = np.abs(z_center1.real - Z_analytical.real)**2
    error_new    = np.abs(z_center2.real - Z_analytical.real)**2
    error_comsol = np.abs(z_center3 - Z_analytical.real)**2


    #ax2.plot(freqvec1, error_old, label=r'\textbf{Error Original ABC}', c='C3', marker='o', markevery = 10, markersize=10, linewidth = 4, markerfacecolor='white')
    #ax2.plot(freqvec2, error_new, label=r'\textbf{Error Proposed ABC}', c='C2', marker='*', markevery = 10, markersize=10, linewidth = 2)

    ax2.plot(freqvec1, error_old, label=r'Error Original ABC', c='C3', linewidth = 2)
    ax2.plot(freqvec2, error_new, label=r'Error Proposed ABC', c='C2', linewidth = 2)
    ax2.plot(freqvec3, error_comsol, label=r'Error COMSOL ABC', c='C4', linewidth = 2)
    ax2.set_ylim(2e-11, 0.5)
    ax2.set_yscale('log')
    #ax2.set_xlabel(r'\textbf{Frequency [Hz]}', fontsize=14)
    #ax2.set_ylabel(r'\textbf{Absolute error (log scale)}', fontsize=14)
    ax2.set_xlabel(r'Frequency [Hz]', fontsize=14)
    ax2.set_ylabel(r'Absolute error (log scale)', fontsize=14)
    legend2 = ax2.legend(loc='upper right', frameon=True, fontsize = 12)
    #legend2.get_frame().set_edgecolor('black')  
    #legend2.get_frame().set_linewidth(1.5)   
    ax2.grid(which='major', linestyle=':')
    ax2.minorticks_off()
    ax2.tick_params(labelsize=16)

def plot_new_old_comsol_woErr(freq_zcent1, freq_zcent2, freq_zcent3):
    freqvec1, z_center1 = freq_zcent1[0], freq_zcent1[1]
    freqvec2, z_center2 = freq_zcent2[0], freq_zcent2[1]
    freqvec3, z_center3 = freq_zcent3[0], freq_zcent3[1]

    Z_analytical = compute_analytical_radiation_factor(freqvec1, radius)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    
    plot_analytical_result_sigma(ax1, freqvec1, radius)
    #ax1.plot(freqvec1, z_center1.real, label=r'\textbf{Original ABC}', c = 'C3', linestyle = "-.", marker='o', markersize=5, linewidth = 2, markerfacecolor='white')
    #ax1.plot(freqvec2, z_center2.real, label=r'\textbf{Proposed ABC}', c = 'C2', linestyle = "--", marker='*', markersize=5, linewidth = 2)
    #ax1.plot(freqvec3, z_center3.real, label=r'\textbf{COMSOL ABC}',   c = 'C4', linestyle = ":", marker='*', markersize=5, linewidth = 1)
    ax1.plot(freqvec1, z_center1.real, label=r'Original ABC', c = 'C3', linestyle = "-.", linewidth = 2)
    ax1.plot(freqvec2, z_center2.real, label=r'Proposed ABC', c = 'C2', linestyle = "--", linewidth = 2)
    ax1.plot(freqvec3, z_center3.real, label=r'COMSOL ABC',   c = 'C4', linestyle = (0, (1, 1)), linewidth = 2)
    ax1.set_ylim(0, 1.7)
    #ax1.set_ylabel(r'\textbf{Radiation coefficient}', fontsize=14)
    ax1.set_ylabel(r'Radiation coefficient', fontsize=18)
    ax1.set_xlabel(r'Frequency [Hz]', fontsize=14)
    ax1.tick_params(labelsize=16)
    legend1 = ax1.legend(loc='upper center', frameon=True, ncol = 2, fontsize = 16)
    #legend1.get_frame().set_edgecolor('black')  
    #legend1.get_frame().set_linewidth(1.5)  
    ax1.grid(linestyle=':')


def smoothing(error, freqvec):
    error_metric_left  = []
    error_metric_right = []

    f0_index = np.where(freqvec == 1000)[0][0]

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


def plot_MOR(list_freq_zcent, N):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [2, 1]})

    for i in range(len(list_freq_zcent)):
        if i == 0 :
            label = "FOM - Original ABC"
            c = 'C3'
            linestyle = "-."
            label_MOR = 'Original ABC ' + f'N = {N}'

        if i == 1 :
            label = "FOM - Proposed  ABC"
            c = 'C2'
            linestyle = "--"
            label_MOR = 'Proposed ABC ' + f'N = {N}'
        freq_zcent = list_freq_zcent[i]
        freqvec, z_center_FOM, z_center_ROM = freq_zcent[0], freq_zcent[1], freq_zcent[2]  

        ax1.plot(freqvec, z_center_FOM.real, label=label, linestyle = linestyle, c = c, linewidth = 2)
        ax1.plot(freqvec, z_center_ROM.real, label='ROM - ' + label_MOR, c = c, linewidth = 2)
        ax1.set_ylim(0, 1.6)
        
        error  = np.abs(z_center_ROM.real - z_center_FOM.real)**2
        error_metric       = smoothing(error, freqvec)

        ax2.plot(freqvec, error_metric,  label='Error - ' + label_MOR,  c=c, linewidth = 2)

    

    
    ax1.set_ylabel(r'Radiation coefficient', fontsize=18)
    #ax1.set_xticklabels([])
    legend1 = ax1.legend(loc='upper center', frameon=True, ncol = 2, fontsize = 14)
    #legend1.get_frame().set_edgecolor('black')  
    #legend1.get_frame().set_linewidth(1.5)  
    ax1.grid(linestyle=':')
    ax1.tick_params(labelsize=16)

    ax2.set_ylim(2e-11, 1e-3)
    ax2.set_yscale('log')
    ax2.set_xticklabels([])
    ax2.set_xlabel(r'Frequency [Hz]', fontsize=16)
    ax2.set_ylabel(r'Error (log scale)', fontsize=18)
    legend2 = ax2.legend(loc='lower center', frameon=True, ncol=2, fontsize = 14)
    #legend2.get_frame().set_edgecolor('black')  
    #legend2.get_frame().set_linewidth(1.5)   
    ax2.grid(which='major', linestyle=':')
    ax2.minorticks_off()
    ax2.tick_params(labelsize=16)