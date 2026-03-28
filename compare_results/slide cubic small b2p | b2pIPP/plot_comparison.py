import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import json
import glob
import os

# Physical constants
radius = 0.1
c0 = 343.8

# Toggle error subplot
plot_error = False

def analytical_radiation_factor(freqvec):
    k = 2 * np.pi * freqvec / c0
    Z = (1 - 2 * special.jv(1, 2*k*radius) / (2*k*radius)
         + 1j * 2 * special.struve(1, 2*k*radius) / (2*k*radius))
    return Z

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    freqvec = np.array(data['frequencies'])
    if 'Z_center' in data:
        Z_real = np.array(data['Z_center']['real'])
        Z_imag = np.array(data['Z_center']['imag'])
    elif 'p_values' in data:
        Z_real = np.array(data['p_values']['real'])
        Z_imag = np.array(data['p_values']['imag'])
    else:
        raise ValueError(f"Unknown data format in {filepath}")
    label = os.path.basename(filepath).replace('.json', '')
    if 'ope' in data:
        label = f"{data.get('geometry1','')}/{data.get('geometry2','')} {data['ope']} P{data.get('dimP','?')}Q{data.get('dimQ','?')} lc={data.get('lc','?')}"
    return freqvec, Z_real, Z_imag, label

def load_comsol_txt(filepath):
    """Load a COMSOL .txt export (header lines starting with %, then freq/value columns)."""
    freq_list, val_list = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            freq_list.append(float(parts[0]))
            val_list.append(float(parts[1]))
    label = os.path.basename(filepath).replace('.txt', '') + ' (COMSOL)'
    return np.array(freq_list), np.array(val_list), label


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_files = sorted(glob.glob(os.path.join(script_dir, "*.json")))
    txt_files = sorted(glob.glob(os.path.join(script_dir, "*.txt")))

    if not json_files and not txt_files:
        print("No .json or .txt files found in", script_dir)
        exit()

    if plot_error:
        fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(16, 9),
                                          gridspec_kw={'height_ratios': [3, 1]},
                                          sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(16, 9))

    # Analytical curve
    freqvec_ana = np.arange(80, 2001, 20)
    Z_ana = analytical_radiation_factor(freqvec_ana)
    ax.plot(freqvec_ana, Z_ana.real, 'k--', linewidth=2, label='Analytical')

    # Plot each JSON file
    for fpath in json_files:
        freqvec, Z_real, Z_imag, label = load_json(fpath)
        line, = ax.plot(freqvec, Z_real, label=label)
        if plot_error:
            Z_ref = analytical_radiation_factor(freqvec).real
            error = np.abs(Z_real - Z_ref)
            ax_err.plot(freqvec, error, color=line.get_color(), label=label)

    # Plot each COMSOL .txt file
    for fpath in txt_files:
        freqvec, values, label = load_comsol_txt(fpath)
        line, = ax.plot(freqvec, values, '--', label=label)
        if plot_error:
            Z_ref = analytical_radiation_factor(freqvec).real
            error = np.abs(values - Z_ref)
            ax_err.plot(freqvec, error, '--', color=line.get_color(), label=label)

    ax.set_ylabel(r'Re($\sigma$)')
    ax.set_title('Radiation factor comparison')
    ax.set_ylim(0, 1.25)
    ax.legend()
    ax.grid(True)

    if plot_error:
        ax_err.set_xlabel('Frequency (Hz)')
        ax_err.set_ylabel('Pointwise error')
        ax_err.set_yscale('log')
        ax_err.legend()
        ax_err.grid(True)
    else:
        ax.set_xlabel('Frequency (Hz)')

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "comparison.png"), dpi=150)
    plt.show()
