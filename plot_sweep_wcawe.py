"""
Plot WCAWE sweep results for B2p_tang_ipp across geometries.

Usage:
    python plot_sweep_wcawe.py                           # plot all 3 geometries
    python plot_sweep_wcawe.py spherical                 # plot one geometry
    python plot_sweep_wcawe.py --N 5 10 20 50 80 100     # select specific N values
    python plot_sweep_wcawe.py --convergence             # plot convergence (error vs N)
    python plot_sweep_wcawe.py --cpu                     # plot CPU time vs N
"""

import numpy as np
import json
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

BASE_DIR = Path(__file__).parent / "raw_results"

# ── Configuration (must match sweep script) ──
STR_OPE = "b2p_tang_ipp"
DIM_P   = 2
DIM_Q   = 2
F0      = 1000
LC_MAP  = {
    'spherical':        1.5e-2,
    'curvedcubic':      1.5e-2,
    'new_broken_cubic': 1.5e-2,
}
GEO2 = 'small'


GEOMETRY_LABELS = {
    'spherical':        'Spherical',
    'curvedcubic':      'Curved cubic',
    'new_broken_cubic': 'Broken cubic',
}


def load_fom(geometry1, lc):
    fname = BASE_DIR / "FOM" / f"{geometry1}_{GEO2}_{STR_OPE}_{lc}_{DIM_P}_{DIM_Q}.json"
    with open(fname) as f:
        data = json.load(f)
    return np.array(data['frequencies']), np.array(data['Z_center']['real'])


def load_rom(geometry1, lc, N):
    fname = (BASE_DIR / "MOR" / "WCAWE" /
             f"{geometry1}_{GEO2}_{STR_OPE}_{lc}_{DIM_P}_{DIM_Q}_{N}_{F0}Hz.json")
    if not fname.exists():
        return None
    with open(fname) as f:
        data = json.load(f)
    return data


def smoothing(error, freqvec, f0):
    """Cumulative smoothed error metric (from plotWCAWE.py)."""
    f0_index = np.argmin(np.abs(freqvec - f0))
    error_metric_left  = []
    error_metric_right = [error[f0_index]]

    j = 1
    for _ in range(f0_index + 1, len(error)):
        sum_error = sum(10**(error[f0_index + ii] / 10) for ii in range(j + 1))
        error_metric_right.append(10 * np.log10(sum_error / (j + 1)))
        j += 1

    j = 1
    for _ in range(f0_index):
        sum_error = sum(10**(error[f0_index - ii] / 10) for ii in range(j + 1))
        error_metric_left.append(10 * np.log10(sum_error / (j + 1)))
        j += 1

    error_metric_left.reverse()
    return np.concatenate([error_metric_left, error_metric_right])


def intersect_on_common_freqs(freqvec_fom, z_fom, freqvec_rom, z_rom):
    """Restrict FOM and ROM data to their common frequencies."""
    common = np.intersect1d(freqvec_fom, freqvec_rom)
    idx_fom = np.isin(freqvec_fom, common)
    idx_rom = np.isin(freqvec_rom, common)
    return common, z_fom[idx_fom], z_rom[idx_rom]


def compute_global_error(freqvec_fom, z_fom, freqvec_rom, z_rom):
    """L2 relative error over the common frequency band."""
    _, z_f, z_r = intersect_on_common_freqs(freqvec_fom, z_fom, freqvec_rom, z_rom)
    return np.sqrt(np.sum((z_r - z_f)**2) / np.sum(z_f**2))


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 1: Radiation factor comparison (FOM vs ROM for selected N values)
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(geometries, N_list):
    n_geo = len(geometries)
    fig, axes = plt.subplots(n_geo, 1, figsize=(12, 5 * n_geo), squeeze=False)

    for i, geo in enumerate(geometries):
        ax = axes[i, 0]
        lc = LC_MAP[geo]
        freqvec, z_fom = load_fom(geo, lc)

        ax.plot(freqvec, z_fom, 'k-', linewidth=2, label='FOM')

        for N in N_list:
            rom_data = load_rom(geo, lc, N)
            if rom_data is None:
                print(f"  [skip] {geo} N={N}: no data")
                continue
            freqvec_rom = np.array(rom_data['frequencies'])
            z_rom = np.array(rom_data['Z_center']['real'])
            ax.plot(freqvec_rom, z_rom, label=f'N = {N}', alpha=0.8)

        ax.set_title(f'{GEOMETRY_LABELS[geo]} — B2p_tang_ipp (lc={lc})', fontsize=14)
        ax.set_ylabel('Radiation coefficient', fontsize=12)
        ax.set_ylim(0, 1.3)
        ax.legend(loc='upper left', fontsize=8, ncol=4)
        ax.grid(linestyle='--')

    axes[-1, 0].set_xlabel('Frequency [Hz]', fontsize=14)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 2: Radiation factor + smoothed error (2 subplots per geometry)
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison_with_error(geometries, N_list):
    n_geo = len(geometries)
    fig, axes = plt.subplots(n_geo, 2, figsize=(16, 5 * n_geo), squeeze=False)

    for i, geo in enumerate(geometries):
        ax1, ax2 = axes[i]
        lc = LC_MAP[geo]
        freqvec, z_fom = load_fom(geo, lc)

        ax1.plot(freqvec, z_fom, 'k-', linewidth=2, label='FOM')

        for N in N_list:
            rom_data = load_rom(geo, lc, N)
            if rom_data is None:
                continue
            freqvec_rom = np.array(rom_data['frequencies'])
            z_rom = np.array(rom_data['Z_center']['real'])

            ax1.plot(freqvec_rom, z_rom, label=f'N={N}', alpha=0.8)

            common, z_fom_c, z_rom_c = intersect_on_common_freqs(
                freqvec, z_fom, freqvec_rom, z_rom)
            error = np.abs(z_rom_c - z_fom_c)**2
            error_smooth = smoothing(error, common, F0)
            ax2.plot(common, error_smooth, label=f'N={N}', alpha=0.8)

        ax1.set_title(f'{GEOMETRY_LABELS[geo]}', fontsize=14)
        ax1.set_ylabel('Radiation coefficient', fontsize=12)
        ax1.set_ylim(0, 1.3)
        ax1.legend(loc='upper left', fontsize=8, ncol=3)
        ax1.grid(linestyle='--')

        ax2.set_title(f'{GEOMETRY_LABELS[geo]} — Smoothed error', fontsize=14)
        ax2.set_ylabel('Error (log scale)', fontsize=12)
        ax2.set_yscale('log')
        ax2.legend(loc='upper right', fontsize=8, ncol=3)
        ax2.grid(which='major', linestyle='--')
        ax2.minorticks_off()

    axes[-1, 0].set_xlabel('Frequency [Hz]', fontsize=14)
    axes[-1, 1].set_xlabel('Frequency [Hz]', fontsize=14)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 3: Convergence — global error vs N
# ─────────────────────────────────────────────────────────────────────────────
def plot_convergence(geometries, N_range=range(1, 101)):
    fig, ax = plt.subplots(figsize=(10, 6))

    for geo in geometries:
        lc = LC_MAP[geo]
        freqvec, z_fom = load_fom(geo, lc)

        Ns, errors = [], []
        for N in N_range:
            rom_data = load_rom(geo, lc, N)
            if rom_data is None:
                continue
            freqvec_rom = np.array(rom_data['frequencies'])
            z_rom = np.array(rom_data['Z_center']['real'])
            Ns.append(N)
            errors.append(compute_global_error(freqvec, z_fom, freqvec_rom, z_rom))

        ax.semilogy(Ns, errors, 'o-', markersize=3, label=GEOMETRY_LABELS[geo])

    ax.set_xlabel('Number of WCAWE vectors N', fontsize=14)
    ax.set_ylabel('Relative L2 error', fontsize=14)
    ax.set_title('WCAWE convergence — B2p_tang_ipp', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(which='both', linestyle='--')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 4: CPU time vs N
# ─────────────────────────────────────────────────────────────────────────────
def plot_cpu_time(geometries, N_range=range(1, 101)):
    fig, ax = plt.subplots(figsize=(10, 6))

    for geo in geometries:
        lc = LC_MAP[geo]
        Ns, cpu_times = [], []
        for N in N_range:
            rom_data = load_rom(geo, lc, N)
            if rom_data is None:
                continue
            Ns.append(N)
            cpu_times.append(rom_data['CPU_time']['total'])

        ax.plot(Ns, cpu_times, 'o-', markersize=3, label=GEOMETRY_LABELS[geo])

    ax.set_xlabel('Number of WCAWE vectors N', fontsize=14)
    ax.set_ylabel('Total CPU time [s]', fontsize=14)
    ax.set_title('WCAWE CPU time — B2p_tang_ipp', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(linestyle='--')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 5: CPU time breakdown (stacked bar: derivatives, building, splitting, solving)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cpu_breakdownV2(geometries, N_range=range(1, 101)):
    n_geo = len(geometries)
    fig, axes = plt.subplots(1, n_geo, figsize=(6 * n_geo, 5), squeeze=False)

    for i, geo in enumerate(geometries):
        ax = axes[0, i]
        lc = LC_MAP[geo]
        Ns = []
        t_deriv, t_build, t_split, t_solve = [], [], [], []

        for N in N_range:
            rom_data = load_rom(geo, lc, N)
            if rom_data is None:
                continue
            cpu = rom_data['CPU_time']
            Ns.append(N)
            t_deriv.append(cpu['derivatives'])
            t_build.append(cpu['buildingVn'])
            t_split.append(cpu['spliting_Vn'])
            t_solve.append(cpu['solvingMOR'])

        Ns = np.array(Ns)
        t_deriv = np.array(t_deriv)
        t_build = np.array(t_build)
        t_split = np.array(t_split)
        t_solve = np.array(t_solve)

        ax.bar(Ns, t_deriv, label='Derivatives', width=0.8)
        ax.bar(Ns, t_build, bottom=t_deriv, label='Building Vn', width=0.8)
        ax.bar(Ns, t_split, bottom=t_deriv + t_build,
               label='Splitting Vn', width=0.8)
        ax.bar(Ns, t_solve, bottom=t_deriv + t_build + t_split,
               label='Solving ROM', width=0.8)

        ax.set_title(GEOMETRY_LABELS[geo], fontsize=14)
        ax.set_xlabel('N', fontsize=12)
        ax.set_ylabel('CPU time [s]', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--')

    fig.suptitle('CPU breakdown — B2p_tang_ipp', fontsize=14)
    fig.tight_layout()
    return fig

def plot_cpu_breakdown(geometries, N_range=range(1, 101)):
    """
    CPU breakdown using stacked area plot (JCP publication style).
    """

    n_geo = len(geometries)

    # --- Aspect ratio: wide and compact (paper column-friendly) ---
    fig, axes = plt.subplots(1, n_geo, figsize=(5.5 * n_geo, 3.2), squeeze=False)

    for i, geo in enumerate(geometries):
        ax = axes[0, i]
        lc = LC_MAP[geo]

        Ns = []
        t_deriv, t_build, t_split, t_solve = [], [], [], []

        for N in N_range:
            rom_data = load_rom(geo, lc, N)
            if rom_data is None:
                continue

            cpu = rom_data['CPU_time']
            Ns.append(N)
            t_deriv.append(cpu['derivatives'])
            t_build.append(cpu['buildingVn'])
            t_split.append(cpu['spliting_Vn'])
            t_solve.append(cpu['solvingMOR'])

        Ns = np.array(Ns)
        t_deriv = np.array(t_deriv)
        t_build = np.array(t_build)
        t_split = np.array(t_split)
        t_solve = np.array(t_solve)

        # --- Stacked area (clean, no edges) ---
        ax.stackplot(
            Ns,
            t_deriv,
            t_build,
            t_split,
            t_solve,
            labels=[
                r'Derivatives',
                r'Build $V_n$',
                r'Split $V_n$',
                r'Solve ROM'
            ],
            alpha=0.9,
            linewidth=0.0
        )

        # --- Titles & labels (LaTeX style) ---
        ax.set_title(rf'{GEOMETRY_LABELS[geo]}')
        ax.set_xlabel(r'Number of WCAWE vectors $N$')
        ax.set_ylabel(r'CPU time [s]')

        # --- Limits ---
        ax.set_xlim(Ns.min(), Ns.max())

        # --- Light horizontal guide only ---
        ax.yaxis.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

        # --- Clean spines (JCP style) ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # --- Legend only once ---
        if i == 0:
            ax.legend(loc='upper left')

    # --- Tight layout with small padding ---
    fig.tight_layout(pad=1.0)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot WCAWE sweep results")
    parser.add_argument('geometries', nargs='*',
                        default=['spherical', 'curvedcubic', 'new_broken_cubic'],
                        help='Geometries to plot')
    parser.add_argument('--N', nargs='+', type=int,
                        default=[5, 10, 20, 30, 50, 80, 100],
                        help='N values for comparison plots')
    parser.add_argument('--convergence', action='store_true',
                        help='Plot convergence (error vs N)')
    parser.add_argument('--cpu', action='store_true',
                        help='Plot CPU time vs N')
    parser.add_argument('--with-error', action='store_true',
                        help='Plot comparison with smoothed error panels')
    parser.add_argument('--cpu-breakdown', action='store_true',
                        help='Plot CPU time breakdown (stacked bar)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save figures to this prefix (e.g. "results_")')

    args = parser.parse_args()

    # Default: show comparison if no specific plot type requested
    show_comparison = not (args.convergence or args.cpu
                           or args.with_error or args.cpu_breakdown)

    figs = []

    if show_comparison:
        figs.append(('comparison', plot_comparison(args.geometries, args.N)))

    if args.with_error:
        figs.append(('comparison_error', plot_comparison_with_error(args.geometries, args.N)))

    if args.convergence:
        figs.append(('convergence', plot_convergence(args.geometries)))

    if args.cpu:
        figs.append(('cpu_time', plot_cpu_time(args.geometries)))

    if args.cpu_breakdown:
        figs.append(('cpu_breakdown', plot_cpu_breakdown(args.geometries)))

    if args.save:
        for name, fig in figs:
            fname = f"{args.save}{name}.pdf"
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"Saved: {fname}")

    plt.show()
