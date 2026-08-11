"""
Compare experimental speedline data with computed results from nasa_hecc.py
This is just a plotting script for the design map in the paper. To be deleted
"""

from adet.tools.plotting import setup_mpl

import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

# Load speedline data from data directory
data_dir = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / 'data'
    / 'opencases'
    / 'nasa_hecc'
)


# plt.style.use('dark_background')
setup_mpl(
    {
        'font.family': 'serif',
        'text.usetex': True,
        'font.size': 19,
    }
)


def load_csv_speedlines(csv_path):
    """Load speedline data from CSV file, grouping by mean speed."""
    speedlines = {}

    with open(csv_path, newline='\n') as data:
        reader = csv.reader(data)
        keys = next(reader)
        curr_speedline_data = {k: [] for k in keys}

        for line in reader:
            if not line or not line[0].strip():
                if curr_speedline_data[keys[0]]:
                    mean_speed = np.mean(
                        [float(v) for v in curr_speedline_data[keys[0]]]
                    )
                    speedlines[round(mean_speed)] = curr_speedline_data

                    curr_speedline_data = {k: [] for k in keys}
                continue

            for i, k in enumerate(keys):
                curr_speedline_data[k].append(float(line[i]))

    return speedlines


def extract_experimental_data():
    """Extract experimental data from CSV file."""
    exp_data = {}

    csv_path = data_dir / 'vaneless_short_data.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Speedline data file not found: {csv_path}')

    speedlines = load_csv_speedlines(csv_path)

    for rpm_rounded, speedline_data in speedlines.items():
        exp_data[f'{rpm_rounded}'] = {
            'massflows': np.array(speedline_data['MDOTC']) / 2.2,
            'pratios': np.array(speedline_data['TPR30']),
            'etas': np.array(speedline_data['ETA30']) * 100,
            'ttratio': np.array(speedline_data['TTR30']),
        }

    return exp_data


def load_computed_data(json_path):
    """Load computed data from JSON file"""
    if not json_path.exists():
        print(f'Computed data file not found: {json_path}')
        print(
            'Please run nasa_hecc.py with RUN_SPEEDLINES=True to generate computed data'
        )
        return None

    with open(json_path, 'r') as f:
        return json.load(f)


def plot_comparison(exp_data, comp_data):
    """Create comparison plots for experimental vs computed data"""
    if comp_data is None:
        print('Cannot create comparison plots without computed data')
        return None, None

    # Sort by RPM for consistent plotting
    rpms = sorted(set(exp_data.keys()) & set(comp_data.keys()), key=float)

    # Create figure 1: Pressure ratio comparison
    fig1, ax = plt.subplots(figsize=(10, 9))
    colors = plt.get_cmap('viridis')(np.linspace(0, 0.8, len(rpms)))

    for rpm, color in zip(rpms, colors):
        exp = exp_data[rpm]
        exp_mf = exp['massflows']
        ax.plot(
            exp_mf,
            exp['pratios'],
            'o',
            color=color,
            label=f'{round(float(rpm) / 1000)}k rpm (exp.)',
            markersize=8,
            linewidth=2.5,
            alpha=0.7,
        )

        if rpm in comp_data:
            comp = comp_data[rpm]
            comp_mf = np.array(comp['massflows'])
            comp_pr = np.array(comp['pratios'])
            ax.plot(
                comp_mf,
                comp_pr,
                '-',
                color=color,
                label=f'{round(float(rpm) / 1000)}k rpm (comp.)',
                linewidth=2.5,
                alpha=0.7,
            )

            # Add 2% error band
            error_lower = comp_pr * 0.98
            error_upper = comp_pr * 1.02
            ax.fill_between(
                comp_mf,
                error_lower,
                error_upper,
                color=color,
                alpha=0.2,
            )
            ax.tick_params('both')

    ax.set_xlabel(r'$\dot{m}$ [$\mathrm{kg/s}$]', fontsize=24)
    ax.set_ylabel(r'$\Pi_{tt}$ [$\mathrm{−}$]', fontsize=24)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(
        'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\presentations'
        '\\images\\HECC_pratios.svg'
    )

    # Create figure 2: Efficiency comparison with separate subplot for each speedline
    num_speedlines = len(rpms)
    num_cols = 2
    num_rows = (num_speedlines + num_cols - 1) // num_cols
    fig2, axs = plt.subplots(
        num_rows, num_cols, figsize=(10, 4 * num_rows), sharey=True, sharex=True
    )
    axs = axs.flatten() if num_speedlines > 1 else [axs]

    for idx, rpm in enumerate(rpms):
        ax = axs[idx]
        exp = exp_data[rpm]
        exp_mf = exp['massflows']
        color = colors[idx]

        # Plot experimental data
        ax.plot(
            exp_mf,
            exp['etas'],
            'o',
            color=color,
            label=f'{round(float(rpm) / 1000)}k rpm (exp.)',
            markersize=8,
            linewidth=2.5,
            alpha=0.8,
        )

        # Plot computed data with error bars
        if rpm in comp_data:
            comp = comp_data[rpm]
            comp_mf = np.array(comp['massflows'])
            comp_eta = np.array([100 * eta for eta in comp['etas']])

            # Plot computed line
            ax.plot(
                comp_mf,
                comp_eta,
                '-',
                color=color,
                label=f'{round(float(rpm) / 1000)}k rpm (comp.)',
                linewidth=2.5,
                alpha=0.8,
            )

            # Add ±2% error band
            error_lower = comp_eta * 0.98
            error_upper = comp_eta * 1.02
            ax.fill_between(
                comp_mf,
                error_lower,
                error_upper,
                color=color,
                alpha=0.2,
                # label=r'$\pm$ 2% error',
            )

        ax.set_ylim(80, 95)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.tick_params('both')

    # Remove unused axes
    for idx in range(num_speedlines, len(axs)):
        fig2.delaxes(axs[idx])
    axs = axs[:num_speedlines]

    # Set x-labels on bottom row
    for idx in range(max(0, num_speedlines - num_cols), num_speedlines):
        axs[idx].set_xlabel(r'$\dot{m}$ [$\mathrm{kg/s}$]', fontsize=24)

    for idx in range(0, len(axs), num_cols):
        axs[idx].set_ylabel(r'$\eta_{tt}$ [$\mathrm{\%}$]', fontsize=24)

    fig2.tight_layout()
    fig2.savefig(
        'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\'
        '\\presentations\\images\\HECC_efficiencies.svg'
    )

    # Create figure 3: Total temperature ratio comparison
    fig3, ax = plt.subplots(figsize=(10, 9))

    for rpm, color in zip(rpms, colors):
        exp = exp_data[rpm]
        exp_mf = exp['massflows']
        ax.plot(
            exp_mf,
            exp['ttratio'],
            'o',
            color=color,
            label=f'{round(float(rpm) / 1000)}k rpm (exp.)',
            markersize=8,
            linewidth=2.5,
            alpha=0.7,
        )

        if rpm in comp_data:
            comp = comp_data[rpm]
            comp_mf = np.array(comp['massflows'])
            comp_ttr = np.array(comp['ttratio'])
            ax.plot(
                comp_mf,
                comp_ttr,
                '-',
                color=color,
                label=f'{round(float(rpm) / 1000)}k rpm (comp.)',
                linewidth=2.5,
                alpha=0.7,
            )

            # Add ±2% error band
            error_lower = comp_ttr * 0.98
            error_upper = comp_ttr * 1.02
            ax.fill_between(
                comp_mf,
                error_lower,
                error_upper,
                color=color,
                alpha=0.2,
            )
            ax.tick_params('both')

    ax.set_xlabel(r'$\dot{m}$ [$\mathrm{kg/s}$]', fontsize=24)
    ax.set_ylabel(r'$\tau_{tt}$ [$\mathrm{−}$]', fontsize=24)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(
        'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\'
        '\\presentations\\images\\HECC_temperature_ratios.svg'
    )
    return fig1, fig2, fig3


if __name__ == '__main__':
    # Load experimental data
    print('Loading experimental data...')
    exp_data = extract_experimental_data()
    print(f'Found {len(exp_data)} experimental speedlines')

    # Load computed data
    json_path = data_dir / 'computed_speedline_data.json'
    print(f'Loading computed data from {json_path}...')
    comp_data = load_computed_data(json_path)

    if comp_data:
        print(f'Found {len(comp_data)} computed speedlines')
        print('\nComputed speedlines:', list(comp_data.keys()))

    # Create comparison plots
    fig1, fig2, fig3 = plot_comparison(exp_data, comp_data)
