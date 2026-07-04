"""
Compare experimental speedline data with computed results from nasa_hecc.py
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


setup_mpl(
    {
        'font.family': 'serif',
        'font.size': 10,
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
    fig1, ax = plt.subplots(figsize=(10, 5))
    colors = plt.get_cmap('viridis')(np.linspace(0, 0.8, len(rpms)))

    for rpm, color in zip(rpms, colors):
        exp = exp_data[rpm]
        exp_mf = exp['massflows']
        ax.plot(
            exp_mf,
            exp['pratios'],
            'o',
            color=color,
            label=f'{int(float(rpm) / 1000)}k rpm (exp.)',
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
                label=f'{int(float(rpm) / 1000)}k rpm (comp.)',
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
            ax.tick_params('both', labelsize=30)

    ax.set_xlabel(r'$\dot{m}$ [kg/s]')
    ax.set_ylabel(r'$\beta_{tt}$ [−]')

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.5)
    fig1.tight_layout()
    fig1.savefig(
        'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\latex'
        '\\gpps26_ADeT\\Images\\HECC_pratios.pdf'
    )

    # Create figure 2: Efficiency comparison with separate subplot for each speedline
    fig2, axs = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
    axs = axs.flatten()

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
            label=f'{int(float(rpm) / 1000)}k rpm (exp.)',
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
                label=f'{int(float(rpm) / 1000)}k rpm (comp.)',
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

        ax.set_ylim(75, 90)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.5)
        ax.tick_params('both', labelsize=30)

    axs[2].set_xlabel(r'$\dot{m}$ [kg/s]')
    axs[3].set_xlabel(r'$\dot{m}$ [kg/s]')
    axs[0].set_ylabel(r'$\eta_{tt}$ [\%]')
    axs[2].set_ylabel(r'$\eta_{tt}$ [\%]')

    fig2.tight_layout()
    fig2.savefig(
        'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\latex'
        '\\gpps26_ADeT\\Images\\HECC_efficiencies.pdf'
    )

    # Create figure 3: Total temperature ratio comparison
    fig3, axs = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
    axs = axs.flatten()

    for idx, rpm in enumerate(rpms):
        ax = axs[idx]
        exp = exp_data[rpm]
        exp_mf = exp['massflows']
        color = colors[idx]

        # Plot experimental data
        ax.plot(
            exp_mf,
            exp['ttratio'],
            'o',
            color=color,
            label=f'{int(float(rpm) / 1000)}k rpm (exp.)',
            markersize=8,
            linewidth=2.5,
            alpha=0.8,
        )

        # Plot computed data with error bars
        if rpm in comp_data:
            comp = comp_data[rpm]
            comp_mf = np.array(comp['massflows'])
            comp_ttr = np.array(comp['ttratio'])

            # Plot computed line
            ax.plot(
                comp_mf,
                comp_ttr,
                '-',
                color=color,
                label=f'{int(float(rpm) / 1000)}k rpm (comp.)',
                linewidth=2.5,
                alpha=0.8,
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

        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.5)
        ax.tick_params('both', labelsize=30)

    axs[2].set_xlabel(r'$\dot{m}$ [kg/s]')
    axs[3].set_xlabel(r'$\dot{m}$ [kg/s]')
    axs[0].set_ylabel(r'$\tau_{tt}$ [−]')
    axs[2].set_ylabel(r'$\tau_{tt}$ [−]')

    fig3.tight_layout()
    fig3.savefig(
        'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\latex'
        '\\gpps26_ADeT\\Images\\HECC_temperature_ratios.pdf'
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
    plt.show()
