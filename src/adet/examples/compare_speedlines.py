"""
Compare experimental speedline data with computed results from nasa_hecc.py
"""

import json
import pathlib
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# Load speedline data from data directory
data_dir = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / 'data'
    / 'opencases'
    / 'nasa_hecc'
)

plt.rcParams.update(
    {
        'text.usetex': True,
        'font.family': 'serif',
    }
)

# Load speedline_data module using importlib
speedline_data_path = data_dir / 'speedline_data.py'
if not speedline_data_path.exists():
    raise FileNotFoundError(
        f'Speedline data file not found: {speedline_data_path}\n'
        'Please ensure the data directory exists at the expected location.'
    )

spec = importlib.util.spec_from_file_location('speedline_data', speedline_data_path)
if spec is None or spec.loader is None:
    raise ImportError(f'Could not load module from {speedline_data_path}')

speedline_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speedline_data_module)

beta_tt_data = speedline_data_module.beta_tt_data
eta_tt_data = speedline_data_module.eta_tt_data


def extract_experimental_data():
    """Extract experimental data from speedline_data.py"""
    exp_data = {}

    # Map speed names to RPM values
    speed_map = {
        '18krpm': 18000,
        '19krpm': 19000,
        '20krpm': 20000,
        '21krpm': 21000,
    }

    for speed_name, rpm in speed_map.items():
        m_key = f'm_{speed_name}'
        pr_key = f'beta_{speed_name}'
        eta_key = f'eta_{speed_name}'

        if m_key in beta_tt_data and pr_key in beta_tt_data:
            exp_data[f'{rpm}'] = {
                'massflows': np.array(beta_tt_data[m_key]),
                'pratios': np.array(beta_tt_data[pr_key]),
                'etas': np.array(eta_tt_data[eta_key]),
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
    fig1, ax = plt.subplots(figsize=(19, 12))
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
            comp_pr = np.array([pr for pr in comp['pratios'] if pr is not None])
            comp_mf_valid = comp_mf[: len(comp_pr)]
            ax.plot(
                comp_mf_valid,
                comp_pr,
                '-',
                color=color,
                label=f'{int(float(rpm) / 1000)}k rpm (comp.)',
                linewidth=2.5,
                alpha=0.7,
            )

            # Add ±2% error band
            error_lower = comp_pr * 0.98
            error_upper = comp_pr * 1.02
            ax.fill_between(
                comp_mf_valid,
                error_lower,
                error_upper,
                color=color,
                alpha=0.2,
            )
            ax.tick_params('both', labelsize=30)

    ax.set_xlabel(r'$\dot{m}$ [kg/s]', fontsize=35)
    ax.set_ylabel(r'$\beta_{tt}$ [−]', fontsize=35)

    ax.legend(loc='upper left', fontsize=30)
    ax.grid(True, alpha=0.5)
    fig1.tight_layout()
    fig1.savefig(
        'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\latex'
        '\\gpps26_ADeT\\Images\\HECC_pratios.pdf'
    )

    # Create figure 2: Efficiency comparison with separate subplot for each speedline
    fig2, axs = plt.subplots(2, 2, figsize=(17, 13), sharey=True, sharex=True)
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
            comp_eta = np.array([100 * eta for eta in comp['etas'] if eta is not None])
            comp_mf_valid = comp_mf[: len(comp_eta)]

            # Plot computed line
            ax.plot(
                comp_mf_valid,
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
                comp_mf_valid,
                error_lower,
                error_upper,
                color=color,
                alpha=0.2,
                # label=r'$\pm$ 2% error',
            )

        ax.set_ylim(75, 90)
        ax.legend(loc='lower left', fontsize=35)
        ax.grid(True, alpha=0.5)
        ax.tick_params('both', labelsize=30)

    axs[2].set_xlabel(r'$\dot{m}$ [kg/s]', fontsize=35)
    axs[3].set_xlabel(r'$\dot{m}$ [kg/s]', fontsize=35)
    axs[0].set_ylabel(r'$\eta_{tt}$ [\%]', fontsize=35)
    axs[2].set_ylabel(r'$\eta_{tt}$ [\%]', fontsize=35)

    fig2.tight_layout()
    fig2.savefig(
        'C:\\Users\\fvaccari\\OneDrive - Delft University of Technology\\latex'
        '\\gpps26_ADeT\\Images\\HECC_efficiencies.pdf'
    )
    return fig1, fig2


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
    fig1, fig2 = plot_comparison(exp_data, comp_data)
    plt.show()
