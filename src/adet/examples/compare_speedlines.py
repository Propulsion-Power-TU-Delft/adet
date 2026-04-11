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
        return

    # Create figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Sort by RPM for consistent plotting
    rpms = sorted(set(exp_data.keys()) & set(comp_data.keys()), key=float)

    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(rpms)))

    # Plot 1: Pressure ratio comparison
    ax = axs[0]
    for rpm, color in zip(rpms, colors):
        exp = exp_data[rpm]
        # Convert to lbm/s
        exp_mf = exp['massflows'] * 2.2
        ax.plot(
            exp_mf,
            exp['pratios'],
            'o',
            color=color,
            label=f'{float(rpm) / 21000:.2f} N_des (exp)',
            markersize=6,
            linewidth=2,
            alpha=0.7,
        )

        if rpm in comp_data:
            comp = comp_data[rpm]
            comp_mf = np.array(comp['massflows']) * 2.2
            comp_pr = np.array([pr for pr in comp['pratios'] if pr is not None])
            comp_mf_valid = comp_mf[: len(comp_pr)]
            ax.plot(
                comp_mf_valid,
                comp_pr,
                '-',
                color=color,
                label=f'{float(rpm) / 21000:.2f} N_des (comp)',
                linewidth=2,
                alpha=0.7,
            )

    ax.set_xlabel('Mass flow [lbm/s]', fontsize=11)
    ax.set_ylabel('Pressure ratio [−]', fontsize=11)
    ax.set_title('Pressure Ratio Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Efficiency comparison
    ax = axs[1]
    for rpm, color in zip(rpms, colors):
        exp = exp_data[rpm]
        exp_mf = exp['massflows'] * 2.2
        ax.plot(
            exp_mf,
            exp['etas'],
            'o',
            color=color,
            label=f'{float(rpm) / 21000:.2f} N_des (exp)',
            markersize=6,
            linewidth=2,
            alpha=0.7,
        )

        if rpm in comp_data:
            comp = comp_data[rpm]
            comp_mf = np.array(comp['massflows']) * 2.2
            comp_eta = np.array([100 * eta for eta in comp['etas'] if eta is not None])
            comp_mf_valid = comp_mf[: len(comp_eta)]
            ax.plot(
                comp_mf_valid,
                comp_eta,
                '-',
                color=color,
                label=f'{float(rpm) / 21000:.2f} N_des (comp)',
                linewidth=2,
                alpha=0.7,
            )

    ax.set_xlabel('Mass flow [lbm/s]', fontsize=11)
    ax.set_ylabel('Total-to-total efficiency [%]', fontsize=11)
    ax.set_title('Efficiency Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


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
    fig = plot_comparison(exp_data, comp_data)
    if fig:
        plt.show()
