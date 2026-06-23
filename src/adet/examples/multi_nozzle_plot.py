from adet.tools.plotting import setup_mpl
import matplotlib.pyplot as plt
import numpy as np

stages = [1, 2, 3, 4, 5, 6, 7]
lin_ratios = [
    0.5282683366174583,
    0.33404434041908715,
    0.23977201439944465,
    0.1850360928339276,
    0.14961057069388345,
    0.12495917638924935,
    0.10689319842831456,
]

lin_area = [
    0.1,
    0.14666666666666667,
    0.19333333333333336,
    0.24000000000000002,
    0.2866666666666667,
    0.33333333333333337,
    0.38,
]


cum_area = [
    0.1,
    0.1404225,
    0.19718478506250003,
    0.27689180480438913,
    0.3888183946014434,
    0.5459885101592119,
    0.7666907156783194,
]

cum_ratios = [
    0.5282682717906668,
    0.3479421339863124,
    0.2327189272908922,
    0.1563001612797737,
    0.10510686331008177,
    0.0707095464145709,
    0.047575712253438984,
]


cns_ratios = [
    0.5282683350059446,
    0.43774592273236734,
    0.38300816662151627,
    0.3447609998026427,
    0.31593638357445303,
    0.293149241307428,
    0.2745245770636253,
]
cns_area = [
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
]


setup_mpl(
    {
        'font.family': 'EB Garamond',
        'font.size': 25,
    }
)

fig, ax = plt.subplots()


def compute_correlation(area_distribution):
    nozzle_pr_crit = cns_ratios[0]
    area_distribution = np.array(area_distribution)
    n = len(area_distribution)

    critical_ratios = []
    for i in range(2, n + 1):
        partial_areas = np.array(area_distribution[:i])
        pr_crit = ((1.035 * nozzle_pr_crit - 0.3953) * i + nozzle_pr_crit) / sum(
            partial_areas[-1] / partial_areas
        )
        critical_ratios.append(float(pr_crit))

    return [nozzle_pr_crit] + critical_ratios


COLORS = ['#990011', '#119900', '#110099']

for area, offset, label, color in zip(
    [cns_area, lin_area, cum_area],
    [-0.3, 0, 0.3],
    ['Constant', 'Linear', 'Cumulative'],
    COLORS,
):
    ax.bar(np.array(stages) + offset, area, width=0.3, label=label, color=color)

ax.grid(alpha=0.4)
ax.set_xlabel('Stage n.')
ax.set_ylabel(r'Throat area / $m^2$')
ax.set_xticks(stages)
ax.legend(fontsize=20)

fig2, ax2 = plt.subplots()

for ratios, label, color, areas in zip(
    [cns_ratios, lin_ratios, cum_ratios],
    ['Constant', 'Linear', 'Cumulative'],
    COLORS,
    [cns_area, lin_area, cum_area],
):
    ax2.plot(
        stages,
        compute_correlation(areas),
        marker='o',
        label=label + ' (correlation)',
        color=color,
        fillstyle='none',
        linestyle='none',
        markersize=10,
    )
    ax2.plot(
        stages,
        ratios,
        marker='o',
        label=label + r' (max $\dot{m}$)',
        color=color,
    )

ax2.set_xlabel('Stage n.')
ax2.set_ylabel(r'$\left(p_{t0}/p_e\right)^*$')
ax2.set_xticks(stages)
ax2.grid(alpha=0.4)
ax2.set_ylim(0, 0.55)
ax2.legend(fontsize=20)

plt.tight_layout()
plt.show(block=False)
