"""ADeT Logo Generator"""

import matplotlib.pyplot as plt
import numpy as np

ANGLE = 60 * np.pi / 180
NUM_ARROWS = 15
TOP_HEIGHT = 0.35


def sin_distr(s):
    return TOP_HEIGHT * np.sin(s / TOP_HEIGHT * np.pi / 2)


def tan_distr(s):
    return TOP_HEIGHT * np.arctan(s / TOP_HEIGHT * 1.5)


func = tan_distr


zero = np.zeros(NUM_ARROWS)
one = np.ones(NUM_ARROWS)
angle = ANGLE * np.ones(NUM_ARROWS)

S = np.linspace(0.0, TOP_HEIGHT, NUM_ARROWS)
Y = func(S)
U = Y * np.sin(ANGLE)


cmap = plt.get_cmap('YlOrRd')
plot_settings = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1}

C = [cmap(i / NUM_ARROWS / 1.2) for i in range(NUM_ARROWS)]
W = np.linspace(0.005, 0.008, NUM_ARROWS)

fig, ax = plt.subplots()
# BOTTOM
for i in range(NUM_ARROWS):
    ax.quiver(
        np.tan(angle)[i],
        Y[i],
        ((Y - 1) * np.sin(angle))[i],
        zero[i],
        color=C[i],
        width=W[i],
        headwidth=4,
        headlength=4,
        headaxislength=4,
        **plot_settings,
    )
# RIGHT SIDE
ax.quiver(
    np.tan(angle),
    one,
    zero,
    (Y - 1),
    color=C[-1],
    width=0.07,
    headwidth=0,
    headlength=0,
    headaxislength=0,
    **plot_settings,
)
# SLANTED
ax.quiver(
    np.tan(angle),
    one,
    ((Y - 1) * np.sin(angle)),
    (Y - 1),
    color=C[-1],
    width=0.015,
    headwidth=0,
    headlength=0,
    headaxislength=0,
    **plot_settings,
)


# Set bounds to center the logo
ax.set_xlim(0.5, np.tan(ANGLE) + 0.1)
ax.set_ylim(0.0, 1.1)
ax.set_aspect('equal')

# Remove axes
ax.axis('off')

plt.show()
