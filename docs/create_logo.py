"""
ADeT Logo Generator

Creates a logo based on velocity triangles - a fundamental concept in turbomachinery.
The logo features multiple right-angle triangles with vector arrows representing
velocity components.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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


cmap = plt.get_cmap('Purples')
plot_settings = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1}

C = [cmap(i / NUM_ARROWS) for i in range(NUM_ARROWS)]
W = np.linspace(0.005, 0.01, NUM_ARROWS)

fig, ax = plt.subplots(figsize=(15, 10))
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
    width=0.03,
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
    width=0.01,
    headwidth=0,
    headlength=0,
    headaxislength=0,
    **plot_settings,
)


ax.set_xbound(0.0, np.tan(ANGLE))
ax.set_ybound(0.0, 1.0)
ax.set_aspect('equal')

plt.show()
