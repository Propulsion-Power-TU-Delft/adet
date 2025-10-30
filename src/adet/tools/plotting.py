import logging

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


logger = logging.getLogger(__name__)

# plt.rcParams.update(
#     {
#         'text.usetex': False,
#         'font.family': 'serif',
#     }
# )


def plot_velocity_triangles(kine, geo, fontsize):
    plot_settings = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1}
    fig, ax = plt.subplots()
    fontsize = 14
    ticksize = fontsize / 1.5 // 1
    fontdict = {'fontsize': fontsize}

    ax.set_ylabel(r'Radial coordinate [mm]', fontdict)
    ax.set_xlabel(r'Axial coordinate [mm]', fontdict)

    ax.tick_params(labelsize=ticksize)
    ax.set_aspect('equal')
    ax.grid()

    # Define a colormap for each quiver
    cmap_v = plt.get_cmap('Reds')
    cmap_w = plt.get_cmap('Blues')
    cmap_u = plt.get_cmap('Purples')

    Wt = kine.Wt
    Wm = kine.Wm
    U = kine.U
    Vt = kine.Vt
    Vm = kine.Vm
    rr = geo.rr
    num_span = geo._spanwise_stations

    colors_v = [cmap_v((i + 1) / num_span) for i in range(num_span)]
    colors_w = [cmap_w((i + 1) / num_span) for i in range(num_span)]
    colors_u = [cmap_u((i + 1) / num_span) for i in range(num_span)]

    # Plot all quivers at once (vectorized)
    ax.quiver(  # W
        np.zeros(num_span), np.zeros(num_span), Wm, Wt, color=colors_w, **plot_settings
    )
    ax.quiver(  # U
        Wm, Wt, np.zeros(num_span), U, color=colors_u, **plot_settings
    )
    ax.quiver(  # V
        np.zeros(num_span), np.zeros(num_span), Vm, Vt, color=colors_v, **plot_settings
    )

    ax.axis('equal')

    # Set the limits of the plot
    ax.set_xlim(0, max(Vm) * 1.05)

    tang_stack = np.stack([np.zeros(kine._spanwise_stations), Wt, Vt])
    ax.set_ylim(-10 + np.min(tang_stack), 10 + np.max(tang_stack))

    ax.grid(alpha=0.3)

    # Add a colorbar based on V velocity and radius distribution

    def fmt(x, pos):
        return '{:.2f}'.format(x)

    if (np.isclose(U, 0.0)).all():
        cmap_colorbar = cmap_v
    else:
        cmap_colorbar = cmap_w

    sm = ScalarMappable(Normalize(rr[0], rr[-1]), cmap_colorbar)

    cbar_v = plt.colorbar(
        sm,
        ax=ax,
        orientation='vertical',
        fraction=0.02,
        pad=0.10,
        ticks=np.linspace(rr[0], rr[-1], 5),
        format=FuncFormatter(fmt),
    )
    cbar_v.set_label('Radius [m]', loc=None, **fontdict)

    # ax.legend(['W', 'U', 'V'], loc='lower right', **fontsett)

    ax.set_xlabel(
        'Meridional Component [m/s]',
        fontdict=fontdict,
    )
    ax.set_ylabel(
        'Tangential Component [m/s]',
        fontdict=fontdict,
    )

    return fig, ax
