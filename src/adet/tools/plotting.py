import logging

import numpy as np

from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mpc
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def plot_velocity_triangles(kine):
    plot_settings = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1}
    fig, ax = plt.subplots()
    FONTSIZE = 26
    FONTDICT = {'fontsize': FONTSIZE}

    ax.set_ylabel(r'Radial coordinate [mm]', FONTDICT)
    ax.set_xlabel(r'Axial coordinate [mm]', FONTDICT)

    ax.tick_params(labelsize=FONTSIZE / 1.5 // 1)
    ax.set_aspect('equal')
    ax.grid()

    # Define a colormap for each quiver
    cmap_v = plt.colormaps['Reds']
    cmap_w = plt.colormaps['Blues']
    cmap_u = plt.colormaps['Purples']

    Wt = kine.get('Wt').to_base_units().magnitude
    Wm = kine.get('Wm').to_base_units().magnitude
    U = kine.get('U').to_base_units().magnitude
    Vt = kine.get('Vt').to_base_units().magnitude
    Vm = kine.get('Vm').to_base_units().magnitude
    try:
        rr = kine.get('rr').to_base_units().magnitude
    except AttributeError:
        rr = np.linspace(0, 1, kine._spanwise_stations)

    # Plot each quiver along the span
    for i in range(kine._spanwise_stations):
        ax.quiver(  # W
            0,
            0,
            Wm[i],
            Wt[i],
            color=cmap_w((i + 2) / kine._spanwise_stations),
            **plot_settings,
        )
        ax.quiver(  # U
            Wm[i],
            Wt[i],
            0,
            U[i],
            color=cmap_u((i + 2) / kine._spanwise_stations),
            **plot_settings,
        )
        ax.quiver(  # V
            0,
            0,
            Vm[i],
            Vt[i],
            color=cmap_v((i + 2) / kine._spanwise_stations),
            **plot_settings,
        )

    ax.axis('equal')

    # Set the limits of the plot
    ax.set_xlim(0, max(Vm) * 1.05)

    tang_stack = np.stack([np.zeros(kine._spanwise_stations), Wt, Vt])
    ax.set_ylim(-10 + np.min(tang_stack), 10 + np.max(tang_stack))

    ax.grid(alpha=0.3)

    # Add a colorbar
    sm_w = plt.cm.ScalarMappable(
        cmap=cmap_w,
        norm=mpc.Normalize(vmin=rr[0], vmax=rr[-1]),
    )

    def fmt(x, pos):
        return '{:.2f}'.format(x)

    fontsett = {'fontsize': 15}
    if 'rr' in kine._variables:
        cbar_w = fig.colorbar(
            sm_w,
            ax=ax,
            orientation='horizontal',
            fraction=0.02,
            pad=0.10,
            ticks=np.linspace(rr[0], rr[-1], 5),
            format=FuncFormatter(fmt),
        )
        cbar_w.set_label('Radius [m]', **fontsett)  # type:ignore

    # ax.legend(['W', 'U', 'V'], loc='lower right', **fontsett)

    ax.set_xlabel(
        'Meridional Component [m/s]',
        fontdict=fontsett,
    )
    ax.set_ylabel(
        'Tangential Component [m/s]',
        fontdict=fontsett,
    )

    return fig, ax
