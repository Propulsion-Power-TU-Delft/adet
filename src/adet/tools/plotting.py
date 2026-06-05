from typing import Any
import logging
import matplotlib as mpl
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)


def setup_mpl(fontdict: dict[str, Any] = {}):
    """
    Setup matplotlib using custom parameters,
    add fonts from system directories
    """
    try:
        fm.fontManager.addfont(
            path='C:/Users/fvaccari/AppData/Local/Microsoft/Windows/'
            'Fonts/EBGaramond-Regular.ttf',
        )
        fm.fontManager.addfont(
            path='C:/Users/fvaccari/AppData/Local/Microsoft/Windows/'
            'Fonts/OldStandardTT-Regular.ttf',
        )
    except FileNotFoundError:
        pass

    custom_params = {
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.weight': 'regular',
        'font.size': 16,
        **fontdict,
    }

    mpl.rcParams.update(custom_params)


def plot_velocity_triangles(Vt, Vm, U, rr, ax: Axes, fontsize=11):
    plot_settings = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1}
    fontdict = {'fontsize': fontsize}
    ticksize = fontsize / 1.5 // 1

    # Preprocess
    num_span = len(Vt)
    Wt = Vt - U

    ax.set_ylabel(r'Radial coordinate [mm]', fontdict)
    ax.set_xlabel(r'Axial coordinate [mm]', fontdict)

    ax.tick_params(labelsize=ticksize)
    ax.grid()

    # Define a colormap for each quiver
    cmap_v = plt.get_cmap('Reds')
    cmap_w = plt.get_cmap('Blues')
    cmap_u = plt.get_cmap('Purples')

    colors_v = [cmap_v((i + 1) / num_span) for i in range(num_span)]
    colors_w = [cmap_w((i + 1) / num_span) for i in range(num_span)]
    colors_u = [cmap_u((i + 1) / num_span) for i in range(num_span)]

    # Plot all quivers at once (vectorized)
    ax.quiver(  # W
        np.zeros(num_span), np.zeros(num_span), Vm, Wt, color=colors_w, **plot_settings
    )
    ax.quiver(  # U
        Vm, Wt, np.zeros(num_span), U, color=colors_u, **plot_settings
    )
    ax.quiver(  # V
        np.zeros(num_span), np.zeros(num_span), Vm, Vt, color=colors_v, **plot_settings
    )

    # Set the limits of the plot
    ax.set_xlim(0, max(Vm) * 1.05)

    tang_stack = np.stack([np.zeros(num_span), Wt, Vt])
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

    ax.legend(['W', 'U', 'V'], loc='lower right', **fontdict)

    ax.set_xlabel(
        'Meridional Component [m/s]',
        fontdict=fontdict,
    )
    ax.set_ylabel(
        'Tangential Component [m/s]',
        fontdict=fontdict,
    )
