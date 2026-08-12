from adet.equations.utils import safe_min_clip
import logging
from pathlib import Path
from typing import Any

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
        repo_root = Path(__file__).parent.parent.parent.parent
        font_path = repo_root / 'fonts' / 'EBGaramond-Regular.ttf'
        fm.fontManager.addfont(path=str(font_path))
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


def plot_velocity_triangles(Vt, Vm, U, rr, ax: Axes):
    plot_settings = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1}

    # Preprocess
    num_span = len(Vt)
    Wt = Vt - U

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
        pad=0.10,
        fraction=0.02,
        orientation='vertical',
        format=FuncFormatter(fmt),
        ticks=np.linspace(rr[0], rr[-1], 5),
    )
    cbar_v.set_label('Radius [m]', loc=None)

    ax.legend(['W', 'U', 'V'], loc='best')
    ax.set_xlabel(r'$V_m$ / [m/s]')
    ax.set_ylabel(r'$V_t$ / [m/s]')


def plot_camberline(
    inlet_angle,
    outlet_angle,
    chord_ax,
    ax,
    color,
    *,
    axial_offset=0.0,
    tangential_offset=0.0,
    n_points=50,
    **kwargs,
):
    """
    Plot a 2D parabolic camber line on the given axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axes to plot on
    inlet_angle : float
        Inlet metal angle [rad]
    outlet_angle : float
        Outlet metal angle [rad]
    chord_ax : float
        Axial chord length [m]
    color : str or color
        Color for the camber line
    axial_offset : float, optional
        Offset in axial direction (default: 0.0)
    tangential_offset : float, optional
        Offset in tangential/pitch direction (default: 0.0)
    n_points : int, optional
        Number of points to generate along camber line (default: 50)
    **kwargs : optional
        Additional keyword arguments passed to matplotlib's plot function
        (e.g., linewidth, linestyle, alpha)
    """
    tan0 = np.tan(inlet_angle)
    tan1 = np.tan(outlet_angle)

    a = (tan1 - tan0) / (2 * chord_ax)
    b = tan0

    a = safe_min_clip(a, 1e-3)

    # y_out = a * chord_ax**2 + b * chord_ax

    x = np.linspace(0, chord_ax, n_points)
    y = a * x**2 + b * x

    ax.plot(axial_offset + x, tangential_offset + y, color=color, **kwargs)
