from adet.equations.utils import safe_min_clip
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from adet.varspec import VarSpec

logger = logging.getLogger(__name__)


def setup_mpl(fontdict: dict[str, Any] = {}):
    """
    Setup matplotlib using custom parameters,
    add fonts from system directories
    """
    try:
        repo_root = Path(__file__).parents[4]
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
    mer_chord,
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
    mer_chord : float
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

    a = (tan1 - tan0) / (2 * mer_chord)
    b = tan0

    a = safe_min_clip(a, 1e-3)

    # y_out = a * mer_chord**2 + b * mer_chord

    x = np.linspace(0, mer_chord, n_points)
    y = a * x**2 + b * x

    ax.plot(axial_offset + x, tangential_offset + y, color=color, **kwargs)


def plot_design_map(
    keys: np.ndarray,
    solution_dicts: Sequence[dict | None],
    quantity: VarSpec | Callable[[dict], float],
    n_points: int,
    *,
    ax: Axes | None = None,
    reduce: Callable[[Any], float] = lambda a: float(np.asarray(a).reshape(-1)[0]),
    x_label: str = r'Flow coefficient $\phi$',
    y_label: str = r'Loading coefficient $\psi$',
    z_label: str | None = None,
    cmap: str = 'viridis',
    levels: int | np.ndarray = 20,
    colorbar: bool = True,
    contourf_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure | Any, Axes, Any]:
    """Contour-plot a scalar quantity over a design map's 2D sweep grid,
    from the ``keys``/``solution_dicts`` produced by
    ``adet.tools.design_map.sweep_design_map``.

    ``quantity`` is either a ``VarSpec`` (looked up in each solution dict
    and reduced via ``reduce``) or a callable ``sol_dict -> float``.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        assert fig is not None

    x = keys[:, 0].reshape(n_points, n_points)
    y = keys[:, 1].reshape(n_points, n_points)

    if isinstance(quantity, VarSpec):
        spec = quantity
        extract: Callable[[dict], float] = lambda sol_dict: reduce(sol_dict[spec])  # noqa: E731
    else:
        extract = quantity

    z = np.full(n_points * n_points, np.nan)
    for i, sol_dict in enumerate(solution_dicts):
        if sol_dict is None:
            continue
        try:
            z[i] = extract(sol_dict)
        except (KeyError, TypeError, ValueError):
            z[i] = np.nan
    z = z.reshape(n_points, n_points)

    cs = ax.contourf(x, y, z, levels=levels, cmap=cmap, **(contourf_kwargs or {}))
    if colorbar:
        default_label = getattr(quantity, 'symbol', '')
        fig.colorbar(cs, ax=ax, label=z_label if z_label is not None else default_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return fig, ax, cs
