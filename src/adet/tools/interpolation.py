from typing import Literal
from numpy.typing import NDArray
from sympy import Symbol
from pint.facets.plain import PlainQuantity
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs


def is_casadi_type(x):
    return isinstance(x, (cs.DM, cs.MX, cs.SX))


def safe_min_clip(x, min_value):
    """
    Lower clipping of the absolute vaue of x
    with respect to a minimum value.
    Type safe for casadi, numpy and pint
    """
    if is_casadi_type(x):
        # OBS: If x is exactly 0 the normal sign function
        # is problematic
        x_sign = cs.if_else(x >= 0, 1, -1)
        x = x_sign * cs.fmax(cs.fabs(x), min_value)
    elif isinstance(x, PlainQuantity):
        x_sign = np.sign(x.magnitude)
        x = x_sign * np.clip(np.abs(x.magnitude), min_value, None) * x.units
    elif isinstance(x, Symbol):
        pass
    else:
        x_sign = np.sign(x)
        x = x_sign * np.clip(np.abs(x), min_value, None)

    return x


def make_casadi_interpolant(
    x: NDArray,
    y: NDArray,
    data: NDArray,
    name: str = 'GenericInterpolant',
    method: str | Literal['linear', 'bspline'] = 'linear',
):
    # NOTE::
    # The flat data should move along the x first
    # and then the y
    # [[1,2,3], [4,5,6]]    -> ravel('C') -> [1,2,3,4,5,6]
    # [[1,4], [2,5], [3,6]] -> ravel('F') -> [1,2,3,4,5,6]

    if len(x) == data.shape[0] and len(y) == data.shape[1]:
        data_flat = data.ravel('F')
    elif len(x) == data.shape[1] and len(y) == data.shape[0]:
        data_flat = data.ravel('C')
    else:
        raise ValueError('Size mismatch for interpolation')

    return cs.interpolant(name, method, [x, y], data_flat)


def fin_diff(f, x, edge_order: Literal['first', 'second'] = 'second'):
    """
    Centered finite difference df/dx derivative along the span,
    fwd and bwd on the edges
    """
    first_delta_x = safe_min_clip(x[1] - x[0], 1e-8)
    inner_delta_x = safe_min_clip(x[2:] - x[:-2], 1e-8)
    final_delta_x = safe_min_clip(x[-1] - x[-2], 1e-8)

    if edge_order == 'first':
        first_der = (f[1] - f[0]) / first_delta_x
        final_der = (f[-1] - f[-2]) / final_delta_x
    else:
        first_der = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * first_delta_x)
        final_der = (f[-3] - 4 * f[-2] + 3 * f[-1]) / (2 * final_delta_x)

    internal_der = (f[2:] - f[:-2]) / inner_delta_x

    return cs.vertcat(first_der, internal_der, final_der)


class TransfiniteInterpolator:
    def __init__(self, curve0, curve1, nu=20, nv=10):
        """
        Transfinite Interpolation (TFI) for generating a quadrilateral grid
        between two boundary curves.

        Parameters
        ----------
        curve0 : (2,N) array
            Points defining the first curve (bottom).
        curve1 : (2,N) array
            Points defining the second curve (top).
        nu : int
            Number of subdivisions along curves (u-direction).
        nv : int
            Number of subdivisions between curves (v-direction).
        """
        self.curve0 = np.asarray(curve0)
        self.curve1 = np.asarray(curve1)
        self.nu = nu
        self.nv = nv
        self.grid = None

        if self.curve0.shape != self.curve1.shape:
            raise ValueError('curve0 and curve1 must have the same shape.')

    def generate_grid(self):
        """Generate a structured quadrilateral grid using TFI."""
        curve0, curve1, nu, nv = self.curve0, self.curve1, self.nu, self.nv
        N = curve0.shape[1]

        # Parameter along curves
        U_curve = np.linspace(0, 1, N)

        def interpolate_curve(curve, u):
            return np.array(
                [np.interp(u, U_curve, curve[0, :]), np.interp(u, U_curve, curve[1, :])]
            )

        # Endpoints
        P00, P10 = curve0[:, 0], curve0[:, -1]
        P01, P11 = curve1[:, 0], curve1[:, -1]

        def C0(u):
            return interpolate_curve(curve0, u)

        def C1(u):
            return interpolate_curve(curve1, u)

        def L0(v):
            return (1 - v) * P00 + v * P01

        def L1(v):
            return (1 - v) * P10 + v * P11

        def TFI(u, v):
            term = (1 - v) * C0(u) + v * C1(u) + (1 - u) * L0(v) + u * L1(v)
            blend = (
                (1 - u) * (1 - v) * P00
                + u * (1 - v) * P10
                + (1 - u) * v * P01
                + u * v * P11
            )
            return term - blend

        U = np.linspace(0, 1, nu)
        V = np.linspace(0, 1, nv)
        grid = np.zeros((nu, nv, 2))

        for i, u in enumerate(U):
            for j, v in enumerate(V):
                grid[i, j] = TFI(u, v)

        self.grid = grid
        return grid

    def plot(self, show=True):
        """Plot the generated grid with boundary curves."""
        if self.grid is None:
            raise ValueError('Grid not generated yet. Call generate_grid() first.')

        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot grid lines
        for j in range(self.grid.shape[1]):
            ax.plot(self.grid[:, j, 0], self.grid[:, j, 1], 'k-', lw=0.7)
        for i in range(self.grid.shape[0]):
            ax.plot(self.grid[i, :, 0], self.grid[i, :, 1], 'k-', lw=0.7)

        # Plot boundary curves
        ax.plot(self.curve0[0], self.curve0[1], 'ro', lw=1.5, label='Curve0')
        ax.plot(self.curve1[0], self.curve1[1], 'bo', lw=1.5, label='Curve1')

        ax.legend()
        # ax.set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Transfinite Interpolated Grid')

        if show:
            plt.show()


def test_transfinite():
    # Example curves
    theta = np.linspace(0, np.pi, 50)
    curve0 = np.array([theta, np.sin(theta)])  # bottom curve
    curve1 = np.array([theta, np.sin(theta) + 1.0])  # top curve

    tfi = TransfiniteInterpolator(curve0, curve1, nu=30, nv=15)
    grid = tfi.generate_grid()
    tfi.plot()

    return grid
