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


def fin_diff(f, x):
    """
    Centered finite difference derivative along the span, fwd and
    bwd on the edges
    """
    first_delta_x = safe_min_clip(x[1] - x[0], 1e-8)
    first_der = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * first_delta_x)

    end_delta_x = safe_min_clip(x[-1] - x[-2], 1e-8)
    last_der = (f[-3] - 4 * f[-2] + 3 * f[-1]) / (2 * end_delta_x)

    inner_delta_x = safe_min_clip(x[2:] - x[:-2], 1e-8)
    internal_der = (f[2:] - f[:-2]) / inner_delta_x

    return cs.vertcat(first_der, internal_der, last_der)


def findpro(x, vec):
    """findpro returns the position of the element of vec whose value is nearest to x.
    In case you have a matrix, you need to select a row as input"""

    N = len(vec)
    diff = np.zeros(N)

    for i in range(N):
        diff[i] = np.abs(vec[i] - x)

    pos = np.nonzero(diff == np.min(diff))

    if len(pos) != 1:
        pos = pos[0]

    return pos


def weight(x, vec):
    """
    Weight returns the positions of the two elements of vec whose values are nearest
    to x, together with the correspondent weights. Weight properly works only if the
    elements of vec are in increasing or decreasing order, not sparse (e.g. functions
    parametrized on y). The closer is the value of vec wrt x, the higher is its weight
    """

    N = len(vec)

    if N == 1:  # correction for simple x-z charts (not parametrized on y)
        pos1 = 0
        w1 = 1
        pos2 = 0
        w2 = 0
    else:
        diff = np.zeros(N)

        if vec[0] < vec[1]:
            ord = 'increasing'
        else:
            ord = 'decreasing'

        for i in range(N):
            diff[i] = np.abs(vec[i] - x)

        pos1 = np.argmin(diff)
        pos2 = 0

        if (x > vec[pos1]) and (ord == 'increasing'):
            pos2 = pos1 + 1
        elif (x > vec[pos1]) and (ord == 'decreasing'):
            pos2 = pos1 - 1
        elif (x < vec[pos1]) and (ord == 'increasing'):
            pos2 = pos1 - 1
        elif (x < vec[pos1]) and (ord == 'decreasing'):
            pos2 = pos1 + 1

        if (pos2 >= 0) and (pos2 <= N - 1) and (x != vec[pos1]):
            w1 = 1 - (diff[pos1] / np.abs(vec[pos2] - vec[pos1]))
            w2 = 1 - w1
        else:
            w1 = 1
            pos2 = 0
            w2 = 0

    return pos1, pos2, w1, w2


def w_extrap(x, y, xq, yq, zq):
    """
    w_extrap receives as input 2 vectors xq, yq, 1 matrix zq and 2 values x, y
    within which you whish to enter the chart. It returns the value z corresponding
    to the weighted average between the curves nearest to y,
    evaluated at point x (or nearest to it)
    """

    pos1, pos2, w1, w2 = weight(y, yq)
    pos_x = findpro(x, xq)
    z = w1 * zq[pos1, pos_x][0] + w2 * zq[pos2, pos_x][0]

    return z


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


if __name__ == '__main__':
    # Example curves
    theta = np.linspace(0, np.pi, 50)
    curve0 = np.array([theta, np.sin(theta)])  # bottom curve
    curve1 = np.array([theta, np.sin(theta) + 1.0])  # top curve

    tfi = TransfiniteInterpolator(curve0, curve1, nu=30, nv=15)
    grid = tfi.generate_grid()
    tfi.plot()
