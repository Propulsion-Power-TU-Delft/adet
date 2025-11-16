from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from bezier import Curve

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


class Point:
    """A 4D point with homogeneous coordinates (x, y, z, w).

    Parameters
    ----------
    x : float, optional
        X coordinate, by default 0.0
    y : float, optional
        Y coordinate, by default 0.0
    z : float, optional
        Z coordinate, by default 0.0
    w : float, optional
        Homogeneous coordinate, by default 1.0
    """

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __call__(self):
        """Return point coordinates as tuple.

        Returns
        -------
        tuple
            (x, y, z) coordinates
        """
        return self.x, self.y, self.z

    def towards(self, target, t):
        """Create new point at fraction t between self and target.

        Parameters
        ----------
        target : Point
            Target point
        t : float
            Interpolation parameter between 0 and 1

        Returns
        -------
        Point
            New interpolated point
        """
        return Point(
            (1.0 - t) * self.x + t * target.x,
            (1.0 - t) * self.y + t * target.y,
            (1.0 - t) * self.z + t * target.z,
        )

    def distance(self, target):
        """Calculate Euclidean distance to target point.

        Parameters
        ----------
        target : Point
            Target point
        Returns
        -------
        float
            Distance between points
        """
        return np.sqrt(
            (self.x - target.x) ** 2
            + (self.y - target.y) ** 2
            + (self.z - target.z) ** 2
        )

    def distance_towards(self, target, t):
        """Calculate distance to interpolated point.

        Parameters
        ----------
        target : Point
            Target point
        t : float
            Interpolation parameter between 0 and 1

        Returns
        -------
        float
            Distance to interpolated point
        """
        temp_point = self.towards(target, t)
        return np.sqrt(
            (self.x - temp_point.x) ** 2
            + (self.y - temp_point.y) ** 2
            + (self.z - temp_point.z) ** 2
        )


class Line:
    """A parametric line in 2D/3D space defined by a point and direction.

    The line is defined by a point P and a direction vector T, such that any point
    on the line can be represented as P + uT, where u is a parameter.

    Parameters
    ----------
    P : Point
        A point object defining the starting point of the line.
    Tx : float
        X component of the direction vector.
    Ty : float
        Y component of the direction vector.
    Tz : float, optional
        Z component of the direction vector. Defaults to 0.0.

    Notes
    -----
    Implementation adapted from the work of Dr. S. Vitale.
    """

    def __init__(self, P, Tx, Ty, Tz=0.0):
        self.P = P
        self.T = [Tx, Ty, Tz]

    def __call__(self, u):
        """Get a point on the line at parameter value u.

        Parameters
        ----------
        u : float
            The parameter value determining the point position along the line.

        Returns
        -------
        Point
            A Point object representing the position P + uT.
        """
        return Point(
            self.P.x + self.T[0] * u,
            self.P.y + self.T[1] * u,
            self.P.z + self.T[2] * u,
        )

    def intersect(self, other):
        """Find the intersection point with another line.

        Parameters
        ----------
        other : Line
            The other line to find intersection with.

        Returns
        -------
        Point
            The point of intersection between the two lines.

        Raises
        ------
        ValueError
            If the lines are parallel and do not intersect.

        """
        P1, T1, P2, T2 = self.P, self.T, other.P, other.T

        # Calculate the determinant
        determinant = T1[0] * T2[1] - T1[1] * T2[0]

        if determinant == 0:
            raise ValueError('Lines are parallel and do not intersect.')

        u = (T2[1] * (P2.x - P1.x) - T2[0] * (P2.y - P1.y)) / determinant
        return self(u)


class GenericCurve(ABC):
    """Abstract base class for meridional curves in turbomachinery flow paths.

    This class defines the interface for creating different types of meridional curves
    used to describe turbomachinery flow paths in the meridional (z-R) plane.
    Subclasses must implement the abstract methods to define specific curve types
    (e.g., Bezier curves, splines, straight lines).

    Parameters
    ----------
    z_in : float
        Starting z-coordinate
    z_out : float
        Ending z-coordinate
    R_in : float
        Starting radius
    R_out : float
        Ending radius
    angle_in : float
        Inlet tangent angle in radians
    angle_out : float
        Outlet tangent angle in radians
    n_points : int, optional
        Number of points used to discretize the curve, by default 100

    Attributes
    ----------
    z_coords : ndarray
        array of z-coordinates along the line
    R_coords : ndarray
        array of radial coordinates along the line
    area : float
        area under the straight line
    """

    def __init__(
        self,
        z_in: float,
        z_out: float,
        radius_in: float,
        radius_out: float,
        angle_in: float,
        angle_out: float,
        n_points: int = 100,
    ):
        self.z_in: float = z_in
        self.z_out: float = z_out
        self.r_in: float = radius_in
        self.r_out: float = radius_out
        self.n_points: int = n_points
        self.angle_in: float = angle_in
        self.angle_out: float = angle_out

        self.z_coords: NDArray = np.array([])
        self.r_coords: NDArray = np.array([])

        self.create_curve()
        self.area = self.get_area()

    @abstractmethod
    def create_curve(self):
        raise NotImplementedError

    @abstractmethod
    def get_area(self):
        raise NotImplementedError

    def plot_curve(self, color: str | None = None, ax=None) -> Line2D:
        if ax is None:
            ax = plt.gca()
        line = ax.plot(self.z_coords, self.r_coords)[0]

        if color:
            line.set_color(color)

        return line


class BezierCurve(GenericCurve):
    """
    Creates a cubic Bezier curve between two points with specified tangent angles.
    The curve is defined by four control points:

    1. Start point
    2. Point along inlet tangent line at 1/3 chord distance
    3. Point along outlet tangent line at 1/3 chord distance
    4. End point

    .. image:: ../svg/bezier_curve.svg
        :width: 400
        :alt: Bezier curve control points

    Uses the bezier library to create and evaluate the curve.
    """

    def create_curve(self):
        start_point = Point(self.z_in, self.r_in)
        end_point = Point(self.z_out, self.r_out)

        # 1/3rd of the distance along the segment between start and end
        chord_third = start_point.distance_towards(end_point, 1 / 3)

        inlet_line = Line(start_point, np.cos(self.angle_in), np.sin(self.angle_in))
        outlet_line = Line(end_point, np.cos(self.angle_out), np.sin(self.angle_out))

        # TODO get length
        middle_point1 = inlet_line(chord_third)
        middle_point2 = outlet_line(-chord_third)

        control_points = [start_point, middle_point1, middle_point2, end_point]

        self._z_cont_points = np.array([p.x for p in control_points]).flatten()
        self._r_cont_points = np.array([p.y for p in control_points]).flatten()

        control_array = np.asfortranarray([self._z_cont_points, self._r_cont_points])
        self.curve_instance = Curve(control_array, degree=3)
        self.z_coords, self.r_coords = self.curve_instance.evaluate_multi(
            np.linspace(0, 1, self.n_points)
        )

    def get_area(self):
        return np.trapezoid(self.r_coords, self.z_coords)


class CSplineCurve(GenericCurve):
    """
    Creates a cubic spline curve between two points with specified tangent angles.

    Note
    ----
        This cannot handle angles of :math:`\\pi / 2` degrees
    """

    def create_curve(self):
        """Creates the cubic spline profile using scipy's CubicSpline.

        Uses two points and their derivatives to create a cubic spline interpolator.
        The derivatives at endpoints are specified through the tangent angles.

        Returns
        -------
        None
            Sets z_coords, R_coords and curve class attributes.
        """
        z_controls = [self.z_in, self.z_out]
        r_controls = [self.r_in, self.r_out]
        boundary_cond = ((1, np.tan(self.angle_in)), (1, np.tan(self.angle_out)))

        # NOTE: This is incorrectly typed by scipy think about reimplementing this
        self.curve_instance = CubicSpline(z_controls, r_controls, bc_type=boundary_cond)  # type: ignore

        self.z_coords = np.linspace(self.z_in, self.z_out, self.n_points)
        self.r_coords = self.curve_instance(self.z_coords)

    def get_area(self):
        """Calculates the exact area under the cubic spline using the antiderivative.

        Computes the area by evaluating the antiderivative of the cubic spline
        at the endpoints and taking their difference.

        Returns
        -------
        None
            Sets area class attribute.
        """
        a, b, c, d = self.curve_instance.c

        def primitive(z):
            return a * z**4 / 4 + b * z**3 / 3 + c * z**2 / 2 + d * z

        return primitive(self.z_out) - primitive(self.z_in)


class StraightLine(GenericCurve):
    def __init__(
        self,
        z_in: float,
        z_out: float,
        radius_in: float,
        radius_out: float,
        n_points: int = 100,
        **kwargs,
    ):
        super().__init__(z_in, z_out, radius_in, radius_out, 0.0, 0.0, n_points)

    def create_curve(self):
        """Creates the straight line profile.

        For vertical lines (z_in = z_out), creates equally spaced radial coordinates.
        For non-vertical lines, creates a line with slope

        .. math::
            (R_{out} - R_{in})/(z_{out}- z_{in}).

        Returns
        -------
        None
            Sets z_coords and R_coords class attributes, and slope and curve_instance
            for non-vertical lines.
        """
        if self.z_out == self.z_in:
            self.z_coords = np.ones(self.n_points) * self.z_in
            self.r_coords = np.linspace(self.r_in, self.r_out, self.n_points)
        else:
            self.slope = (self.r_out - self.r_in) / (self.z_out - self.z_in)
            self.curve_instance = lambda z: self.slope * (z - self.z_in) + self.r_in
            self.z_coords = np.linspace(self.z_in, self.z_out, self.n_points)
            self.r_coords = self.curve_instance(self.z_coords)

    def get_area(self):
        """Calculates the area under the straight line using exact integration.

        For non-vertical lines, uses the antiderivative of the line equation.
        For vertical lines, the area is zero.

        Returns
        -------
        None
            Sets area class attribute.
        """

        def primitive(z):
            return self.slope * z**2 / 2 + self.r_in * z

        if hasattr(self, 'curve_instance'):
            area = primitive(self.z_out) - primitive(self.z_in)
        else:
            area = 0

        return area


if __name__ == '__main__':
    inputs = {
        'z_in': 0.0,
        'z_out': 0.5,
        'radius_in': 1.0,
        'radius_out': 1.5,
        'angle_in': np.pi / 6,
        'angle_out': np.pi / 3,
        'n_points': 100,
    }

    bez = BezierCurve(**inputs)
    spl = CSplineCurve(**inputs)
    lin = StraightLine(**inputs)

    fig, ax = plt.subplots()

    ax.add_line(bez.plot_curve())
    ax.add_line(spl.plot_curve())
    ax.add_line(lin.plot_curve())

    ax.set_aspect('equal')
    ax.grid(True)

    ax.legend(labels=('bezier', 'spline', 'straight'))

    fig.show()
