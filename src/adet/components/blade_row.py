from dataclasses import dataclass
import logging
from typing import Any, Literal

from matplotlib.lines import Line2D
import numpy as np

from adet.components import BaseComponent, Shaft
from adet.equations import EquationBase
from adet.equations.definitions import (
    AngleDeflection,
    BladePitch,
    DeviationAngle,
    MeridionalVelocityRatio,
    Solidity,
    ThicknessToPitch,
)
from adet.equations.fundamental import (
    BladeBlockage,
    ConstantAngMomentum,
    ConstantEnergy,
    EulerEquation,
    MassConservation,
    ZeroBlockage,
)
from adet.equations.geometrical import (
    CamberFunction,
    EndwallProperties,
    MeridionalUniform,
    GeometricalRatios,
    MeridionalVariable,
)
from adet.equations.nondimensional import EnthalpyDropCoefficient
from adet.equations.special import GeometricalAdder
from adet.geometry import BezierCurve, StraightLine
from adet.losses.basic import ZeroDeviation
from adet.losses.mixing import MixingMomentumBalances
from adet.node import FlowNode

logger = logging.getLogger(__name__)

ABSOLUTE_LINK = [
    # Absolute triangle
    'kin_Vt',
    'kin_Vm',
    # No work & no entropy
    'tot_hmass',
    'stc_smass',
]
"""
This preserves the absolute triangles,
energy. Therefore it is a
full variable transfer except for the
rotating frame (omega)
"""

# Geometry
GEOM_LINK = [
    'geo_height',
    'geo_meridional_angle',
    'geo_rr_midspan',
]


class BladeRow(BaseComponent):
    base_equations = [
        # *** Fundamental equations - do not remove
        (EulerEquation, (0, 1)),  # Adiabatic and steady
        (MassConservation, (0, 1)),
        # *** Meridional streamtube distributions
        (MeridionalVariable, 0),
        (MeridionalVariable, 1),
        # *** Blockage - Zero by default
        (ZeroBlockage, 0),
        (ZeroBlockage, 1),
        # *** Common definitions
        (EndwallProperties, 0),
        (EndwallProperties, 1),
        # (MidspanVelocities, 0),
        # (MidspanVelocities, 1),
        (GeometricalRatios, (0, 1)),
        (MeridionalVelocityRatio, (0, 1)),
        # *** Blade count, pitch, channel massflow
        # |> TODO: Make this user-enabled
        (BladePitch, 0),
        (BladePitch, 1),
        (ThicknessToPitch, 0),
        (ThicknessToPitch, 1),
        (Solidity, 1),
        # *** Properties for bounding
        (AngleDeflection, (0, 1)),
        (EnthalpyDropCoefficient, (0, 1)),
        (CamberFunction, (0, 1)),
    ]

    from_previous_node = ABSOLUTE_LINK + GEOM_LINK

    constant_variables = [
        'kin_omega',
        'geo_num_blades',
    ]

    def __init__(
        self,
        name: str,
        shaft: Shaft,
        row_type: Literal['stator', 'rotor'],
        in_constraints: dict[
            str,
            dict[str, Any],
        ],
        out_constraints: dict[
            str,
            dict[str, Any],
        ],
        extra_equations: dict[
            EquationBase,
            int | tuple[int, ...],
        ] = {},
        from_previous_node: list[str] = [],
        constant_variables: list[str] = [],
    ):
        """
        Class that represents a blade row, compressor/turbine,
        stator/rotor.
        Note that the row type only influences the guesses and
        bounds that are enforced on the kinematic variables for
        that specific blare row.
        """
        super().__init__(
            name,
            in_constraints,
            out_constraints,
            extra_equations,
            from_previous_node,
            constant_variables,
        )
        self._shaft = None
        # This uses the setter logic
        self.shaft = shaft
        self.row_type: Literal['stator', 'rotor'] = row_type

    @property
    def shaft(self) -> Shaft | None:
        return self._shaft

    @shaft.setter
    def shaft(self, shaft: Shaft):
        # Fix omega at the outlet node
        if shaft.is_constrained:
            self.outlet_bc['kin']['omega'] = shaft.omega
        else:
            self.outlet_bc['kin'].pop('omega', None)
        self._shaft = shaft


class VanelessDiffuser(BaseComponent):
    base_equations = [
        # Fundamental equations
        (ConstantEnergy, (0, 1)),
        (ConstantAngMomentum, (0, 1)),
        (MassConservation, (0, 1)),
        # Meridional Geometry
        (MeridionalUniform, 0),
        (MeridionalUniform, 1),
        # No blades
        (ZeroBlockage, 0),
        (ZeroBlockage, 1),
        # Extra definitions
        (GeometricalRatios, (0, 1)),
    ]

    constant_variables = [
        'kin_omega',
        'geo_meridional_angle',
    ]

    from_previous_node = ABSOLUTE_LINK + GEOM_LINK

    def _post_init(self):
        self.inlet_bc['kin']['omega'] = 0
        # WARN: Hypothesis => Null axial chord = exactly radial diffuser
        self.outlet_bc['geo']['chord_ax'] = 0


class DownstreamMixer(BaseComponent):
    base_equations = [
        # *** Fundamental
        (ConstantEnergy, (0, 1)),
        (MassConservation, (0, 1)),
        (MixingMomentumBalances, (0, 1)),
        (DeviationAngle, (0, 1)),
        # *** Blockage
        (BladeBlockage, 0),
        (ZeroBlockage, 1),  # No blockage mixed out
        # *** Definition of channel massflow and num_blades
        (BladePitch, 0),
        (BladePitch, 1),
        # Special adders
        (GeometricalAdder, 0),
        (GeometricalAdder, 1),
        (ZeroDeviation, 1),  # Creates a dummy metal angle (for plots)
    ]

    # Keep the absolute triangle
    # energy, meridional geometry
    from_previous_node = (
        ABSOLUTE_LINK
        + GEOM_LINK
        + [
            # Copy the geometry
            'geo_hh',
            'geo_rr',
            # Get the base pressure
            'oth_p_base',
            # Stay in the same MRF as blade row
            'kin_omega',
            # Geometry
            'geo_num_blades',
            'geo_metal_angle',
            # Boundary layer and blade thicknesses
            'geo_bld_thick',
            'oth_disp_thick',
            'oth_mom_thick',
        ]
    )

    constant_variables = GEOM_LINK + [
        # Keep the span geometry constant
        'geo_hh',
        'geo_rr',
        # Keep reference frame alive
        'kin_omega',
        # Keep geometry
        'geo_num_blades',
    ]


@dataclass
class StationGeometry:
    radius: float
    height: float
    meridional_angle: float
    z_offset: float

    def __post_init__(self):
        self.r1 = self.radius + self.height * np.cos(self.meridional_angle) / 2
        self.r2 = self.r1 - self.height * np.cos(self.meridional_angle)

        self.z2 = self.height * np.sin(self.meridional_angle) / 2
        self.z1 = -self.z2

        self.z1 += self.z_offset
        self.z2 += self.z_offset

    def get_line(self):
        return StraightLine(self.z1, self.z2, self.r1, self.r2)


@dataclass
class BladeData:
    z_in: float
    z_out: float
    radius_in: float
    radius_out: float
    angle_in: float
    angle_out: float

    def to_dict(self):
        return self.__dict__


@dataclass
class RowGeometry:
    r_in: float
    r_out: float
    height_in: float
    height_out: float
    mer_angle_in: float
    mer_angle_out: float
    axial_chord: float
    semi_cone_angle: bool = False
    axial_offset: float = 0.0

    def __post_init__(self):
        """
        Whether to add or not a semicone angle at the leading and trailing edge
        """

        self.in_geo = StationGeometry(
            self.r_in,
            self.height_in,
            self.mer_angle_in,
            self.axial_offset,
        )
        self.out_geo = StationGeometry(
            self.r_out,
            self.height_out,
            self.mer_angle_out,
            self.axial_offset + self.axial_chord,
        )

        self._build_curves()

    def _compute_parameters(self):
        if self.semi_cone_angle:
            delta_H = 0.5 * (self.height_out - self.height_in) / self.axial_chord
            tip_angle_le = self.in_geo.meridional_angle + np.arctan(delta_H)
            tip_angle_te = self.out_geo.meridional_angle + np.arctan(delta_H)

            hub_angle_le = self.in_geo.meridional_angle - np.arctan(delta_H)
            hub_angle_te = self.out_geo.meridional_angle - np.arctan(delta_H)

        else:
            tip_angle_le = self.in_geo.meridional_angle
            tip_angle_te = self.out_geo.meridional_angle

            hub_angle_le = self.in_geo.meridional_angle
            hub_angle_te = self.out_geo.meridional_angle

        tip_params = BladeData(
            self.in_geo.z1,
            self.out_geo.z1,
            self.in_geo.r1,
            self.out_geo.r1,
            tip_angle_le,
            tip_angle_te,
        )

        hub_params = BladeData(
            self.in_geo.z2,
            self.out_geo.z2,
            self.in_geo.r2,
            self.out_geo.r2,
            hub_angle_le,
            hub_angle_te,
        )

        return tip_params, hub_params

    def _build_curves(self):
        tip_params, hub_params = self._compute_parameters()
        try:
            self._tip_curve = BezierCurve(**tip_params.to_dict())
            self._hub_curve = BezierCurve(**hub_params.to_dict())
        except ValueError as e:
            logger.warning(f'Error creating bezier curves, defaulting to lines: {e}')
            self._tip_curve = StraightLine(**tip_params.to_dict())
            self._hub_curve = StraightLine(**hub_params.to_dict())
        except TypeError as e:
            raise TypeError(f'Impossible to create meridional curves: {e}')

        self._le_curve = self.in_geo.get_line()
        self._te_curve = self.out_geo.get_line()

    def plot_meridional_profile(
        self,
        color=None,
        debug: bool = False,
        ax=None,
    ) -> tuple[Line2D, ...]:
        """
        Plots the meridional profile of the blade row.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot on.
        filename : str, optional
            If provided, saves the plot to the given filename.
        """

        color = None if debug else color

        tip = self._tip_curve.plot_curve(color, ax=ax)
        hub = self._hub_curve.plot_curve(color, ax=ax)
        le = self._le_curve.plot_curve(color, ax=ax)
        te = self._te_curve.plot_curve(color, ax=ax)

        return (tip, hub, le, te)

    def get_meridional_channel_area(self):
        """
        Computes the meridional surface area of the blade row.

        This method calculates the total meridional surface area by taking the
        difference between the tip and hub curve areas, and then adding the leading
        and trailing edge surface areas with appropriate signs based on their
        geometric orientation.

        Returns
        -------
        float
            The total meridional surface area.

        Notes
        -----
        The calculation includes:

        - Difference between tip and hub curve areas
        - Leading edge area (signed based on axial position difference)
        - Trailing edge area (signed based on axial position difference)

        The surface area is stored in the ``meridional_surface`` attribute.
        """
        self.merdional_surface_area = self._tip_curve.area - self._hub_curve.area

        self._le_sign = self.in_geo.z2 - self.in_geo.z1
        self._te_sign = self.out_geo.z2 - self.out_geo.z1

        self.merdional_surface_area += np.sign(self._le_sign) * self._le_curve.area
        self.merdional_surface_area += -np.sign(self._te_sign) * self._te_curve.area


def plot_from_nodes(
    n0: FlowNode | None,
    n1: FlowNode | None,
    semi_cone_angle: bool = False,
    axial_offset: float = 0.0,
    color: tuple | str = 'k',
    ax=None,
):
    """
    Utility plot function, for now the chord is
    just specified manually, to be deleted
    """

    if n0 is None or n1 is None:
        raise AttributeError('None nodes found')

    TO_READ = ('rr_midspan', 'height', 'meridional_angle')

    args = []
    for var in TO_READ:
        for node in [n0, n1]:
            args.append(node.geo.get(var).to_base_units().magnitude[0])

    args.append(n1.geo.chord_ax[0])

    geom = RowGeometry(
        *args,
        semi_cone_angle=semi_cone_angle,
        axial_offset=axial_offset,
    )

    lines = geom.plot_meridional_profile(color, ax=ax)

    for line in lines:
        line.set_linewidth(2.5)

    return lines


def geometry_main(geo_inputs, color):
    """
    Testing geometry function
    """

    geo1 = RowGeometry(*geo_inputs, semi_cone_angle=True)
    geo2 = RowGeometry(*geo_inputs, semi_cone_angle=False)

    geo1.plot_meridional_profile('b', debug=False)
    lines = geo2.plot_meridional_profile(color, debug=False)

    [ln.set_linewidth(2.5) for ln in lines]


if __name__ == '__main__':
    from adet.tools.interpolation import TransfiniteInterpolator

    geo = RowGeometry(0.4, 1.0, 0.3, 0.2, np.pi / 2, np.pi / 2, 0.1)

    tip_curve = np.vstack(
        [
            geo._tip_curve.z_coords,
            geo._tip_curve.r_coords,
        ]
    )

    hub_curve = np.vstack(
        [
            geo._hub_curve.z_coords,
            geo._hub_curve.r_coords,
        ]
    )

    interp = TransfiniteInterpolator(hub_curve, tip_curve, 80, 20)
    interp.generate_grid()
    interp.plot()
