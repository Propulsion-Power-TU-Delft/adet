import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.lines import Line2D

from adet.components import BaseComponent, Shaft
from adet.equations import EquationBase
from adet.equations.fundamental import (
    BladeBlockage,
    ConstantAngMomentum,
    ConstRelEnthalpy,
    EulerEquation,
    MassConservation,
    ZeroBlockage,
)
from adet.equations.geometrical import (
    BladePitch,
    BladeRatios,
    EndwallProperties,
    MeridionalGeometry,
    MeridionalRatios,
    MinimalCamberLine,
    RadialGeometry,
)
from adet.geometry import BezierCurve, StraightLine
from adet.losses.basic import ZeroDeviation
from adet.losses.compressors import AmiranteDiffuserMomentum
from adet.variables import (
    GeometricVariables,
    KinematicVariables,
    NodeVariables,
    OtherVariables,
    ThermoVariables,
)
from adet.varspec import DEF_NODE, NodeStates, VarSpec

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Initialize variable enums with DEF_NODE to get base VarSpec objects
_kin = KinematicVariables(node=DEF_NODE)
_geo = GeometricVariables(node=DEF_NODE)
_oth = OtherVariables(node=DEF_NODE)
_thrm_tot = ThermoVariables(node=DEF_NODE, state=NodeStates.TOTAL)
_thrm_stc = ThermoVariables(node=DEF_NODE, state=NodeStates.STATIC)

n0 = NodeVariables(0)
n1 = NodeVariables(1)

ABSOLUTE_LINK = [
    # Absolute triangle
    _kin.V_tan,
    _kin.V_mer,
    # No work & no entropy
    _thrm_stc.Entropy,
    _thrm_tot.Enthalpy,
]
"""
This preserves the absolute triangles,
energy. Therefore it is a
full variable transfer except for the
rotating frame (omega)
"""

# Geometry
GEOM_LINK = [
    _geo.Rmid,
    _geo.Height,
    _geo.MeridionalAngle,
]
_oth = OtherVariables(node=DEF_NODE)


class BladeRow(BaseComponent):
    base_equations = [
        # *** Conservation equations
        (EulerEquation, (0, 1)),
        (MassConservation, (0, 1)),
        # *** Meridional geometry
        (MeridionalGeometry, 0),
        (MeridionalGeometry, 1),
        (MeridionalRatios, (0, 1)),
        # *** Blockage - Zero by default
        (ZeroBlockage, 0),
        (ZeroBlockage, 1),
        # *** Out Deviation - Zero by default
        (ZeroDeviation, 1),
        # *** Row Geometry
        (BladePitch, 0),
        (BladePitch, 1),
        (BladeRatios, 0),
        (BladeRatios, 1),
        (MinimalCamberLine, (0, 1)),
        # *** Common definitions (OPTIONAL)
        (EndwallProperties, 0),
        (EndwallProperties, 1),
    ]

    from_previous_node = ABSOLUTE_LINK + GEOM_LINK

    # Stored on both nodes
    constant_variables = [
        _kin.Omega,
        _geo.Chord,
        _geo.NumBlades,
    ]

    def __init__(
        self,
        name: str,
        shaft: Shaft,
        bound_cond: dict[VarSpec, Any] = {},
        extra_equations: dict[
            EquationBase,
            int | tuple[int, ...],
        ] = {},
        constant_variables: list[VarSpec] = [],
        spanwise_constants: list[VarSpec] = [],
        from_prev_node: list[VarSpec] = [],
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
            bound_cond,
            extra_equations,
            constant_variables,
            spanwise_constants,
            from_prev_node,
        )
        self._shaft = None
        self.shaft = shaft  # This uses the setter logic

    def _post_init(self):
        # Set zero thickness if not specified
        for node in [n0, n1]:
            if (
                node.geo.BldThick not in self._boundary_conditions
                and node.geo.ThickByPitch not in self._boundary_conditions
            ):
                self._boundary_conditions[node.geo.BldThick] = 0

        self._boundary_conditions = {
            **{
                # *** Profile losses params
                n1.oth.CdProfile: 0.002,
                n1.oth.XiCambLenA: 0.375,
                n1.oth.XiCambLenB: 0.675,
            },
            **self._boundary_conditions,
        }

    @property
    def shaft(self) -> Shaft | None:
        return self._shaft

    @shaft.setter
    def shaft(self, shaft: Shaft):
        self._shaft = shaft
        from adet.variables import NodeVariables

        n1 = NodeVariables(1)
        if shaft.is_constrained:
            # Fix omega at the outlet node
            self._boundary_conditions[n1.kin.Omega] = shaft.omega
        else:
            # When switching from a fixed to nonfixed shaft
            # remove omega from the boundary conditions
            self._boundary_conditions.pop(n1.kin.Omega, None)

    @property
    def row_type(self) -> str:
        if self._shaft and self._shaft.is_constrained:
            omega_value = (
                self._shaft.omega.magnitude
                if hasattr(self._shaft.omega, 'magnitude')
                else self._shaft.omega
            )
            if omega_value == 0:
                return 'stator'
        return 'rotor'


class VanelessDiffuser(BaseComponent):
    base_equations = [
        # Fundamental equations
        (ConstRelEnthalpy, (0, 1)),
        # (ConstantAngMomentum, (0, 1)),
        (AmiranteDiffuserMomentum, (0, 1)),
        # (JansenDiffuserLoss, (0, 1)),
        (MassConservation, (0, 1)),
        # Meridional Geometry
        (MeridionalGeometry, 1),
        # No blades
        (ZeroBlockage, 0),
        (ZeroBlockage, 1),
        # Extra definitions
        (RadialGeometry, (0, 1)),
    ]

    constant_variables = [
        _kin.Omega,
    ]

    from_previous_node = (
        ABSOLUTE_LINK
        + GEOM_LINK
        + [
            _geo.HDistr,
            _geo.RDistr,
            _oth.WakeFrac,
        ]
    )

    def _post_init(self):
        self._boundary_conditions[n1.kin.Omega] = 0
        # NOTE: Null axial chord => exactly radial diffuser
        self._boundary_conditions[n1.geo.ChordAx] = 0.0


class Interspace(BaseComponent):
    base_equations = [
        # Fundamental equations
        (ConstRelEnthalpy, (0, 1)),
        (MassConservation, (0, 1)),
        (ConstantAngMomentum, (0, 1)),
        # Meridional Geometry
        (MeridionalGeometry, 1),
        (RadialGeometry, (0, 1)),
        # No blades
        (ZeroBlockage, 0),
        (ZeroBlockage, 1),
    ]

    constant_variables = [
        _kin.Omega,
    ]

    from_previous_node = (
        ABSOLUTE_LINK
        + GEOM_LINK
        + [
            _geo.HDistr,
            _geo.RDistr,
        ]
    )

    def _post_init(self):
        self._boundary_conditions[n1.kin.Omega] = 0
        # NOTE: Null axial chord => exactly radial diffuser
        self._boundary_conditions[n1.geo.ChordAx] = 0


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
    force_straight: bool = False

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
            raise TypeError(f'Impossible to create meridional curves: {e}') from e

        if self.force_straight:
            self._tip_curve = StraightLine(**tip_params.to_dict())
            self._hub_curve = StraightLine(**hub_params.to_dict())

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
