from dataclasses import dataclass
import logging
from typing import Type, Any

from matplotlib.lines import Line2D
import numpy as np

# Equation objects
from adet.equations.definitions import (
    AngleDeflection,
    BladeCount,
    HeightRatio,
    MeridionalVelocityRatio,
    Solidity,
)
from adet.equations.fundamental import (
    EulerEquation,
    MassConservation,
    ParabolicCamberline,
)
from adet.equations import EquationBase
from adet.equations.linkers import SpeedLinker, ComponentLinker
from adet.losses import LossModel

# Dependencies and tooling
from adet.losses.mixing import MixingBalances, RowMixerLink
from adet.node import FlowNode
from adet.components import BaseComponent, Shaft
from adet.geometry import BezierCurve, StraightLine
from adet.tools.strings import get_index

logger = logging.getLogger(__name__)


class BladeRow(BaseComponent):
    base_equations = [
        # ***
        # |> Fundamental equations - do not remove
        (EulerEquation, (0, 1)),
        (MassConservation, (0, 1)),
        # |> Link inlet and outlet omega
        (SpeedLinker, (1, 1)),
        (SpeedLinker, (1, 0)),
        # ***
        (HeightRatio, (0, 1)),
        (AngleDeflection, (0, 1)),
        (MeridionalVelocityRatio, (0, 1)),
        # ***
        # |> TODO: These are hardcoded for testing REMOVE!
        # |> Common courtesy definitions
        (Solidity, 1),
        (BladeCount, 1),
        (ParabolicCamberline, (0, 1)),  # Camber line geometry
    ]

    linker_equations = [ComponentLinker]

    def __init__(
        self,
        name: str,
        boundary_conditions: dict[
            str,
            dict[str, Any],
        ],
        shaft: Shaft,
        extra_equations: dict[
            EquationBase,
            int | tuple[int, ...],
        ] = {},
        loss_models: list[LossModel] = [],
    ):
        """
        Class that represents a blade row, compressor/turbine,
        stator/rotor
        """
        super().__init__(name, boundary_conditions, extra_equations)
        if shaft.is_constrained:
            self.boundary_conditions['kin']['omega'] = shaft.omega

        # NOTE: Loss models are being added just as equations now
        # self._loss_models = loss_models
        # self._add_loss_parameters()
        # self._build_loss_matcher()

    # TODO: Fix this for multiple formulations interacting
    # Total pressure, enthalpy, entropy, etc.
    # This below is unused for now, loss models are just added as equations
    # The code below was a minimal example but currently it's out of
    # order and missing major features.

    def _add_loss_parameters(self):
        raise NotImplementedError
        for model in self._loss_models:
            self.boundary_conditions['oth'].update(model.parameters)
            local_indices = {get_index(arg) for arg in model.arguments}
            self._equations[model] = tuple(local_indices)

    def _build_loss_matcher(self):
        raise NotImplementedError
        model_variables = [f'{model.VALUE_VARIABLE}' for model in self._loss_models]
        CLASS_NAME = 'EntropyProduction'

        code_gen = f"""
class {CLASS_NAME}(EquationBase):
    def residual(self, stc_smass1, stc_smass0, {', '.join(model_variables)}):
        return stc_smass1 - stc_smass0 - {'- '.join(model_variables)}
        """

        # Execute the code in an isolated namespace
        nspace: dict[str, Type[EquationBase]] = {}
        exec(code_gen, None, nspace)
        GenClass = nspace[CLASS_NAME]

        self._equations[GenClass()] = (0, 1)


class DownstreamMixer(BaseComponent):
    base_equations = [
        (MixingBalances, (0, 1)),
    ]

    linker_equations = [
        ComponentLinker,
        RowMixerLink,
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

        tip = self._tip_curve.plot_curve(color)
        hub = self._hub_curve.plot_curve(color)
        le = self._le_curve.plot_curve(color)
        te = self._te_curve.plot_curve(color)

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
    n0: FlowNode,
    n1: FlowNode,
    semi_cone_angle: bool = False,
    axial_offset: float = 0.0,
    color: tuple | str = 'k',
):
    """
    Utility plot function, for now the chord is
    just specified manually, to be deleted
    """

    TO_READ = ('rmid', 'height', 'meridional_angle')

    args = []
    for var in TO_READ:
        for node in [n0, n1]:
            args.append(node.geo.get(var).to_base_units().magnitude[0])

    args.append(node.geo.get('chord_ax').to_base_units().magnitude[0])

    geom = RowGeometry(
        *args,
        semi_cone_angle=semi_cone_angle,
        axial_offset=axial_offset,
    )

    lines = geom.plot_meridional_profile(color)

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

    geo = RowGeometry(0.4, 1.0, 0.3, 0.05, 0.0, np.pi / 2, 0.45)

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
