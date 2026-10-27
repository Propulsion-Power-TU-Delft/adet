"""Pre-assembled container for the quasi-3D streamline-curvature model
(hub-to-shroud + blade-to-blade flow, Van den Braembussche Secs. 3.1.1-3.1.2),
bundling the equations in ``adet.equations.streamline_curvature.equations``
onto a full ``(n_stream x n_span)`` grid -- analogous in spirit to
``adet.components.BladeRow``, but for this grid-shaped model rather than a
2-node inlet/outlet component.
"""

from typing import Any, Callable, cast

import numpy as np

from adet.equations.fundamental import Kinematics
from adet.equations.streamline_curvature.equations import (
    AbsoluteEnergyClosure,
    BladeSurfacePressures,
    BladeToBladeLoading,
    CarryLoadingToLE,
    LocalThermoState,
    MassFluxAccumulation,
    MeridionalMomentumMarch,
    RothalpyConservation,
    SlipTransition,
    SuctionPressureVelocities,
)
from adet.equations.streamline_curvature.geometry import (
    circular_beta_law,
    front_loaded_beta_law,
    linear_beta_law,
    parabolic_beta_law,
    rear_loaded_beta_law,
)
from adet.equations.streamline_curvature.variables import (
    BladeLoadingVariables,
    MeridionalFlowVariables,
)
from adet.fluid.ideal_eos import IdealGasState
from adet.losses.basic import ZeroDeviation
from adet.variables import NodeVariables

EquationPositions = dict[Any, int | tuple[int, int]]
BetaLaw = Callable[..., tuple[np.ndarray, np.ndarray]]

BETA_LAWS: dict[str, BetaLaw] = {
    'linear': linear_beta_law,
    'parabolic': parabolic_beta_law,
    'circular': circular_beta_law,
    'front_loaded': front_loaded_beta_law,
    'rear_loaded': rear_loaded_beta_law,
}
"""Blade-angle laws selectable by name through ``SurfaceFlow(beta_law=...)``,
see ``adet.equations.streamline_curvature.geometry`` for each one's
definition. ``'front_loaded'``/``'rear_loaded'`` need extra
``split_frac``/``frac_at_split``/``half_width`` parameters, passed via
``beta_law_kwargs``."""


class SurfaceFlow:
    """Pre-assembles the hub-to-shroud + blade-to-blade streamline-curvature
    equations onto an ``(n_stream x n_span)`` grid.

    Node ``(i, j)`` (streamwise index ``i``, spanwise index ``j``) maps to
    the flat node ``node_index(i, j)``; ``nodes``/``mf_nodes``/``bl_nodes``
    give the corresponding per-node variable accessors (``NodeVariables``,
    ``MeridionalFlowVariables``, ``BladeLoadingVariables``), to be reused for
    boundary conditions, guesses and post-processing exactly like the module
    -level node/mf_nodes/bl_nodes lists in the streamline-curvature examples.

    The leading/trailing edge each independently blend from a free (solved)
    flow angle onto the prescribed blade-metal-angle law via
    :class:`~adet.equations.streamline_curvature.equations.SlipTransition`
    (``use_slip_transition_le``/``_te``), and/or close the system with a
    zero-loading Kutta condition (``use_kutta_le``/``_te``, enforced as a
    boundary condition by :meth:`closure_boundary_conditions`). With the
    Kutta condition disabled at the LE, loading is instead carried back to
    it from the first downstream node
    (:class:`~.equations.CarryLoadingToLE`).

    At the TE specifically, the Kutta condition only actually leaves the
    flow angle free: it needs ``use_slip_transition_te=True`` too, since
    ``SlipTransition`` is what hands off from the fixed blade-angle law
    to that free value. With ``use_slip_transition_te=False`` (regardless
    of ``use_kutta_te``), ``ZeroDeviation`` is used all the way through
    the TE instead -- the flow angle there exactly matches the blade
    metal angle, and the Kutta ``DeltaW=0`` boundary condition is not
    imposed (``DeltaW`` at the TE is then whatever loading results, not
    forced to zero).

    ``include_blade_to_blade`` (default ``True``) toggles the blade-to-blade
    plane off entirely: with it ``False``, :meth:`assemble` only adds the
    hub-to-shroud equations (``Kinematics``, ``LocalThermoState``,
    ``MeridionalMomentumMarch``, ``MassFluxAccumulation``) -- no
    ``SuctionPressureVelocities``/``BladeSurfacePressures``/
    ``BladeToBladeLoading``/``SlipTransition``/``ZeroDeviation``/
    ``CarryLoadingToLE``, and the relative flow angle is instead a direct,
    per-node boundary condition from the prescribed ``beta_bl`` field
    (set by :meth:`closure_boundary_conditions`) rather than a solved
    unknown -- matching a prescribed-beta hub-to-shroud-only model like
    ``radial_impeller_meridional_flow.py``. ``use_slip_transition_le``/
    ``_te``, ``use_kutta_le``/``_te`` and ``rotation_sign`` are then unused.

    ``beta_law`` selects the blade-metal-angle distribution used by
    :meth:`beta_field` to turn each span's inlet/outlet blade angle into
    the full ``beta_bl``/``beta_bl_dshat`` fields :meth:`assemble` needs --
    either one of the names in :data:`BETA_LAWS` (``'linear'``,
    ``'parabolic'``, ``'circular'``, ``'front_loaded'``) or any callable
    with the same ``(s_hat, beta_in, beta_out, **beta_law_kwargs) ->
    (beta, dbeta_dshat)`` signature (see
    ``adet.equations.streamline_curvature.geometry``). ``'front_loaded'``
    requires ``split_frac``/``frac_at_split``/``half_width`` via
    ``beta_law_kwargs``.

    ```axial_absolute_inflow` (default ``False``) changes how the LE
    relative flow angle is closed. By default (``False``) it is a direct
    boundary condition, ``beta_bl[0, :] + incidence_le`` -- a prescribed
    incidence onto an otherwise-known blade inlet angle. With it ``True``,
    :meth:`closure_boundary_conditions` instead fixes the *absolute*
    tangential velocity to zero at the LE (uniform, purely axial absolute
    inflow -- the usual "swallowing capacity" closure: the axial velocity
    follows from the mass flow and annulus area, the peripheral speed from
    ``Omega`` x ``RDistr``, and the relative velocity triangle, hence the
    relative flow angle, is whatever those two then imply through
    ``Kinematics``). ``incidence_le`` is then unused: the LE relative flow
    angle is a free unknown instead of a boundary condition, generally
    different from the blade metal angle (a real aerodynamic incidence),
    and -- exactly as for any other free LE value -- blended onto the
    blade-angle law by :class:`SlipTransition` when
    ``use_slip_transition_le=True`` (recommended in this mode, since
    otherwise the stations right after the LE jump straight to
    ``ZeroDeviation``, i.e. exactly the blade metal angle).
    """

    def __init__(
        self,
        n_stream: int,
        n_span: int,
        *,
        include_blade_to_blade: bool = True,
        use_slip_transition_le: bool = True,
        use_slip_transition_te: bool = True,
        use_kutta_le: bool = True,
        use_kutta_te: bool = True,
        axial_absolute_inflow: bool = False,
        transition_frac_le: float = 0.25,
        transition_frac_te: float = 0.6,
        rotation_sign: float = 1.0,
        beta_law: str | BetaLaw = 'linear',
        beta_law_kwargs: dict[str, Any] | None = None,
    ):
        if n_stream < 2 or n_span < 2:
            raise ValueError('`n_stream` and `n_span` must both be >= 2')

        self.n_stream = n_stream
        self.n_span = n_span
        self.include_blade_to_blade = include_blade_to_blade
        self.use_slip_transition_le = use_slip_transition_le
        self.use_slip_transition_te = use_slip_transition_te
        self.use_kutta_le = use_kutta_le
        self.use_kutta_te = use_kutta_te
        self.axial_absolute_inflow = axial_absolute_inflow
        self.rotation_sign = rotation_sign

        if isinstance(beta_law, str):
            try:
                beta_law = BETA_LAWS[beta_law]
            except KeyError:
                raise ValueError(
                    f'Unknown beta_law {beta_law!r}, expected one of '
                    f'{list(BETA_LAWS)} or a callable'
                ) from None
        self.beta_law: BetaLaw = beta_law
        self.beta_law_kwargs = dict(beta_law_kwargs) if beta_law_kwargs else {}

        self.trailing_edge = n_stream - 1
        self.transition_star = int(round(transition_frac_te * self.trailing_edge))
        self.transition_star_in = max(
            1, int(round(transition_frac_le * self.trailing_edge))
        )
        self.le_blend_end = self.transition_star_in if use_slip_transition_le else 0
        # The trailing edge is only left free (pinned by the Kutta
        # DeltaW=0 condition instead of ZeroDeviation) when there's also
        # a SlipTransition blend connecting it back to the law -- without
        # that blend, Kutta has no equation to smoothly hand off from, so
        # ZeroDeviation is used all the way through the trailing edge
        # instead, and the Kutta condition is simply not imposed there
        # (see `closure_boundary_conditions`); DeltaW is then whatever
        # loading results, not forced to zero.
        self.last_fixed_beta_station = (
            self.trailing_edge - 1
            if (use_kutta_te and use_slip_transition_te)
            else self.trailing_edge
        )
        self.te_blend_start = (
            self.transition_star + 1
            if (use_slip_transition_te and use_kutta_te)
            else self.last_fixed_beta_station + 1
        )

        self.nodes = [NodeVariables(k) for k in range(self.n_nodes)]
        self.mf_nodes = [MeridionalFlowVariables(k) for k in range(self.n_nodes)]
        self.bl_nodes = [BladeLoadingVariables(k) for k in range(self.n_nodes)]

    @property
    def n_nodes(self) -> int:
        return self.n_stream * self.n_span

    def node_index(self, i: int, j: int) -> int:
        return i * self.n_span + j

    def beta_field(
        self,
        s_stations: np.ndarray,
        s_max: np.ndarray,
        beta_in_span: np.ndarray,
        beta_out_span: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``beta_law`` across the full grid.

        Parameters
        ----------
        s_stations : ndarray, shape ``(n_stream, n_span)``
            Streamwise arc length from the leading edge, per span (as
            used by :meth:`assemble`).
        s_max : ndarray, shape ``(n_span,)``
            Total streamwise arc length (trailing edge), per span.
        beta_in_span, beta_out_span : ndarray, shape ``(n_span,)``
            Blade metal angle at the leading/trailing edge, per span.

        Returns
        -------
        beta_bl, beta_bl_dshat : ndarray, shape ``(n_stream, n_span)``
            Directly usable as the ``beta_bl``/``beta_bl_dshat`` arguments
            to :meth:`assemble`.
        """
        s_hat = s_stations / s_max[None, :]
        beta_bl = np.empty((self.n_stream, self.n_span))
        beta_bl_dshat = np.empty((self.n_stream, self.n_span))
        for j in range(self.n_span):
            beta_bl[:, j], beta_bl_dshat[:, j] = self.beta_law(
                s_hat[:, j], beta_in_span[j], beta_out_span[j], **self.beta_law_kwargs
            )
        return beta_bl, beta_bl_dshat

    def assemble(
        self,
        system,
        *,
        total_enthalpy: float,
        entropy: float,
        gas_state: IdealGasState,
        omega: float,
        r: np.ndarray,
        gamma: np.ndarray,
        inv_rn: np.ndarray,
        delta_n: np.ndarray,
        delta_s: np.ndarray | None = None,
        s_stations: np.ndarray | None = None,
        s_max: np.ndarray | None = None,
        beta_bl: np.ndarray | None = None,
        beta_bl_dshat: np.ndarray | None = None,
    ) -> EquationPositions:
        """Add every hub-to-shroud (+ blade-to-blade, unless
        ``include_blade_to_blade=False``) equation to ``system``.

        Parameters
        ----------
        system : CasadiSystem
            The system equations are added to (``system.add_equation``).
        total_enthalpy : float
            Upstream absolute total enthalpy.
        entropy : float
            Shared (globally uniform, isentropic) entropy.
        gas_state : IdealGasState
            Used by :class:`LocalThermoState`/:class:`BladeSurfacePressures`.
        omega : float
            Shaft angular speed [rad/s].
        r, inv_rn : ndarray, shape ``(n_stream, n_span)``
            Local radius and spanwise normal curvature (1/Rn).
        gamma : ndarray, shape ``(n_stream,)``
            Meridional midline tangent angle.
        delta_n : ndarray, shape ``(n_stream, n_span - 1)``
            Spanwise spacing between neighboring quasi streamsurfaces.
        delta_s : ndarray, shape ``(n_stream - 1, n_span)``
            Streamwise spacing between neighboring stations, per span.
            Required unless ``include_blade_to_blade=False``.
        s_stations : ndarray, shape ``(n_stream, n_span)``
            Streamwise arc length from the leading edge, per span.
            Required unless ``include_blade_to_blade=False``.
        s_max : ndarray, shape ``(n_span,)``
            Total streamwise arc length (trailing edge), per span.
            Required unless ``include_blade_to_blade=False``.
        beta_bl, beta_bl_dshat : ndarray, shape ``(n_stream, n_span)``
            Blade metal angle and its ``d(beta)/d(s_hat)`` derivative
            (streamwise fraction, per span) -- see
            :func:`~adet.equations.streamline_curvature.geometry.front_loaded_beta_law`.
            Required unless ``include_blade_to_blade=False``.

        Returns
        -------
        dict
            The ``{equation: position}`` mapping added to ``system``, in
            the same format ``system.add_equation`` takes (for inspection).
        """
        n_stream, n_span = self.n_stream, self.n_span
        include_b2b = self.include_blade_to_blade

        if include_b2b:
            if (
                delta_s is None
                or s_stations is None
                or s_max is None
                or beta_bl is None
                or beta_bl_dshat is None
            ):
                raise ValueError(
                    '`delta_s`, `s_stations`, `s_max`, `beta_bl` and '
                    '`beta_bl_dshat` are required when `include_blade_to_blade=True`'
                )
            beta_bl_slope = beta_bl_dshat[self.transition_star, :] / s_max
            beta_bl_slope_in = beta_bl_dshat[self.transition_star_in, :] / s_max

        equations: EquationPositions = {}
        for j in range(n_span):
            for i in range(n_stream):
                k = self.node_index(i, j)
                equations[Kinematics()] = k
                equations[LocalThermoState(entropy=entropy, gas_state=gas_state)] = k

                # Local static enthalpy is a free, solved quantity: it's
                # anchored once per streamline at the LE from the known
                # upstream absolute total enthalpy, then propagated by
                # rothalpy conservation -- never a single rothalpy
                # constant forced onto the whole domain (see
                # AbsoluteEnergyClosure/RothalpyConservation docstrings).
                if i == 0:
                    equations[AbsoluteEnergyClosure(total_enthalpy=total_enthalpy)] = k
                else:
                    equations[RothalpyConservation()] = (
                        self.node_index(i - 1, j),
                        k,
                    )

                if not include_b2b:
                    # No blade-to-blade equation ties the flow angle to
                    # anything: it's a direct boundary condition at every
                    # node instead (see `closure_boundary_conditions`).
                    continue

                equations[SuctionPressureVelocities()] = k
                equations[
                    BladeSurfacePressures(entropy=entropy, gas_state=gas_state)
                ] = k

                if i == 0:
                    # LE flow angle is always a direct boundary condition
                    # (see `closure_boundary_conditions`).
                    pass
                elif i <= self.le_blend_end:
                    s_stations_ = cast(np.ndarray, s_stations)
                    beta_bl_ = cast(np.ndarray, beta_bl)
                    equations[
                        SlipTransition(
                            s_i=s_stations_[i, j],
                            s_star=s_stations_[self.transition_star_in, j],
                            s_end=0.0,
                            beta_bl_star=beta_bl_[self.transition_star_in, j],
                            slope_star=beta_bl_slope_in[j],
                        )
                    ] = (k, self.node_index(0, j))
                elif i <= self.last_fixed_beta_station and i < self.te_blend_start:
                    equations[ZeroDeviation()] = k
                elif i <= self.last_fixed_beta_station:
                    s_stations_ = cast(np.ndarray, s_stations)
                    s_max_ = cast(np.ndarray, s_max)
                    beta_bl_ = cast(np.ndarray, beta_bl)
                    equations[
                        SlipTransition(
                            s_i=s_stations_[i, j],
                            s_star=s_stations_[self.transition_star, j],
                            s_end=s_max_[j],
                            beta_bl_star=beta_bl_[self.transition_star, j],
                            slope_star=beta_bl_slope[j],
                        )
                    ] = (k, self.node_index(self.trailing_edge, j))
                # Beyond `last_fixed_beta_station` (only possible at the
                # trailing edge, with the Kutta condition on): the flow
                # angle is left free, pinned by `closure_boundary_conditions`.

            if not include_b2b:
                continue

            delta_s_ = cast(np.ndarray, delta_s)
            for i in range(1, n_stream):
                equations[
                    BladeToBladeLoading(
                        delta_s=delta_s_[i - 1, j], rotation_sign=self.rotation_sign
                    )
                ] = (self.node_index(i - 1, j), self.node_index(i, j))

            if not self.use_kutta_le:
                equations[CarryLoadingToLE()] = (
                    self.node_index(0, j),
                    self.node_index(1, j),
                )

        for i in range(n_stream):
            for j in range(1, n_span):
                k0, k1 = self.node_index(i, j - 1), self.node_index(i, j)
                equations[
                    MeridionalMomentumMarch(
                        delta_n=delta_n[i, j - 1],
                        r0=r[i, j - 1],
                        r1=r[i, j],
                        gamma=gamma[i],
                        omega=omega,
                        inv_rn0=inv_rn[i, j - 1],
                        inv_rn1=inv_rn[i, j],
                    )
                ] = (k0, k1)
                equations[
                    MassFluxAccumulation(
                        delta_n=delta_n[i, j - 1], r0=r[i, j - 1], r1=r[i, j]
                    )
                ] = (k0, k1)

        for eq, pos in equations.items():
            system.add_equation(eq, pos)

        return equations

    def closure_boundary_conditions(
        self,
        *,
        beta_bl: np.ndarray,
        mass_flow: float,
        incidence_le: float = 0.0,
    ) -> dict:
        """Boundary conditions that close the system set up by
        :meth:`assemble`: the Eq. (3.6) integration constants (``CumFlow``
        = 0 at the hub, = ``mass_flow`` at the shroud, at every streamwise
        station), plus either (with ``include_blade_to_blade=True``) a LE
        closure and the Kutta zero-loading condition (``DeltaW`` = 0) at
        whichever edge(s) have it enabled, or (with
        ``include_blade_to_blade=False``) the relative flow angle directly
        prescribed from ``beta_bl`` at *every* node, since no blade-to-blade
        equation ties it to anything otherwise.

        The LE closure is one of two kinds, set by
        ``axial_absolute_inflow`` at construction:

        - ``False`` (default): the relative flow angle itself is the
          boundary condition, ``beta_bl[0, :] + incidence_le`` -- a
          prescribed incidence onto a known blade inlet angle.
        - ``True``: the *absolute* tangential velocity is fixed to zero
          instead (uniform, purely axial absolute inflow -- the
          swallowing-capacity closure), leaving the LE relative flow
          angle a free unknown that ``Kinematics`` determines from the
          axial velocity and ``Omega`` x ``RDistr``; ``incidence_le`` is
          then unused.

        Parameters
        ----------
        beta_bl : ndarray, shape ``(n_stream, n_span)``
            Same blade metal angle field passed to :meth:`assemble`.
        mass_flow : float
            Target mass flow rate [kg/s].
        incidence_le : float
            Deviation of the solved LE flow angle from ``beta_bl[0, :]``
            (0 = true zero incidence). Unused when
            ``include_blade_to_blade=False`` (the whole field is
            prescribed as-is, with no concept of slip/incidence) or when
            ``axial_absolute_inflow=True`` (the LE incidence is then
            whatever the velocity triangle implies, not prescribed).
        """
        boundary_conditions = {}
        for i in range(self.n_stream):
            boundary_conditions[self.mf_nodes[self.node_index(i, 0)].CumFlow] = 0.0
            boundary_conditions[
                self.mf_nodes[self.node_index(i, self.n_span - 1)].CumFlow
            ] = mass_flow

        if not self.include_blade_to_blade:
            for i in range(self.n_stream):
                for j in range(self.n_span):
                    boundary_conditions[
                        self.nodes[self.node_index(i, j)].kin.FlowAngleRel
                    ] = beta_bl[i, j]
            return boundary_conditions

        for j in range(self.n_span):
            if self.axial_absolute_inflow:
                boundary_conditions[self.nodes[self.node_index(0, j)].kin.V_tan] = 0.0
            else:
                boundary_conditions[
                    self.nodes[self.node_index(0, j)].kin.FlowAngleRel
                ] = beta_bl[0, j] + incidence_le
            if self.use_kutta_le:
                boundary_conditions[self.bl_nodes[self.node_index(0, j)].DeltaW] = 0.0
            if self.use_kutta_te and self.use_slip_transition_te:
                boundary_conditions[
                    self.bl_nodes[self.node_index(self.trailing_edge, j)].DeltaW
                ] = 0.0

        return boundary_conditions
