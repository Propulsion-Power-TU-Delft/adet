"""Module that gathers fundamental equations for internal flows"""

import numpy as np
import matplotlib.pyplot as plt

from adet.equations import EquationBase
from adet.tools.interpolation import safe_min_clip, fin_diff


class EulerEquation(EquationBase):
    def residual(self, tot_hmass0, kin_U0, kin_Vt0, tot_hmass1, kin_U1, kin_Vt1):
        return (tot_hmass1 - tot_hmass0) - (kin_U1 * kin_Vt1 - kin_U0 * kin_Vt0)


class MassConservation(EquationBase):
    def residual(self, oth_massflow0, oth_massflow1):
        return oth_massflow0 - oth_massflow1


class MassAreaRelation(EquationBase):
    """
    .. math::
        \\dot_{m} = \\rho_0 V_{m0} A_0
    """

    def residual(self, kin_Vm0, geo_area0, stc_rhomass0, oth_massflow0):
        return oth_massflow0 - stc_rhomass0 * kin_Vm0 * geo_area0


class TotalStaticMatching(EquationBase):
    """
    Match the total and static states imposing equal
    entropy and

    .. math::
        h_{t0} = h_0 + \\frac{V_0^2}{2}

    .. math::
        h_{t0}^{rel} = h_0 + \\frac{W_0^2}{2}

    Note
    ----
    The total and relative total entropy do not have a real
    physical sense, as the total and rel. tot. states are defined
    by an isentropic alting of the flow.

    Nonetheless, our formulation treats the
    three states as independent equations of state, which are
    matched in an Equation-Oriented using this equation.

    Observe that mathematically the entropy equality does not affect
    in any way the convergence of the system, and it is immediatly
    satisfied after the first N-R Iteration
    """

    def residual(
        self,
        tot_hmass0,
        stc_hmass0,
        rlt_hmass0,
        tot_smass0,
        stc_smass0,
        rlt_smass0,
        kin_V0,
        kin_W0,
        # Force to add density as variables in all states
    ):
        r1 = tot_hmass0 - (stc_hmass0 + kin_V0**2 / 2)
        r2 = rlt_hmass0 - (stc_hmass0 + kin_W0**2 / 2)
        r3 = tot_smass0 - stc_smass0
        r4 = rlt_smass0 - stc_smass0

        return r1, r2, r3, r4


class SimpleRadialEquilibrium(EquationBase):
    """
    Most implementation of a radial equilibrium,
    zero streamline curvature is assumed
    """

    skip_unit_check = True
    manual_units = ('J / kg / m',)

    def residual(self, geo_rr0, stc_p0, kin_Vt0, stc_rhomass0):
        dp_dr = fin_diff(stc_p0, geo_rr0)
        return dp_dr / stc_rhomass0 - kin_Vt0**2 / geo_rr0


class NisRe(EquationBase):
    """Non-ISentropic Radial Equilibrium"""

    skip_unit_check = True
    manual_units = ('J / kg / m',)

    def residual(self, geo_rr0, kin_Vt0, kin_Vm0, tot_hmass0, stc_T0, stc_smass0):
        dVt_dr = fin_diff(kin_Vt0, geo_rr0)
        dVm_dr = fin_diff(kin_Vm0, geo_rr0)
        dht_dr = fin_diff(tot_hmass0, geo_rr0)
        ds_dr = fin_diff(stc_smass0, geo_rr0)

        lhs = kin_Vm0 * dVm_dr + kin_Vt0 * dVt_dr + kin_Vt0**2 / geo_rr0
        rhs = dht_dr - stc_T0 * ds_dr
        return lhs - rhs


class FreeVortexDistribution(EquationBase):
    def residual(self, geo_rr0, kin_Vt0, geo_rmid0, oth_Vt_mid0):
        return geo_rr0 * kin_Vt0 - geo_rmid0 * oth_Vt_mid0


class MidspanAngle(EquationBase):
    def residual(self, kin_alpha0, oth_alpha_mid0):
        n_span = max(kin_alpha0.shape)
        return oth_alpha_mid0 - kin_alpha0[n_span // 2]


class ForcedVortexDistribution(EquationBase):
    def residual(self, geo_rr0, kin_Vt0, geo_rmid0, oth_Vtmid0):
        return kin_Vt0 / geo_rr0 - oth_Vtmid0 / geo_rmid0


class GeneralWhirl(EquationBase):
    def residual(self, geo_rr0, kin_Vt0, gen_whirl_a, gen_whirl_b, gen_whirl_n):
        return kin_Vt0 - gen_whirl_a * geo_rr0**gen_whirl_n + gen_whirl_b / geo_rr0


class Kinematics(EquationBase):
    def residual(
        self,
        kin_V0,
        kin_Vm0,
        kin_Vt0,
        kin_W0,
        kin_Wt0,
        kin_Wm0,
        kin_U0,
        kin_alpha0,
        kin_beta0,
    ):
        # Only if Vm and Vt are zero the denominator
        # nullifies, but Vm > 0 always, thus the
        # square root should pose no problems
        r1 = kin_V0 - (kin_Vm0**2 + kin_Vt0**2) ** 0.5
        r2 = kin_W0 - (kin_Wm0**2 + kin_Wt0**2) ** 0.5

        r3 = kin_Vm0 - kin_Wm0
        r4 = kin_Vt0 - (kin_Wt0 + kin_U0)

        # atan2 ensures that the angles are between
        # - pi / 2 and pi / 2
        r5 = kin_alpha0 - np.atan2(kin_Vt0, kin_Vm0)
        r6 = kin_beta0 - np.atan2(kin_Wt0, kin_Wm0)

        # Alternative formulation
        # r5 = kin_Wm0 - kin_W0 * np.cos(kin_beta0)
        # r6 = kin_Vm0 - kin_V0 * np.cos(kin_alpha0)

        return r1, r2, r3, r4, r5, r6


class MeridionalUniform(EquationBase):
    # = * = * = * = * = * = * = * = * = * = * = * = * = * = *
    # BOUNTY:                                               =
    # > Add differential equation for streamline curvature  *
    # > instead of uniform distribution                     =
    # = * = * = * = * = * = * = * = * = * = * = * = * = * = *

    def residual(
        self,
        geo_rr0,
        geo_rmid0,
        geo_height0,
        geo_hh0,
        geo_meridional_angle0,
        geo_area0,
    ):
        #          ==    rr[n] + hh[n] / 2
        #           \
        #  |\        +  <- rr[n]
        #  |_\        \
        #  |  \       ==  rr[n] + hh[n] / 2
        #   mer_angle

        spanwise_stations = max(geo_rr0.shape)
        if spanwise_stations == 1:
            r1 = geo_rr0 - geo_rmid0
        else:
            unit_space = np.linspace(0, 1, spanwise_stations)

            # Segment between innermost and outermost stations
            # quasi_height = (spanwise_stations - 1) * geo_hh0
            quasi_height = (spanwise_stations - 1) * geo_hh0

            r_hub = geo_rmid0 - quasi_height / 2 * np.cos(geo_meridional_angle0)

            r1 = geo_rr0 - (
                r_hub + unit_space * quasi_height * np.cos(geo_meridional_angle0)
            )

        # The power enforces hh to be positive, otherwise a negative
        # height, massflow, etc. is techincally a valid solution
        r2 = geo_hh0 - geo_height0 / spanwise_stations

        # Circular annuli at various spanwise
        r3 = geo_area0 - np.pi * (
            (geo_rr0 + geo_hh0 / 2) ** 2 - (geo_rr0 - geo_hh0 / 2) ** 2
        )

        return r1, r2, r3


class ParabolicCamberline(EquationBase):
    skip_unit_check = True
    manual_units = ('m', 'm', 'rad')

    @staticmethod
    def _compute_parabola(inlet_angle, outlet_angle, chord_ax):
        tan0 = np.tan(inlet_angle)
        tan1 = np.tan(outlet_angle)

        a = (tan1 - tan0) / (2 * chord_ax)
        b = tan0

        a = safe_min_clip(a, 1e-3)

        y_out = a * chord_ax**2 + b * chord_ax
        stagger = np.arctan(y_out / chord_ax)

        return a, b, stagger

    @staticmethod
    def _parabolic_arc_len(a, b, chord_ax):
        """Exact arc length of y = ax² + bx from x = 0 to x = chord_ax"""
        term1 = 2 * a * chord_ax + b
        term0 = b

        sqrt1 = np.sqrt(1 + term1**2)
        sqrt0 = np.sqrt(1 + term0**2)

        asinh1 = np.arcsinh(term1)
        asinh0 = np.arcsinh(term0)

        length = (1 / (4 * a)) * (term1 * sqrt1 + asinh1 - term0 * sqrt0 - asinh0)

        return length

    def residual(
        self,
        geo_metal_angle0,
        geo_metal_angle1,
        geo_chord1,
        geo_stagger1,
        geo_chord_ax1,
        geo_camb_len1,
        geo_pitch1,
    ):
        a, b, stagger = self._compute_parabola(
            geo_metal_angle0, geo_metal_angle1, geo_chord_ax1
        )
        arc_len = self._parabolic_arc_len(a, b, geo_chord_ax1)

        r1 = geo_chord_ax1 - geo_chord1 * np.cos(geo_stagger1)
        r2 = geo_camb_len1 - arc_len
        r3 = geo_stagger1 - stagger
        return r1, r2, r3

    # CAMBERLINE PLOTTING FUNCTIONS
    def plot_camber_line(
        self, axis, inlet_angle, outlet_angle, chord_ax, color, pitch=None
    ):
        a, b, _ = self._compute_parabola(inlet_angle, outlet_angle, chord_ax)
        N_PTS = 50
        x = np.linspace(0, chord_ax, N_PTS)
        y = a * x**2 + b * x

        axis.plot(x, y, color=color)
        if pitch is not None:
            axis.plot(x, y + pitch, color=color)

    def plot_3d_blade(
        self,
        inlet_angles,
        outlet_angles,
        chord_ax_values,
        radial_positions=None,
        n_points=50,
        axis=None,
        color='blue',
        alpha=0.7,
        axial_offset=0.0,
        label=None,
    ):
        """
        Plot 3D blade profile across multiple spanwise stations

        Parameters
        ----------
        inlet_angles : array-like
            Inlet flow angles at each spanwise station [rad]
        outlet_angles : array-like
            Outlet flow angles at each spanwise station [rad]
        chord_ax_values : array-like
            Axial chord lengths at each spanwise station [m]
        radial_positions : array-like, optional
            Radial positions of spanwise stations [m]
            If None, uses unit spacing from 0 to n_stations-1
        n_points : int, optional
            Number of points along each camber line (default: 50)
        axis : matplotlib 3D axis, optional
            Axis to plot on. If None, creates new figure
        color : str, optional
            Color for the blade surface (default: 'blue')
        alpha : float, optional
            Transparency of blade surface (default: 0.7)
        axial_offset : float, optional
            Axial offset to shift the blade along x-axis (default: 0.0)
        label : str, optional
            Label for legend (default: None)

        Returns
        -------
        fig, ax : matplotlib figure and axis objects
        """
        inlet_angles = np.atleast_1d(inlet_angles)
        outlet_angles = np.atleast_1d(outlet_angles)
        chord_ax_values = np.atleast_1d(chord_ax_values)

        n_stations = len(inlet_angles)

        if radial_positions is None:
            radial_positions = np.linspace(0, 1, n_stations)
        else:
            radial_positions = np.atleast_1d(radial_positions)

        # Validate inputs
        if not (
            len(inlet_angles)
            == len(outlet_angles)
            == len(chord_ax_values)
            == len(radial_positions)
        ):
            raise ValueError('All input arrays must have the same length')

        # Create figure if axis not provided
        if axis is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axis
            fig = ax.get_figure()

        # Generate camber lines for each spanwise station
        X = []  # axial direction
        Y = []  # tangential direction
        Z = []  # radial/spanwise direction

        # First pass: compute midpoint of midspan station for reference
        midspan_idx = n_stations // 2
        a_mid, b_mid, _ = self._compute_parabola(
            inlet_angles[midspan_idx],
            outlet_angles[midspan_idx],
            chord_ax_values[midspan_idx],
        )
        x_mid_ref = chord_ax_values[midspan_idx] / 2
        y_mid_ref = a_mid * x_mid_ref**2 + b_mid * x_mid_ref

        for i in range(n_stations):
            a, b, _ = self._compute_parabola(
                inlet_angles[i], outlet_angles[i], chord_ax_values[i]
            )

            # Compute midpoint of current parabola
            x_mid_current = chord_ax_values[i] / 2
            y_mid_current = a * x_mid_current**2 + b * x_mid_current

            # Calculate offset to align midpoint with midspan reference
            y_offset = y_mid_ref - y_mid_current

            # Generate parabola with alignment offset
            x = np.linspace(0, chord_ax_values[i], n_points) + axial_offset
            y = a * (x - axial_offset) ** 2 + b * (x - axial_offset) + y_offset
            z = np.full_like(x, radial_positions[i])

            X.append(x)
            Y.append(y)
            Z.append(z)

            # Plot individual camber lines (only add label on first iteration)
            if i == 0 and label is not None:
                ax.plot(x, y, z, color=color, alpha=0.3, linewidth=1, label=label)
            else:
                ax.plot(x, y, z, color=color, alpha=0.3, linewidth=1)

        # Convert to arrays for surface plotting
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)

        # Plot surface connecting spanwise stations
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, edgecolor='none')

        # Add wireframe for better visualization
        ax.plot_wireframe(X, Y, Z, color='black', alpha=0.2, linewidth=0.5)

        # Labels and formatting
        ax.set_xlabel('Axial Direction [m]')
        ax.set_ylabel('Tangential Direction [m]')
        ax.set_zlabel('Radial Position [m]')
        ax.set_title('3D Blade Profile')

        # Set aspect ratio to be more realistic
        ax.set_box_aspect([1, 1, 1])

        return fig, ax


if __name__ == '__main__':
    pbc = ParabolicCamberline()
    angle_in = 0
    N_PROFILES = 50
    angles_out = np.linspace(-np.pi / 2.5, np.pi / 2.5, N_PROFILES)
    chord_axial = 0.15

    # 2D Analysis plots
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    staggers = []
    arc_lengths = []
    cm = plt.get_cmap('autumn')
    for i, a_out in enumerate(angles_out):
        a, b, stag = pbc._compute_parabola(angle_in, a_out, chord_axial)
        staggers.append(stag)
        arc_len = pbc._parabolic_arc_len(a, b, chord_axial)
        arc_lengths.append(arc_len)
        pbc.plot_camber_line(ax[0], angle_in, a_out, chord_axial, cm(i / N_PROFILES))
        ax[0].set_aspect('equal')
        ax[0].set_title('Camber lines')
        ax[0].grid()

    ax[1].set_title('Stagger angles')
    ax[1].plot(angles_out, staggers)

    ax[2].set_title('Arc lengths')
    ax[2].plot(angles_out, arc_lengths)

    plt.tight_layout()

    # 3D Blade visualization
    # Example: blade with varying turning along span
    radii = np.linspace(0.25, 0.75, N_PROFILES)  # Hub to tip [m]
    inlet_angles_3d = np.linspace(np.deg2rad(0), np.deg2rad(0), N_PROFILES)
    # outlet_angles_3d = np.linspace(np.deg2rad(-20), np.deg2rad(-40), n_spanwise)
    chord_ax_3d = np.linspace(0.12, 0.15, N_PROFILES)  # Varying chord

    fig_3d, ax_3d = pbc.plot_3d_blade(
        inlet_angles=inlet_angles_3d,
        outlet_angles=angles_out,
        chord_ax_values=chord_ax_3d,
        radial_positions=radii,
        color='steelblue',
        alpha=0.6,
        axial_offset=0.4,
    )
    ax_3d.view_init(elev=20, azim=45)

    plt.show()
