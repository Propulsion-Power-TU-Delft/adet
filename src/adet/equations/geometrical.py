import matplotlib.pyplot as plt
import numpy as np

from adet.equations.base_equation import (
    CamberLineGeom,
    EquationBase,
    MeridionalGeom,
    EquationConfig,
)
from adet.equations.utils import (
    get_midspan_idx,
    safe_abs,
    safe_max,
    safe_min_clip,
    safe_sum,
)
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)

# NOTE:
# Meridional Geometry
#
#          ==    rr[n] + hh[n] / 2
#           \
#  |\        +  <- rr[n]
#  |_\        \
#  |  \       ==  rr[n] - hh[n] / 2
#   mer_angle


class MeridionalGeometry(MeridionalGeom):
    # N + 1 equations
    # NOTE: = 2N when N == 1
    def residual(
        self,
        rr0: n0.geo.RDistr.Hint,
        hh0: n0.geo.HDistr.Hint,
        h0: n0.geo.Height.Hint,
        rr_mid0: n0.geo.Rmid.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
    ):
        residuals = []

        _min_radius = rr_mid0 - (h0 - hh0[0]) / 2 * np.cos(mer_angle0)

        if max(rr0.shape) > 1:
            r1 = rr0[:-1] + (hh0[:-1] + hh0[1:]) / 2 * np.cos(mer_angle0) - rr0[1:]
            residuals.append(r1)
            r2 = rr0[0] - _min_radius
        else:
            r2 = rr0[0] - rr_mid0

        r3 = safe_sum(hh0) - h0

        residuals.extend([r2, r3])

        return residuals


class AnnulusAreas(EquationBase):
    def residual(
        self,
        a0: n0.geo.Area.Hint,
        rr0: n0.geo.RDistr.Hint,
        hh0: n0.geo.HDistr.Hint,
    ):
        return a0 - 2 * np.pi * rr0 * hh0


class MeridionalRatios(EquationBase):
    def residual(
        self,
        hgt0: n0.geo.Height.Hint,
        hgt1: n1.geo.Height.Hint,
        h_ratio1: n1.geo.HeightRatio.Hint,
        fl_angle1: n1.geo.FlareAngle.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        rr_mid0: n0.geo.Rmid.Hint,
        rr_mid1: n1.geo.Rmid.Hint,
        rad_ratio1: n1.geo.RadiusRatio.Hint,
        asp_ratio1: n1.geo.AspectRatio.Hint,
    ):
        midspan = get_midspan_idx(chord_ax1)

        r1 = h_ratio1 - hgt1 / hgt0
        r2 = np.tan(fl_angle1) * 2 * chord_ax1[midspan] - (hgt1 - hgt0)
        r3 = rad_ratio1 * rr_mid0 - rr_mid1
        r4 = asp_ratio1 * chord_ax1[midspan] - (hgt0 + hgt1) / 2

        return r1, r2, r3, r4


# WARN: This is a hacky way of achieving what I am trying to do
# it can be done but leads to defining within the equations the
# constant values for the dynamic constraints you want to define


class FlareAngleLimitedAR(EquationBase):
    def __init__(
        self,
        aspect_ratio_targ: float,
        max_flare: float,
        **kwargs,
    ):
        """
        Parameters
        ----------
        aspect_ratio_targ: float
            The target aspect ratio you want to achieve
        max_flare: float,
            The maximum flare angle above which the b.c. switches to it
        """
        super().__init__(**kwargs)
        self._aspect_ratio = aspect_ratio_targ
        self._max_flare = max_flare

    def residual(
        self,
        h0: n0.geo.Height.Hint,
        h1: n1.geo.Height.Hint,
        h_ratio1: n1.geo.HeightRatio.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        rr_mid0: n0.geo.Rmid.Hint,
        rr_mid1: n1.geo.Rmid.Hint,
        rad_ratio1: n1.geo.RadiusRatio.Hint,
        fl_angle1: n1.geo.FlareAngle.Hint,
    ):
        midspan = get_midspan_idx(chord_ax1)

        r1 = h_ratio1 - h1 / h0
        r2 = rad_ratio1 * rr_mid0 - rr_mid1

        # This is a hacky way of imposing dynamic constraints
        ar_tgt = self._aspect_ratio
        flr_max = self._max_flare

        mid_chord_ax = chord_ax1[midspan]
        tan_flare_max = np.tan(flr_max)

        half_delta_height = (h1 - h0) / 2
        ave_height = (h0 + h1) / 2

        chord_AR = ave_height / ar_tgt
        chord_FL_MAX = half_delta_height / tan_flare_max

        # Define the flare angle
        r3 = fl_angle1 - np.arctan(half_delta_height / mid_chord_ax)
        # Set the axial chord
        r4 = mid_chord_ax - safe_max(chord_AR, chord_FL_MAX)

        return r1, r2, r3, r4


class RadialGeometry(EquationBase):
    def residual(
        self,
        chord1: n1.geo.Chord.Hint,
        rr_mid0: n0.geo.Rmid.Hint,
        rr_mid1: n1.geo.Rmid.Hint,
        h0: n0.geo.Height.Hint,
        h1: n1.geo.Height.Hint,
        h_ratio1: n1.geo.HeightRatio.Hint,
        r_ratio1: n1.geo.RadiusRatio.Hint,
        mer_ang0: n0.geo.MeridionalAngle.Hint,
        mer_ang1: n1.geo.MeridionalAngle.Hint,
    ):
        r1 = chord1 - (rr_mid1 - rr_mid0)
        r2 = h0 * h_ratio1 - h1
        r3 = rr_mid0 * r_ratio1 - rr_mid1
        r4 = mer_ang0 - mer_ang1

        return r1, r2, r3, r4


class BladePitch(EquationBase):
    """
    Define pitch as circumference / num_blades and the massflow per blade channel

    Note:
    -----
    I deliberately did not include a mechanism for imposing an integer
    number of blades. It should be done by the loading criteria
    e.g. If the user imposes no loading criteria and just specifies radius and
    pitch, the num of blades might be forced by input not to be an integer.
    Therefore, we choose not to violate the user's constraints for a single
    root problem because it is not compatible with the current architecture.
    """

    def residual(
        self,
        rr0: n0.geo.RDistr.Hint,
        pitch0: n0.geo.Pitch.Hint,
        n_blades0: n0.geo.NumBlades.Hint,
        mf0: n0.oth.MassFlow.Hint,
        ch_mf0: n0.oth.ChMassflow.Hint,
    ):
        r1 = pitch0 * n_blades0 - 2 * np.pi * rr0
        r2 = n_blades0 * ch_mf0 - mf0
        return r1, r2


class BladeRatios(EquationBase):
    def residual(
        self,
        pitch0: n0.geo.Pitch.Hint,
        solid0: n0.geo.Solidity.Hint,
        solid_mid0: n0.geo.SolidityMidspan.Hint,
        chord0: n0.geo.Chord.Hint,
        bld_thick0: n0.geo.BldThick.Hint,
        thick_by_pitch0: n0.geo.ThickByPitch.Hint,
    ):
        midspan = get_midspan_idx(chord0)
        # Ratios
        r1 = pitch0 * solid0 - chord0
        r3 = solid_mid0 * pitch0[midspan] - chord0[midspan]
        r2 = thick_by_pitch0 * pitch0 - bld_thick0
        return r1, r2, r3


class EndwallProperties(EquationBase):
    def residual(
        self,
        h0: n0.geo.Height.Hint,
        rr_mid0: n0.geo.Rmid.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
        ht_ratio0: n0.geo.HubTipRatio.Hint,
        vt0: n0.kin.V_tan.Hint,
        wm0: n0.kin.W_mer.Hint,
        omega0: n0.kin.Omega.Hint,
        a_sound0: n0.stc.SpeedSound.Hint,
        rr_hub0: n0.geo.Rhub.Hint,
        rr_tip0: n0.geo.Rtip.Hint,
        w_hub0: n0.kin.W_hub.Hint,
        w_tip0: n0.kin.W_tip.Hint,
        rel_mach_hub0: n0.kin.RelMach_hub.Hint,
        rel_mach_tip0: n0.kin.RelMach_tip.Hint,
        beta_hub0: n0.kin.Beta_hub.Hint,
        beta_tip0: n0.kin.Beta_tip.Hint,
    ):
        midspan = get_midspan_idx(vt0)

        # Auxiliary variables
        delta_radius = h0 / 2 * np.cos(mer_angle0)
        r_hub = rr_mid0 - delta_radius
        r_tip = rr_mid0 + delta_radius

        U_hub = omega0 * r_hub
        U_tip = omega0 * r_tip

        Wt_hub = vt0[midspan] - U_hub
        Wt_tip = vt0[midspan] - U_tip

        # Residual Equations
        r1 = rr_hub0 - r_hub
        r2 = rr_tip0 - r_tip
        r3 = ht_ratio0 - rr_hub0 / rr_tip0

        r4 = w_hub0 - (wm0[midspan] ** 2 + Wt_hub**2) ** 0.5
        r5 = w_tip0 - (wm0[midspan] ** 2 + Wt_tip**2) ** 0.5

        r6 = beta_hub0 - np.atan2(Wt_hub, wm0[midspan])
        r7 = beta_tip0 - np.atan2(Wt_tip, wm0[midspan])

        # Use closest streamline for speed out sound
        r8 = a_sound0[0] * rel_mach_hub0 - w_hub0
        r9 = a_sound0[-1] * rel_mach_tip0 - w_tip0

        return r1, r2, r3, r4, r5, r6, r7, r8, r9


class LaxByOutradius(EquationBase):
    def residual(
        self,
        rr_mid0: n0.geo.Rmid.Hint,
        chord_ax0: n0.geo.ChordAx.Hint,
        chax_rad_ratio0: n0.ndim.ChAxOutRadRatio.Hint,
    ):
        return rr_mid0 * chax_rad_ratio0 - chord_ax0


class CamberFunction(EquationBase):
    def residual(
        self,
        metal_angle0: n0.geo.MetalAngle.Hint,
        metal_angle1: n1.geo.MetalAngle.Hint,
        camber_coeff1: n1.ndim.CamberCoeff.Hint,
    ):
        delta_angle = metal_angle1 - metal_angle0
        return camber_coeff1 + np.tanh(8 * delta_angle)


class ModifiedZweifel(EquationBase):
    def residual(
        self,
        wt0: n0.kin.W_tan.Hint,
        wt1: n1.kin.W_tan.Hint,
        rho0: n0.stc.Density.Hint,
        rho1: n1.stc.Density.Hint,
        wm0: n0.kin.W_mer.Hint,
        wm1: n1.kin.W_mer.Hint,
        p_rlt0: n0.rlt.Pressure.Hint,
        p1: n1.stc.Pressure.Hint,
        zweif_coeff1: n1.geo.ZweifelCoeff.Hint,
        n_blades1: n1.geo.NumBlades.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        rr_mid1: n1.geo.Rmid.Hint,
    ):
        midspan = get_midspan_idx(wm0)
        delta_Vt = safe_abs(wt1 - wt0)
        solidity_ax = (
            0.5 * (rho0 * wm0 + rho1 * wm1) * delta_Vt / (zweif_coeff1 * (p_rlt0 - p1))
        )

        optimal_pitch = chord_ax1[midspan] / solidity_ax[midspan]
        num_blades_opt = (2 * np.pi * rr_mid1) / optimal_pitch
        return n_blades1 - np.floor(num_blades_opt)


class MinimalCamberLine(CamberLineGeom):
    """
    Minimal camberline. Consider stagger as average
    angle between inlet and outlet and the camberline
    is just equal to the chord.
    This is the most numerically stable geometry, and can
    be used for initiating stiffer solutions
    """

    def residual(
        self,
        chord1: n1.geo.Chord.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        stagger1: n1.geo.Stagger.Hint,
        camb_len1: n1.geo.CamberLength.Hint,
        metal_angle0: n0.geo.MetalAngle.Hint,
        metal_angle1: n1.geo.MetalAngle.Hint,
    ):
        stagger_computed = (metal_angle1 + metal_angle0) / 2

        r1 = camb_len1 - chord1
        r2 = stagger1 - stagger_computed
        r3 = chord1 * np.cos(stagger1) - chord_ax1
        return r1, r2, r3


class TwoSegmentCamberline(CamberLineGeom):
    """
    Camberline made up of two segments aligned with the metal angle that
    meet at half of the axial chord
    """

    def _compute_lines(self, inlet_angle, outlet_angle, chord_ax):
        tan0 = np.tan(inlet_angle)
        tan1 = np.tan(outlet_angle)

        y_end = chord_ax / 2 * (tan0 + tan1)

        len0 = (chord_ax**2 / 4 * (1 + tan0**2)) ** 0.5
        len1 = (chord_ax**2 / 4 * (1 + tan1**2)) ** 0.5

        stagger = np.arctan(y_end / chord_ax)
        camber_len = len0 + len1

        return stagger, camber_len

    def residual(
        self,
        metal_angle0: n0.geo.MetalAngle.Hint,
        metal_angle1: n1.geo.MetalAngle.Hint,
        chord1: n1.geo.Chord.Hint,
        stagger1: n1.geo.Stagger.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        camb_len1: n1.geo.CamberLength.Hint,
    ):
        stagger_computed, camber_len_computed = self._compute_lines(
            metal_angle0, metal_angle1, chord_ax1
        )
        r1 = camb_len1 - camber_len_computed
        r2 = stagger1 - stagger_computed
        r3 = chord1 * np.cos(stagger1) - chord_ax1
        return r1, r2, r3


if __name__ == '__main__':
    inlet_angle = 1.0
    outlet_angle = -1.5
    chord_ax = 0.5

    tan0 = np.tan(inlet_angle)
    tan1 = np.tan(outlet_angle)

    half_chord = chord_ax / 2
    y_end = half_chord * (tan0 + tan1)

    len0 = (half_chord**2 * (1 + tan0**2)) ** 0.5
    len1 = (half_chord**2 * (1 + tan1**2)) ** 0.5

    stagger = np.arctan(y_end / chord_ax)
    camber_len = len0 + len1

    plt.plot([0, chord_ax / 2], [0, chord_ax / 2 * tan0])
    plt.plot([chord_ax / 2, chord_ax], [chord_ax / 2 * tan0, y_end])
    plt.plot(chord_ax, y_end)
    plt.show()


class ParabolicCamberline(CamberLineGeom):
    config = EquationConfig(manual_units=('m', 'm', 'rad'))

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
        metal_angle0: n0.geo.MetalAngle.Hint,
        metal_angle1: n1.geo.MetalAngle.Hint,
        chord1: n1.geo.Chord.Hint,
        stagger1: n1.geo.Stagger.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        camb_len1: n1.geo.CamberLength.Hint,
    ):
        a, b, stagger_computed = self._compute_parabola(
            metal_angle0, metal_angle1, chord_ax1
        )
        arc_len_computed = self._parabolic_arc_len(a, b, chord_ax1)

        r1 = camb_len1 - arc_len_computed
        r2 = chord1 * np.cos(stagger1) - chord_ax1
        r3 = stagger1 - stagger_computed
        return r1, r2, r3

    # === CAMBERLINE PLOTTING FUNCTIONS ===
    def plot_camber_line(
        self,
        axis,
        inlet_angle,
        outlet_angle,
        chord_ax,
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
        a, b, _ = self._compute_parabola(inlet_angle, outlet_angle, chord_ax)
        x = np.linspace(0, chord_ax, n_points)
        y = a * x**2 + b * x

        axis.plot(axial_offset + x, tangential_offset + y, color=color, **kwargs)

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


def test_parabolic_camberline():
    pbc = ParabolicCamberline()
    angle_in = 0
    N_PROFILES = 50
    angles_out = np.linspace(-np.pi / 2.5, np.pi / 2.5, N_PROFILES)
    chord_axial = 0.15

    # 2D Analysis plots
    _, ax = plt.subplots(1, 3, figsize=(12, 4))
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

    _, ax_3d = pbc.plot_3d_blade(
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
