import matplotlib.pyplot as plt
import numpy as np

from adet.equations.base_equation import (
    CamberLineGeom,
    EquationBase,
    EquationConfig,
    MeridionalGeom,
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
        area: n0.geo.Area.Hint,
        rr: n0.geo.RDistr.Hint,
        hh: n0.geo.HDistr.Hint,
    ):
        return area - 2 * np.pi * rr * hh


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
        mf0: n0.oth.StreamMassFlow.Hint,
        ch_mf0: n0.oth.ChanMassflow.Hint,
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
        beta0: n0.kin.FlowAngleRel.Hint,
        beta_mid0: n0.kin.Beta_mid.Hint,
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
        r8 = beta_mid0 - beta0[midspan]

        # Use closest streamline for speed out sound
        r9 = a_sound0[0] * rel_mach_hub0 - w_hub0
        r10 = a_sound0[-1] * rel_mach_tip0 - w_tip0

        return r1, r2, r3, r4, r5, r6, r7, r8, r9, r10


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
        Zw: n1.geo.ZweifelCoeff.Hint,
        n_bl_opt: n1.geo.NumBladesOpt.Hint,
        chord_ax1: n1.geo.ChordAx.Hint,
        rr_mid1: n1.geo.Rmid.Hint,
    ):
        midspan = get_midspan_idx(wm0)
        delta_Vt = safe_max(
            safe_abs(wt1 - wt0),
            0.01 * wt0,
        )
        solidity_ax = (rho0 * wm0 + rho1 * wm1) * delta_Vt / (2 * Zw * (p_rlt0 - p1))

        optimal_pitch = chord_ax1[midspan] / solidity_ax[midspan]
        num_blades_opt = (2 * np.pi * rr_mid1) / optimal_pitch
        return n_bl_opt - num_blades_opt


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
