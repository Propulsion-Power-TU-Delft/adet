"""
Simple quantity defintions, for defining differences or ratios rather than
the single quantities
"""

from adet.equations.utils import safe_sum
from adet.equations.base_equation import EquationBase

import numpy as np
import CoolProp as cp


class AngleDeflection(EquationBase):
    def residual(self, kin_beta0, kin_beta1, kin_deflection1):
        return kin_beta1 - kin_beta0 - kin_deflection1


class GeometricalRatios(EquationBase):
    def residual(
        self,
        geo_height0,
        geo_height1,
        geo_heightRatio1,
        geo_flare_angle1,
        geo_chord_ax1,
        geo_rr_midspan0,
        geo_rr_midspan1,
        geo_radiusRatio1,
        geo_aspRatio1,
    ):
        r1 = geo_heightRatio1 - geo_height1 / geo_height0
        r2 = np.tan(geo_flare_angle1) - (geo_height1 - geo_height0) / (
            2 * geo_chord_ax1
        )
        r3 = geo_rr_midspan0 * geo_radiusRatio1 - geo_rr_midspan1
        r4 = geo_chord_ax1 * geo_aspRatio1 - geo_height0
        return r1, r2, r3, r4


class AreaAveragePressure(EquationBase):
    def residual(self, oth_p_AreaAve0, stc_p0, geo_area0):
        return safe_sum(geo_area0) * oth_p_AreaAve0 - safe_sum(geo_area0 * stc_p0)


# TODO: Vm constant or height ratio
class RepeatedStage(EquationBase):
    """0 - [Stator] - 1 = 2 - [Rotor] - 3"""

    def residual(self, kin_alpha0, kin_alpha3, kin_Vm0, kin_Vm1, kin_Vm2, kin_Vm3):
        r1 = kin_alpha0 - kin_alpha3
        r2 = kin_Vm3 - kin_Vm2
        r3 = kin_Vm1 - kin_Vm0

        return r1, r2, r3


class MeridionalVelocityRatio(EquationBase):
    def residual(self, kin_Vm0, kin_Vm1, kin_VmRatio1):
        return kin_Vm0 * kin_VmRatio1 - kin_Vm1


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
        # Geometry
        geo_rr0,
        geo_pitch0,
        geo_num_blades0,
        # Massflow
        oth_massflow0,
        oth_ch_massflow0,
    ):
        r1 = geo_pitch0 * geo_num_blades0 - 2 * np.pi * geo_rr0
        r2 = geo_num_blades0 * oth_ch_massflow0 - oth_massflow0

        return r1, r2


class EffectiveBladeNumber(EquationBase):
    def residual(self, geo_num_blades0, geo_num_splitters0, geo_num_blades_eff0):
        return geo_num_blades_eff0 - (geo_num_blades0 + 0.75 * geo_num_splitters0)


class IsentropicProperties(EquationBase):
    input_pair = cp.PSmass_INPUTS
    output_quantities = ('hmass', 'T')
    manual_units = ('J / kg', 'K', 'J / kg', 'K')

    def residual(
        self,
        # Actual properties
        stc_smass0,
        tot_p1,
        stc_p1,
        # Isentropic properties
        oth_tot_hmass_is1,
        oth_tot_T_is1,
        oth_stc_hmass_is1,
        oth_stc_T_is1,
    ):
        hmass_tot_is, temp_tot_is = self.eos(tot_p1, stc_smass0)
        hmass_is, temp_stat_is = self.eos(stc_p1, stc_smass0)

        r1 = oth_tot_hmass_is1 - hmass_tot_is
        r2 = oth_tot_T_is1 - temp_tot_is
        r3 = oth_stc_hmass_is1 - hmass_is
        r4 = oth_stc_T_is1 - temp_stat_is

        return r1, r2, r3, r4


class Solidity(EquationBase):
    def residual(self, geo_solidity0, geo_pitch0, geo_chord0):
        return geo_pitch0 * geo_solidity0 - geo_chord0


class ThicknessToPitch(EquationBase):
    def residual(self, geo_bld_thick0, geo_thick_by_pitch0, geo_pitch0):
        return geo_bld_thick0 - geo_thick_by_pitch0 * geo_pitch0


class ClearanceByHeight(EquationBase):
    def residual(self, geo_clearance_by_height0, geo_height0, geo_tip_clearance0):
        return geo_clearance_by_height0 * geo_height0 - geo_tip_clearance0


class ReducedThermoQuantities(EquationBase):
    def residual(
        self,
        tot_T0,
        tot_p0,
        oth_tot_T_red0,
        oth_tot_p_red0,
        stc_p_critical0,
        stc_T_critical0,
    ):
        r1 = tot_p0 - oth_tot_p_red0 * stc_p_critical0
        r2 = tot_T0 - oth_tot_T_red0 * stc_T_critical0

        return r1, r2


class BoundaryLayerRatios(EquationBase):
    """Boundary layer properties ratios definitions
    based on trailing edge thickness"""

    def residual(
        self,
        # Geometry
        geo_height0,
        geo_bld_thick0,
        # Boundary layer
        oth_mom_thick0,  # Momentum thickness
        oth_disp_thick0,  # Blade disp. thickness
        oth_disp_thick_ew0,  # Endwall disp. thickness
        # Ratios
        oth_disp_by_mom0,  # disp / mom
        oth_mom_by_bld0,  # mom / blade thick
        oth_disp_by_hgt0,  # ew disp / height
    ):
        r1 = oth_disp_thick0 - oth_disp_by_mom0 * oth_mom_thick0
        r2 = oth_mom_thick0 - oth_mom_by_bld0 * geo_bld_thick0
        r3 = oth_disp_thick_ew0 - oth_disp_by_hgt0 * geo_height0

        return r1, r2, r3
