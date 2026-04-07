"""
Simple quantity defintions, for defining differences or ratios rather than
the single quantities
"""

import CoolProp as cp
import numpy as np

from adet.equations.base_equation import EquationBase


class AngleDeflection(EquationBase):
    def residual(self, kin_beta0, kin_beta1, kin_deflection1):
        return kin_deflection1 - (kin_beta1 - kin_beta0)


class IncidenceAngle(EquationBase):
    def residual(self, kin_beta0, kin_inc_angle0, geo_metal_angle0):
        return kin_inc_angle0 - (kin_beta0 - geo_metal_angle0)


class DeviationAngle(EquationBase):
    def residual(
        self,
        kin_beta0,
        kin_beta1,
        kin_dev_angle1,
        geo_metal_angle0,
    ):
        return kin_dev_angle1 + np.sign(geo_metal_angle0) * (kin_beta1 - kin_beta0)


class OptimalIncidence(EquationBase):
    """
    Angle for which there is no change in tangential velocity due to
    blade blockage
    """

    def residual(
        self,
        kin_Wt1,
        kin_Wm0,
        kin_beta_opt0,
    ):

        return kin_beta_opt0 - np.atan2(kin_Wt1, kin_Wm0)


class RepeatedStage(EquationBase):
    """0 - [Stator] - 1 = 2 - [Rotor] - 3"""

    def residual(
        self,
        kin_alpha0,
        kin_alpha3,
        kin_Vm0,
        kin_Vm1,
        kin_Vm2,
        kin_Vm3,
    ):
        r1 = kin_alpha0 - kin_alpha3
        r2 = kin_Vm3 - kin_Vm2
        r3 = kin_Vm1 - kin_Vm0

        return r1, r2, r3


class MeridionalVelocityRatio(EquationBase):
    def residual(self, kin_Vm0, kin_Vm1, kin_VmRatio1):
        return kin_Vm0 * kin_VmRatio1 - kin_Vm1


class MidspanVelocities(EquationBase):
    def residual(
        self,
        kin_V0,
        kin_Vm0,
        kin_Vt0,
        kin_V_midspan0,
        kin_Vm_midspan0,
        kin_Vt_midspan0,
    ):
        num_span = max(kin_V0.shape)
        if num_span == 1:
            midspan = 0
        else:
            midspan = num_span // 2

        r1 = kin_V_midspan0 - kin_V0[midspan]
        r2 = kin_Vm_midspan0 - kin_Vm0[midspan]
        r3 = kin_Vt_midspan0 - kin_Vt0[midspan]

        return r1, r2, r3


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
