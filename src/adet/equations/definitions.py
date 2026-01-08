"""
Simple quantity defintions, for defining differences or ratios rather than
the single quantities
"""

from adet.equations.base_equation import EquationBase
from adet.equations.utils import safe_sum
import numpy as np


class AngleDeflection(EquationBase):
    def residual(self, kin_beta0, kin_beta1, kin_deflection1):
        return kin_beta1 - kin_beta0 - kin_deflection1


class RadiusRatio(EquationBase):
    def residual(self, geo_rmid0, geo_rmid1, geo_radiusRatio1):
        return geo_radiusRatio1 - geo_rmid1 / geo_rmid0


class HeightRatio(EquationBase):
    def residual(self, geo_height0, geo_height1, geo_heightRatio1):
        return geo_heightRatio1 - geo_height1 / geo_height0


class AreaAveragePressure(EquationBase):
    def residual(self, oth_p_AreaAve0, stc_p0, geo_area0):
        return safe_sum(geo_area0) * oth_p_AreaAve0 - safe_sum(geo_area0 * stc_p0)


class RepeatedStage(EquationBase):
    """0 - [Stator] - 1 = 2 - [Rotor] - 3"""

    def residual(self, kin_alpha0, kin_alpha3, kin_Vm0, kin_Vm1, kin_Vm2, kin_Vm3):
        r1 = kin_alpha0 - kin_alpha3
        r2 = kin_Vm3 - kin_Vm2
        r3 = kin_Vm1 - kin_Vm0

        return r1, r2, r3


class DegreeOfReaction(EquationBase):
    """
    0 - [Stator] - 1 === 2 - [Rotor] - 3
    This assumes the stator is on nodes 0,1 and the stator on 2,3 is the rotor.
    The degree of reaction is an `oth` property of node 3
    """

    def residual(
        self,
        stc_hmass0,
        stc_hmass1,
        stc_hmass2,
        stc_hmass3,
        oth_reactDegree3,
    ):
        delta_hmass_rotor = stc_hmass3 - stc_hmass2
        delta_hmass_stage = stc_hmass3 - stc_hmass0

        return delta_hmass_stage * oth_reactDegree3 - delta_hmass_rotor


class MeridionalVelocityRatio(EquationBase):
    def residual(self, kin_Vm0, kin_Vm1, kin_VmRatio1):
        return kin_Vm0 * kin_VmRatio1 - kin_Vm1


class EndWallVelocities(EquationBase):
    """Computation of the velocities at the endwall, also for single span cases"""

    def residual(
        self,
        kin_Wt0,
        kin_Wm0,
        geo_meridional_angle0,
        kin_omega0,
        geo_rr0,
        geo_height0,
        kin_W_hub0,
        kin_W_shroud0,
    ):
        num_span = max(kin_Wt0.shape)
        if num_span == 1:
            midspan = 0
        else:
            midspan = num_span // 2

        deltaW = kin_omega0 * geo_height0 * np.cos(geo_meridional_angle0) / 2
        Wt_hub = kin_Wt0[midspan] - deltaW
        Wt_shroud = kin_Wt0[midspan] + deltaW

        r1 = kin_W_hub0 - np.sqrt(kin_Wm0[midspan] ** 2 + Wt_hub**2)
        r2 = kin_W_shroud0 - np.sqrt(kin_Wm0[midspan] ** 2 + Wt_shroud**2)

        return r1, r2


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
        geo_pitch0,
        geo_num_blades0,
        geo_rr0,
        oth_ch_massflow0,
        oth_massflow0,
    ):
        r1 = geo_pitch0 * geo_num_blades0 - 2 * np.pi * geo_rr0
        r2 = geo_num_blades0 * oth_ch_massflow0 - oth_massflow0

        return r1, r2


class Solidity(EquationBase):
    def residual(self, geo_solidity0, geo_pitch0, geo_chord0):
        return geo_pitch0 * geo_solidity0 - geo_chord0


class ThicknessToPitch(EquationBase):
    def residual(self, geo_bld_thick0, geo_thick_by_pitch0, geo_pitch0):
        return geo_bld_thick0 - geo_thick_by_pitch0 * geo_pitch0


class BoundaryLayerRatios(EquationBase):
    """Boundary layer properties ratios definitions
    based on trailing edge thickness"""

    def residual(
        self,
        # Geometry
        geo_pitch0,
        geo_bld_thick0,
        geo_thick_by_pitch0,
        # Boundary layer
        oth_disp_thick0,
        oth_disp_by_mom_thick0,
        oth_mom_thick0,
        oth_mom_by_bld_thick0,
    ):
        r1 = oth_disp_thick0 - oth_disp_by_mom_thick0 * oth_mom_thick0
        r2 = oth_mom_thick0 - oth_mom_by_bld_thick0 * geo_bld_thick0

        return r1, r2
