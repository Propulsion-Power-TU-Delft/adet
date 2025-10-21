"""
Simple quantity defintions, for defining differences or ratios rather than
the single quantities
"""

from adet.equations.base_equation import EquationBase


class AngleDeflection(EquationBase):
    def residual(self, kin_alpha0, kin_alpha1, oth_deflection1):
        return kin_alpha1 - kin_alpha0 - oth_deflection1


class RadiusRatio(EquationBase):
    def residual(self, kin_rmid0, kin_rmid1, oth_radiusRatio1):
        return oth_radiusRatio1 - kin_rmid1 / kin_rmid0


class HeightRatio(EquationBase):
    def residual(self, kin_height0, kin_height1, oth_heightRatio1):
        return oth_heightRatio1 - kin_height1 / kin_height0


class MeridionalVelocityRatio(EquationBase):
    def residual(self, kin_Vm0, kin_Vm1, oth_VmRatio1):
        return oth_VmRatio1 - kin_Vm1 / kin_Vm0
