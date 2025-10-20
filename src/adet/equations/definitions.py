from adet.equations.base_equation import EquationBase


class AngleDeflection(EquationBase):
    def _compute_residual(self, kin_alpha0, kin_alpha1, oth_deflection1):
        return kin_alpha1 - kin_alpha0 - oth_deflection1
