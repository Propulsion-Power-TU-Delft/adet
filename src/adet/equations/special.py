"""
These are a `ghost` equation, but don't get scared!
They essentially force the system to add the variables
that are declared in their signature to the system without them appearing
explicitly in any equation, it is currently REQUIRED to force
the addition of thermodynamic vars to be used in state updates.
Otherwise you might declare you want to update with (p,T) but they
do not even appear in the system, leading to an exception
"""

from adet.equations import EquationBase
from adet.equations.variables import NodeVariables

n0 = NodeVariables(0)


class ThermoVarsAdder(EquationBase):
    def residual(
        self,
        p_rlt0: n0.rlt.Pressure.Hint,
        p_tot0: n0.tot.Pressure.Hint,
        p0: n0.stc.Pressure.Hint,
        T_rlt0: n0.rlt.Temperature.Hint,
        T_tot0: n0.tot.Temperature.Hint,
        T0: n0.stc.Temperature.Hint,
        rho_rlt0: n0.rlt.Density.Hint,
        rho_tot0: n0.tot.Density.Hint,
        rho0: n0.stc.Density.Hint,
        cp_rlt0: n0.rlt.Cp.Hint,
        cp_tot0: n0.tot.Cp.Hint,
        cp0: n0.stc.Cp.Hint,
        cv_rlt0: n0.rlt.Cv.Hint,
        cv_tot0: n0.tot.Cv.Hint,
        cv0: n0.stc.Cv.Hint,
    ):
        return ()


class GeometricalAdder(EquationBase):
    def residual(
        self,
        h0: n0.geo.Height.Hint,
        rr_mid0: n0.geo.Rmid.Hint,
        mer_angle0: n0.geo.MeridionalAngle.Hint,
    ):
        return ()
