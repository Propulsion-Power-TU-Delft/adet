from adet.equations import EquationBase


class ThermoVarsAdder(EquationBase):
    """
    This is a `ghost` equation, but don't get scared!
    It essentially forces to add variables
    to the system at runtime without them appearing
    explicitly in any equation, it is currently to force
    the addition of thermodynamic vars to be used in state updates
    """

    def residual(
        self,
        rlt_p0,
        tot_p0,
        stc_p0,
        rlt_T0,
        tot_T0,
        stc_T0,
        rlt_rhomass0,
        tot_rhomass0,
        stc_rhomass0,
        # Cp and Cv
        rlt_cpmass0,
        tot_cpmass0,
        stc_cpmass0,
        rlt_cvmass0,
        tot_cvmass0,
        stc_cvmass0,
        # Visc
        stc_viscosity0,
        # stc_T_critical0,
        # stc_p_critical0,
    ):
        return ()


class GeometricalAdder(EquationBase):
    """
    This is a `ghost` equation, but don't get scared!
    It essentially forces to add variables
    to the system at runtime without them appearing
    explicitly in any equation, it is currently to force
    the addition of thermodynamic vars to be used in state updates
    """

    def residual(
        self,
        geo_height0,
        geo_rr_midspan0,
        geo_meridional_angle0,
    ):
        return ()
