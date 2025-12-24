from adet.equations import EquationBase


# TODO: This linking method is becoming a bit verbose and error prone,
# I would like to just add links bewteen variables or, even
# better, to make it so that the same variable types at subsequent nodes
# are actually referncing the same symbol when they are linked.
# (Careful about reference frame-specific variables, omega, U, W, etc.
# make it explicit)
# DONE


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
    ):
        return ()
