from adet.equations import EquationBase


class SpeedLinker(EquationBase):
    """
    Impose the same rotational speed (omega) between inlet
    and outlet of a component
    """

    def residual(self, kin_omega0, kin_U1, geo_rr1):
        return kin_U1 - kin_omega0 * geo_rr1


class ComponentLinker(EquationBase):
    """
    Link bewteen outlet of a component and inlet of the next,
    needed to change the relative frame reference
    """

    def residual(
        self,
        # Kine
        kin_Vt0,
        kin_Vm0,
        kin_Vt1,
        kin_Vm1,
        # Thermo
        tot_hmass0,
        tot_hmass1,
        stc_smass0,
        stc_smass1,
        # Geometry
        geo_rmid0,
        geo_rmid1,
        geo_height0,
        geo_height1,
        geo_meridional_angle0,
        geo_meridional_angle1,
        # Others
        oth_massflow0,
        oth_massflow1,
    ):
        # 1. no entropy generation, no work exchange
        r1 = tot_hmass0 - tot_hmass1
        r2 = stc_smass0 - stc_smass1

        # 2. Same ABSOLUTE velocity triangle
        # (the relative changes)
        r3 = kin_Vt0 - kin_Vt1
        r4 = kin_Vm0 - kin_Vm1

        # 3. Same geometry
        # NOTE: => Should i use hh and rr instead?
        #          (Compatible with non-uniform distibutions)

        r5 = geo_rmid0 - geo_rmid1
        r6 = geo_height0 - geo_height1
        r7 = geo_meridional_angle0 - geo_meridional_angle1

        return r1, r2, r3, r4, r5, r6, r7


class VariableAdder(EquationBase):
    """
    This is a `ghost` equation, it just forces the system to
    add variables to the at runtime without them appearing
    explicitly in any equation, it is mainly done to force
    the addition of thermodynamic vars to be used in updates
    """

    def residual(
        self,
        rlt_rhomass0,
        tot_rhomass0,
        stc_rhomass0,
        rlt_p0,
        tot_p0,
        stc_p0,
        rlt_T0,
        tot_T0,
        stc_T0,
    ):
        return ()
