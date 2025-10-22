from adet.equations import EquationBase


class SpeedLinker(EquationBase):
    """
    Linker between inlet and outlet node of a blade row
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
    ):
        # 1. no entropy generation, no work exchange
        r1 = tot_hmass0 - tot_hmass1
        r2 = stc_smass0 - stc_smass1

        # 2. Same ABSOLUTE velocity triangle
        r3 = kin_Vt0 - kin_Vt1
        r4 = kin_Vm0 - kin_Vm1

        # 3. Same geometry
        r5 = geo_rmid0 - geo_rmid1
        r6 = geo_height0 - geo_height1
        r7 = geo_meridional_angle0 - geo_meridional_angle1

        return r1, r2, r3, r4, r5, r6, r7
