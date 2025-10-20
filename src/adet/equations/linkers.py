from adet.equations import EquationBase


class SpeedLinker(EquationBase):
    """
    Linker between inlet and outlet node of a blade row
    """

    def _compute_residual(self, kin_omega0, kin_U1, kin_rr1):
        return kin_U1 - kin_omega0 * kin_rr1


class ComponentLinker(EquationBase):
    """
    Link bewteen outlet of a component and inlet of the next,
    needed to change the relative frame reference
    """

    def _compute_residual(
        self,
        tot_hmass0,
        tot_hmass1,
        stc_smass0,
        stc_smass1,
        kin_Vt0,
        kin_Vm0,
        kin_Vt1,
        kin_Vm1,
        kin_rmid0,
        kin_rmid1,
        kin_height0,
        kin_height1,
        kin_meridional_angle0,
        kin_meridional_angle1,
    ):
        # 1. no entropy generation, no work exchange
        r1 = tot_hmass0 - tot_hmass1
        r2 = stc_smass0 - stc_smass1

        # 2. Same ABSOLUTE velocity triangle
        r3 = kin_Vt0 - kin_Vt1
        r4 = kin_Vm0 - kin_Vm1

        # 3. Same geometry
        r5 = kin_rmid0 - kin_rmid1
        r6 = kin_height0 - kin_height1
        r7 = kin_meridional_angle0 - kin_meridional_angle1

        return r1, r2, r3, r4, r5, r6, r7
