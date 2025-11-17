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
    needed to change the relative frame reference.

    Note
    ----
    This does NOT model the flow in an annular duct, but only acts
    as an exchange of information between components. If you wanted
    to model the interspace between rows the workflow would have to be
    something like
    Row -> ComponentLinker -> Interspace -> ComponentLinker -> Row
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
        # 1. no entropy generation, no work exchange -> Same thermo state
        r1 = tot_hmass0 - tot_hmass1
        r2 = stc_smass0 - stc_smass1

        # 2. Same ABSOLUTE velocity triangle
        # (the relative changes)
        r3 = kin_Vt0 - kin_Vt1
        r4 = kin_Vm0 - kin_Vm1

        # 3. Same geometry distribution
        r5 = geo_rmid0 - geo_rmid1
        r6 = geo_height0 - geo_height1
        r7 = geo_meridional_angle0 - geo_meridional_angle1

        return r1, r2, r3, r4, r5, r6, r7


class RowMixerLink(EquationBase):
    """
    Data passthrough between blade outlet and mixing object, to be used in addition to
    ComponentLinker. It mainly gives access to the geometrical properties
    of the blade row to the mixing object
    """

    def residual(
        self,
        # Reference frame
        kin_omega0,
        kin_omega1,
        # Geometry
        geo_pitch0,
        geo_pitch1,
        geo_throat0,
        geo_throat1,
        geo_te_thick0,
        geo_te_thick1,
        # Boundary layer
        oth_disp_thick0,
        oth_disp_thick1,
        oth_mom_thick0,
        oth_mom_thick1,
    ):
        r1 = geo_throat0 - geo_throat1
        r2 = geo_te_thick0 - geo_te_thick1
        r3 = geo_te_thick0 - geo_te_thick1
        r4 = oth_mom_thick0 - geo_te_thick1
        r5 = oth_disp_thick0 - geo_te_thick1
        r6 = kin_omega0 - kin_omega1

        return r1, r2, r3, r4, r5, r6


class VariableAdder(EquationBase):
    """
    This is a `ghost` equation, but don't get scared!
    It essentially forces to add variables
    to the system at runtime without them appearing
    explicitly in any equation, it is currently to force
    the addition of thermodynamic vars to be used in state updates
    """

    # TODO: This could be made dynamic

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
