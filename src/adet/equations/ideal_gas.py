import numpy as np
from adet.equations import EquationBase


def ideal_gas_residual(
    p0,
    T0,
    rhomass0,
    hmass0,
    umass0,
    smass0,
    speed_sound0,
    cpmassid0,
    cvmassid0,
    T_ref0,
    p_ref0,
):
    R = cpmassid0 - cvmassid0
    r1 = p0 - rhomass0 * R * T0
    r2 = hmass0 - cpmassid0 * T0
    r3 = umass0 - cvmassid0 * T0
    r4 = smass0 - cpmassid0 * np.log(T0 / T_ref0) + R * np.log(p0 / p_ref0)
    r5 = speed_sound0 - np.sqrt(cpmassid0 / cvmassid0 * R * T0)
    return r1, r2, r3, r4, r5


class IdealStcEos(EquationBase):
    def residual(
        self,
        stc_p0,
        stc_T0,
        stc_rhomass0,
        stc_hmass0,
        stc_umass0,
        stc_smass0,
        stc_speed_sound0,
        oth_cpmassid0,
        oth_cvmassid0,
        oth_T_ref0,
        oth_p_ref0,
    ):
        r1, r2, r3, r4, r5 = ideal_gas_residual(
            stc_p0,
            stc_T0,
            stc_rhomass0,
            stc_hmass0,
            stc_umass0,
            stc_smass0,
            stc_speed_sound0,
            oth_cpmassid0,
            oth_cvmassid0,
            oth_T_ref0,
            oth_p_ref0,
        )
        return r1, r2, r3, r4, r5


class IdealTotEos(EquationBase):
    def residual(
        self,
        tot_p0,
        tot_T0,
        tot_rhomass0,
        tot_hmass0,
        tot_umass0,
        tot_smass0,
        tot_speed_sound0,
        oth_cpmassid0,
        oth_cvmassid0,
        oth_T_ref0,
        oth_p_ref0,
    ):
        r1, r2, r3, r4, r5 = ideal_gas_residual(
            tot_p0,
            tot_T0,
            tot_rhomass0,
            tot_hmass0,
            tot_umass0,
            tot_smass0,
            tot_speed_sound0,
            oth_cpmassid0,
            oth_cvmassid0,
            oth_T_ref0,
            oth_p_ref0,
        )

        return r1, r2, r3, r4, r5


class IdealRltEos(EquationBase):
    def residual(
        self,
        rlt_p0,
        rlt_T0,
        rlt_rhomass0,
        rlt_hmass0,
        rlt_umass0,
        rlt_smass0,
        rlt_speed_sound0,
        oth_cpmassid0,
        oth_cvmassid0,
        oth_T_ref0,
        oth_p_ref0,
    ):
        r1, r2, r3, r4, r5 = ideal_gas_residual(
            rlt_p0,
            rlt_T0,
            rlt_rhomass0,
            rlt_hmass0,
            rlt_umass0,
            rlt_smass0,
            rlt_speed_sound0,
            oth_cpmassid0,
            oth_cvmassid0,
            oth_T_ref0,
            oth_p_ref0,
        )
        return r1, r2, r3, r4, r5
