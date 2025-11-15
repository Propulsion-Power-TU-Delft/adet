"""Mixing losses downstream of turbomachinery blades"""

import numpy as np
from adet.equations.base_equation import EquationBase


class MixingBalances(EquationBase):
    """Balances of mass, momentum and energy for a mixing"""

    def _get_base_pressure(self):
        raise NotImplementedError

    # TODO:
    # 1. Check if choked / sonic loop
    # 2. Add Sieverding base pressures (Check turbosim)

    def residual(
        self,
        # Thermo
        stc_rhomass0,
        stc_rhomass1,
        stc_p0,
        stc_p1,
        tot_hmass0,
        tot_hmass1,
        # Kine
        kin_V0,
        kin_V1,
        kin_alpha0,
        kin_alpha1,
        # Geometry
        geo_throat0,
        geo_pitch0,
        geo_te_thick0,
        # Boundary layer
        oth_disp_thick0,
        oth_mom_thick0,
        oth_massflow0,
    ):
        mass_in = stc_rhomass0 * kin_V0 * (geo_throat0 - oth_disp_thick0)
        mass_out = stc_rhomass1 * kin_V1 * np.cos(kin_alpha1) * geo_pitch0

        p_base = self._get_base_pressure()

        # NOTE: The sign of deviation should not be relevant
        # because the cosine makes it positive anyways
        # Note to self, if numerical problems => Add fabs
        deviation = kin_alpha0 - kin_alpha1

        mom_in = (
            mass_in * kin_V0
            - stc_rhomass0 * kin_V0**2 * oth_disp_thick0
            + stc_p0 * geo_throat0
            + p_base * geo_te_thick0 * kin_alpha0
        )
        mom_out = mass_out * kin_V1 * np.cos(deviation) + stc_p1 * geo_pitch0 * np.cos(
            kin_alpha1
        )

        # Mass conservation
        r1 = mass_in - mass_out
        # Momentum balance
        r2 = mom_in - mom_out
        # Energy balance (isenthalpic)
        r3 = tot_hmass0 - tot_hmass1

        return r1, r2, r3


class RowMixerLink(EquationBase):
    """
    Data passthrough between blade outlet and mixing object, to be used in addition to
    ComponentLinker. It mainly gives access to the geometrical properties
    of the blade row to the mixing object
    """

    def residual(
        self,
        geo_pitch0,
        geo_pitch1,
        geo_throat0,
        geo_throat1,
        geo_te_thick0,
        geo_te_thick1,
    ):
        r1 = geo_throat0 - geo_throat1
        r2 = geo_pitch0 - geo_pitch1
        r3 = geo_te_thick0 - geo_te_thick1

        return r1, r2, r3
