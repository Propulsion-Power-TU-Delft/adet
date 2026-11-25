"""
Stationary component loss models: volute and volute exit cone.
"""

import numpy as np

from adet.equations.base_equation import (
    EquationBase,
)
from adet.equations.utils import safe_if_else
from adet.variables import NodeVariables

n0 = NodeVariables(0)
n1 = NodeVariables(1)


class VoluteLossJapikse(EquationBase):
    """
    Japikse (1996) - Centrifugal compressor design and performance.
    Model of the total pressure loss of an overhung volute downstream
    of a vaneless diffuser.

    The outlet flow is aligned with the pipe axis (V_t = 0, V = V_m).
    The outlet section is given as an equivalent annulus with the boundary
    conditions n1.geo.Rmid = centerline radius R_c and
    n1.geo.Height = A_vol / (2 pi R_c), plus n1.oth.VoluteF1 and VoluteF2.

    - swirl parameter lambda = V_t0 / V_m0
    - area ratio AR = A_vol / A_0, with A_0 = 2 pi r_0 b_0
    - K_m = F1 / (1 + lambda^2), loss of the meridional dynamic head
    - K_theta = F2 (r_0 / R_c)^2 (lambda - 1 / AR)^2 / (1 + lambda^2) if
      AR lambda > 1, else 0 (tangential velocity recovered)
    - Delta p_t = (K_m + K_theta) (p_t0 - p_0)

    - F1: Meridional-velocity loss correction factor [0.6, 1].
           Values below 1 account for partial recovery of the inlet
           meridional velocity head in the downstream exit cone.
    - F2: Tangential-velocity loss correction factor [0.5, 1].
           Accounts for reduced tangential-velocity losses in external
           volutes with a mean radius much larger than the inlet radius.

    Note: formulas can be also found in Giuffre' (2022) - The Effect of
    Size and Working Fluid on the Multi-Objective Design of High-Speed
    Centrifugal Compressors.
    """

    def residual(
        self,
        f1_1: n1.oth.VoluteF1.Hint,
        f2_1: n1.oth.VoluteF2.Hint,
        vt0: n0.kin.V_tan.Hint,
        vm0: n0.kin.V_mer.Hint,
        rr_mid0: n0.geo.Rmid.Hint,
        rr_mid1: n1.geo.Rmid.Hint,
        cum_area0: n0.geo.CumArea.Hint,
        cum_area1: n1.geo.CumArea.Hint,
        pt0: n0.tot.Pressure.Hint,
        p0: n0.stc.Pressure.Hint,
        pt1: n1.tot.Pressure.Hint,
    ):
        swirl = vt0 / vm0
        area_ratio = cum_area1 / cum_area0

        k_m = f1_1 / (1 + swirl**2)
        k_theta = (
            f2_1
            * (rr_mid0 / rr_mid1) ** 2
            * (swirl - 1 / area_ratio) ** 2
            / (1 + swirl**2)
        )
        k_theta = safe_if_else(area_ratio * swirl > 1, k_theta, 0.0)

        return pt1 - (pt0 - (k_m + k_theta) * (pt0 - p0))
