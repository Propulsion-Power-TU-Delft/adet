# Tip clearance and leakage

Flow passing over the blade tip or through the clearance gap, which does no work.

| Model | Machine type |
|---|---|
| [Denton trapezoidal leakage](#denton-trapezoidal-leakage) | Axial |
| [Denton rectangular leakage](#denton-rectangular-leakage) | Axial |
| [Impeller tip leakage (Baines)](#impeller-tip-leakage-baines) | Radial |
| [Tip clearance (Qi et al.)](#tip-clearance-qi-et-al) | Radial |

## Denton trapezoidal leakage

*Class:* {py:class}`adet.losses.turbines.axial.DentonTrapLeakage`

*Machine type:* Axial

*Citation:* Denton (1993), Loss mechanisms in turbomachines, *ASME J. Turbomach.* 115(4), 621–656.

Entropy generation from tip-clearance leakage over a trapezoidal velocity profile.

$$
\Delta s_{leak} = \frac{C_d\,\tau}{\dot m_{ch}\,T_{is}}\int_0^{l}\sqrt{2\rho_{ps}\,(p_{ps}-p_{ss})}\;W_{ss}\,(W_{ss}-W_{ps})\,d\xi
$$

## Denton rectangular leakage

*Class:* {py:class}`adet.losses.turbines.axial.DentonRectLeakage`

*Machine type:* Axial

*Citation:* Denton (1993), *ASME J. Turbomach.* 115(4), 621–656.

Same as the trapezoidal leakage with a rectangular velocity profile, $W_{ss,ps}=\bar W\pm k\,\Delta W$.

$$
\Delta s_{leak} = \frac{C_d\,\tau\, l}{\dot m_{ch}\,T_{is}}\,\sqrt{2\rho_{ps}\,(p_{ps}-p_{ss})}\;W_{ss}\,(W_{ss}-W_{ps})
$$

## Impeller tip leakage (Baines)

*Class:* {py:class}`adet.losses.turbines.centripetal.ImpellerLeakageLoss`

*Machine type:* Radial

*Citation:* Baines (1998), as cited in the source (full reference to be verified).

Tip-leakage loss from axial, radial and mixed axial–radial clearance discharge coefficients $K_x, K_r, K_{xr}$.

$$
\begin{aligned}
C_x &= \frac{1-r_{3s}/r_{2m}}{V_{m,0}\,h_2}, \qquad C_r = \frac{(r_{3s}/r_{2m})\,(c_m-h_2)}{V_{m,1}\,r_{3m}\,h_3} \\
\Delta h_{leak} &= \frac{Z\,U_0^3}{8\pi}\left(K_x\,c_{l,0}\,C_x + K_r\,c_{l,1}\,C_r + K_{xr}\sqrt{c_{l,0}\,C_x\,c_{l,1}\,C_r}\right)
\end{aligned}
$$

## Tip clearance (Qi et al.)

*Class:* {py:class}`adet.losses.turbines.centripetal.QiClearanceLoss`

*Machine type:* Radial

*Citation:* Qi et al. (2017), Eq. (14), response-surface fit to the data of Futral & Holeski, NASA TN D-5513 (1970).

Clearance loss as a polynomial fraction of the ideal enthalpy drop, in terms of axial and radial gap ratios.

$$
\begin{aligned}
A &= \frac{100\,\varepsilon_{ax}}{b_{in}}, \qquad R = \frac{100\,\varepsilon_{rad}}{b_{out}} \\
\Delta h_{tip} &= \Delta h_0\,\frac{0.09678\,A+1.69997\,R-0.096844\,R^2+0.03379\,AR}{100}
\end{aligned}
$$

with $\Delta h_0 = h^{t}_0 - h^{t,is}_1$.
