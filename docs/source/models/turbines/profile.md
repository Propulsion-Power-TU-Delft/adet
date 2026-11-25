# Profile losses

Boundary-layer growth and wake losses on the blade surfaces, away from the endwalls.

| Model | Machine type |
|---|---|
| [Rectangular profile incidence](#rectangular-profile-incidence) | Axial |
| [Denton trapezoidal profile loss](#denton-trapezoidal-profile-loss) | Axial |
| [Denton rectangular profile loss](#denton-rectangular-profile-loss) | Axial |
| [Stator profile loss (Glassman)](#stator-profile-loss-glassman) | Radial |

## Rectangular profile incidence

*Class:* {py:class}`adet.losses.turbines.axial.DentonRectProfInc`

*Machine type:* Axial

*Citation:* Greitzer, *Internal Flows*, Sec. 5.4.3; Denton (1993), Sec. 7.1.

Total pressure loss coefficient of a rectangular-profile cascade.

$$
p^t_0 - p^t_1 = \zeta\cdot\tfrac{1}{2}\rho_1 W_1^2, \qquad \zeta = 2C_d\frac{l}{s}\left(2\left(\frac{\bar W}{W_0}\right)^3+6\,\frac{\bar W}{W_1}\left(\frac{\Delta W}{W_1}\right)^2\right)
$$

## Denton trapezoidal profile loss

*Class:* {py:class}`adet.losses.turbines.axial.DentonTrapProfile`

*Machine type:* Axial

*Citation:* Denton (1993), *ASME J. Turbomach.* 115(4), 621–656.

Profile loss from a tangential momentum balance and a viscous entropy integral over a trapezoidal velocity profile.

$$
\begin{aligned}
\dot m\,|V_{t,1}-V_{t,0}| &= h\cos\xi_s\int_0^{l}\left(p_{ps}-p_{ss}\right)d\xi \\
\Delta s_{prof} &= \frac{h\,C_d}{\dot m}\int_0^{l}\left(\frac{\rho_{ps}W_{ps}^3}{T_{ps}}+\frac{\rho_{ss}W_{ss}^3}{T_{ss}}\right)d\xi
\end{aligned}
$$

## Denton rectangular profile loss

*Class:* {py:class}`adet.losses.turbines.axial.DentonRectProfile`

*Machine type:* Axial

*Citation:* Denton (1993), *ASME J. Turbomach.* 115(4), 621–656.

Same balances as the trapezoidal profile with a rectangular velocity distribution.

$$
\begin{aligned}
\dot m\,|V_{t,1}-V_{t,0}| &= h\cos\xi_s\,(p_{ps}-p_{ss})\,l \\
\Delta s_{prof} &= \frac{h\,l\,C_d}{\dot m}\left(\frac{\rho_{ps}W_{ps}^3}{T_{ps}}+\frac{\rho_{ss}W_{ss}^3}{T_{ss}}\right)
\end{aligned}
$$

## Stator profile loss (Glassman)

*Class:* {py:class}`adet.losses.turbines.centripetal.StatorProfileLoss`

*Machine type:* Radial

*Citation:* Glassman (1976), Computer program for design analysis of radial-inflow turbines, NASA TN D-8164.

Profile entropy rise from a loss coefficient built on the Prust–Herman functions $E(Q)$ and $H(Q)$.

$$
\begin{aligned}
Q &= \frac{\gamma_{pv}-1}{\gamma_{pv}+1}\left(\frac{W_1}{W_{cr}}\right)^2 \\
\zeta &= \frac{E\,\theta_s}{\cos\alpha_1 - t_s - H\,\theta_s}, \qquad t_s=\frac{t}{s},\ \theta_s=\frac{\theta}{s} \\
\Delta s_{prof} &= s\!\left(h_1^{lss},\,p_1\right)-s_0, \qquad h_1^{lss} = h^{r}_0 - \tfrac{1}{2}W_{1,lss}^2+\tfrac{1}{2}U_1^2, \qquad W_{1,lss} = W_{1,is}\sqrt{1-\zeta}
\end{aligned}
$$
