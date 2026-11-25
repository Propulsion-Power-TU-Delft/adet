# Mixing and base pressure

Mixing of non-uniform wakes and jets, sudden expansion, and trailing-edge base pressure.

| Model | Machine type |
|---|---|
| [Denton trailing-edge mixing](#denton-trailing-edge-mixing) | Axial |
| [Denton–Baumgartner trailing-edge mixing](#dentonbaumgartner-trailing-edge-mixing) | Axial |
| [Sieverding base pressure](#sieverding-base-pressure) | Axial, radial |
| [Mixing momentum balances](#mixing-momentum-balances) | Axial, radial |
| [Simplified mixing balances](#simplified-mixing-balances) | Axial, radial |

## Denton trailing-edge mixing

*Class:* {py:class}`adet.losses.turbines.axial.DentonMixingLoss`

*Machine type:* Axial

*Citation:* Denton (1993), *ASME J. Turbomach.* 115(4), 621–656.

Mixing loss from the trailing-edge wake, with an incompressible base-pressure coefficient.

$$
\begin{aligned}
q &= \tfrac{1}{2}\rho_0\min(a_0,W_0)^2, \qquad C_{pb} = \frac{p_b-p_0}{q} \\
\zeta &= -\frac{C_{pb}\,t}{w}+\frac{2\theta}{w}+\left(\frac{\delta^*+t}{w}\right)^2, \qquad w = s\cos\beta_m \\
\Delta s_{mix} &= s\!\left(h^{r}_0,\ p^{r}_0-q\,\zeta\right) - s_0
\end{aligned}
$$

## Denton–Baumgartner trailing-edge mixing

*Class:* {py:class}`adet.losses.turbines.axial.DentonBaumgartnerMixingLoss`

*Machine type:* Axial

*Citation:* Baumgartner et al. (2020) compressible base-pressure correction; full reference to be verified.

Denton mixing loss with a compressible base-pressure coefficient.

$$
\begin{aligned}
C_{pb} &= 2\left(\frac{p_b}{p_0}-1\right)\frac{2}{\gamma M^2}\,\frac{t}{s} \\
q &= \tfrac{1}{2}\gamma p_0 M^2
\end{aligned}
$$

with $\zeta$ and $\Delta s_{mix}$ as in `DentonMixingLoss`.

## Sieverding base pressure

*Class:* {py:class}`adet.losses.turbines.common.SieverdingBasePressure`

*Machine type:* Axial, radial

*Citation:* Sieverding et al. (1980), base pressure correlation built on 16 blade profiles (to be verified against the original).

Trailing-edge base pressure read from a tabulated correlation in $p_1/p_{t,0}$.

$$
\frac{p_b}{p_{t,0}} = f_{\mathrm{Sieverding}}\!\left(\frac{p_1}{p_{t,0}},\ 2\right)
$$

## Mixing momentum balances

*Class:* {py:class}`adet.losses.turbines.common.MixingMomentumBalances`

*Machine type:* Axial, radial

*Citation:* Denton (1993) mixing model, with the momentum balances written per unit span.

Streamwise and cross-stream momentum balances across the mixing region, with the mixed-out pressure from the mixing coefficient $\zeta$.

$$
\begin{aligned}
\dot m'\,W_0 + p_0(s\cos\beta_m-t)+p_b\,t-\rho_0W_0^2\theta &= \dot m'\,W_1\cos\delta+p_1\,s\cos\beta_m \\
p_{suct}\,s|\sin\beta_0| &= p_1\,s|\sin\beta_0|+\dot m'\,W_1\sin\delta \\
p^{r}_1 &= p^{r}_0 - q\,\zeta, \qquad \delta = \mathrm{sign}(\beta_m)(\beta_0-\beta_1) \\
\Delta s &= s_1 - s_0
\end{aligned}
$$

## Simplified mixing balances

*Class:* {py:class}`adet.losses.turbines.common.SimplifiedMixingBalances`

*Machine type:* Axial, radial

*Citation:* Denton (1993) mixing model (simplified form).

Reduced mixing closure: zero deviation when subsonic, choking at the throat when $M_1\ge 0.9$.

$$
\begin{aligned}
\text{if } M_1<0.9:&\quad \beta_1=\beta_0 \\
\text{if } M_1\ge 0.9:&\quad M_0^{rel}=1 \\
p^{r}_1 &= p^{r}_0 - q\,\zeta \\
\Delta s &= s_1 - s_0
\end{aligned}
$$
