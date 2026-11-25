# Endwall and secondary flows

Losses from endwall boundary layers and secondary-flow vortices.

| Model | Machine type |
|---|---|
| [Secondary losses (Benner et al.)](#secondary-losses-benner-et-al) | Axial |
| [Endwall loss (Denton-based)](#endwall-loss-denton-based) | Radial |

## Secondary losses (Benner et al.)

*Class:* {py:class}`adet.losses.turbines.axial.SecondaryBSM`

*Machine type:* Axial

*Citation:* Benner, Sjolander & Moustapha (2006), An empirical prediction method for secondary losses in turbines, Part II, *ASME J. Turbomach.* 128(2), 281–291.

Secondary-flow entropy rise from an effective outlet pressure weighted by a loss coefficient $Y$ (two correlations, switched at $h/c=2$, and clipped to $Y\ge 0$).

$$
\begin{aligned}
p^{r}_{out} &= \frac{p^{r}_0 + Y\,p_1}{1+Y} \\
\Delta s_{sec} &= s\!\left(h^{r}_0,\ p^{r}_{out}\right)-s_0
\end{aligned}
$$

with $Y$ given by the Benner correlation in terms of $\delta^*_{ew}/h$, $h/c$, $\cos\beta_0/\cos\beta_1$ and $\cos\xi$.

## Endwall loss (Denton-based)

*Class:* {py:class}`adet.losses.turbines.centripetal.EndwallLoss`

*Machine type:* Radial

*Citation:* Denton (1993), dissipation-coefficient closure for endwall boundary layers (as referenced in the source as Ref 7).

Endwall entropy generation integrated over five radial stations along hub and shroud walls.

$$
\Delta s_{ew,i} = \Delta s_{ew,i-1} + \frac{C_d\,\bar\rho\,\bar W^3}{\bar T}\,\frac{A_{w,i}}{\dot m}, \qquad A_{w,i} = A_{hub,i}+A_{tip,i}, \qquad A_{w,i}=2\pi\bar r_i\,\Delta m_i
$$
