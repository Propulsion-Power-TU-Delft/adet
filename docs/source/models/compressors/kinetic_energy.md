# Kinetic energy dissipation

Dissipation of the residual kinetic energy at the volute exit.

| Model | Machine type |
|---|---|
| [Volute loss (Japikse)](#volute-loss-japikse) | Radial |

## Volute loss (Japikse)

*Class:* {py:class}`adet.losses.stationary.volute.VoluteLossJapikse`

*Machine type:* Radial

*Citation:* Japikse (1996), Centrifugal Compressor Design and Performance; formulas also in Giuffrè (2022).

Total pressure loss of an overhung volute, split into meridional and tangential contributions.

$$
\begin{aligned}
\lambda &= \frac{V_{t,0}}{V_{m,0}}, \qquad AR = \frac{A_{vol}}{A_0} \\
K_m &= \frac{F_1}{1+\lambda^2}, \qquad K_\theta = \begin{cases} F_2\left(\dfrac{r_0}{R_c}\right)^2\dfrac{(\lambda-1/AR)^2}{1+\lambda^2}, & AR\,\lambda>1 \\ 0, & \text{otherwise}\end{cases} \\
p_{t,1} &= p_{t,0} - (K_m+K_\theta)\,(p_{t,0}-p_0)
\end{aligned}
$$
