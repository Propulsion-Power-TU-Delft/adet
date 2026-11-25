# Slip and deviation

Models that set the flow angle at the blade outlet from the metal angle, slip and shock relations.

| Model | Machine type |
|---|---|
| [Free deviation](#free-deviation) | Any |
| [Zero deviation](#zero-deviation) | Any |
| [Zero midspan deviation](#zero-midspan-deviation) | Any |
| [Backström slip](#backstrom-slip) | Radial |
| [Qiu slip](#qiu-slip) | Radial, mixed |
| [Aungier deviation](#aungier-deviation) | Axial |
| [Oblique shock deviation](#oblique-shock-deviation) | Any |
| [Chen and Baines slip](#chen-and-baines-slip) | Radial, mixed |
| [Stanitz slip](#stanitz-slip) | Radial, mixed |

## Free deviation

*Class:* {py:class}`adet.deviation.FreeDeviation`

*Machine type:* Any

*Citation:* none (structural model).

Flow angle is decoupled from the metal angle; no constraint is imposed.

$$
\beta_0 \ \text{free}
$$

## Zero deviation

*Class:* {py:class}`adet.deviation.ZeroDeviation`

*Machine type:* Any

*Citation:* none (structural model).

Flow angle equals the metal angle at the node.

$$
\beta_0 = \beta_{m,0}
$$

## Zero midspan deviation

*Class:* {py:class}`adet.deviation.ZeroMidspanDeviation`

*Machine type:* Any

*Citation:* none (structural model).

Zero deviation imposed only at the midspan station.

$$
\beta_{0,\mathrm{mid}} = \beta_{m,0,\mathrm{mid}}
$$

(backstrom-slip)=
## Backström slip

*Class:* {py:class}`adet.deviation.BackstromSlip`

*Machine type:* Radial

*Citation:* Backström (2006), A unified correlation for slip factor in centrifugal impellers, *ASME J. Turbomach.* 128, 1–10 (taken from secondary references; to be verified against the original).

Geometry-based slip factor with effective solidity and a tangential velocity closure.

$$
\begin{aligned}
\sigma_{\mathrm{eff},1} &= \frac{Z_{\mathrm{tot}}}{2\pi\cos\beta_{m,1}}\left(1-\frac{r_0}{r_1}\right) \\
\mu_1 &= 1-\frac{1}{1+k_1\,\sigma_{\mathrm{eff},1}\sqrt{\cos\beta_{m,1}}} \\
W_{t,1} &= W_{m,1}\tan\beta_{m,1} - (1-\mu_1)\,U_1
\end{aligned}
$$

## Qiu slip

*Class:* {py:class}`adet.deviation.QiuSlip`

*Machine type:* Radial, mixed

*Citation:* Qiu et al. (2011), Analysis and Validation of a Unified Slip Factor Model for Impellers at Design and Off-Design Conditions (Eq. 6, 8, 10a; blade-turning and width terms neglected).

Unified slip model for radial and mixed-flow impeller outlets, with the meridional inclination entering the shape factor.

$$
\begin{aligned}
F &= 1-2\sin\frac{\pi}{Z}\sin\!\left(\frac{\pi}{Z}+\beta_{m}\right)\cos\beta_{m}\sin\phi-\frac{t}{s_2\cos\beta_{m}}, \qquad s_2=\frac{2\pi r_2}{Z} \\
\mu &= 1-F\,\frac{\pi\cos\beta_{m}\sin\phi}{Z} \\
W_t &= W_m\tan\beta_m-(1-\mu)\,U
\end{aligned}
$$

## Aungier deviation

*Class:* {py:class}`adet.deviation.AungierDeviationModel` (helpers `aungier_delta0` at line 148)

*Machine type:* Axial

*Citation:* Aungier (2003), *Axial-Flow Compressors*, ASME Press (deviation correlation; to be verified).

Incompressible deviation $\delta_0$ from the metal angle, corrected by a subsonic Mach-number polynomial for $M_2>0.5$.

$$
\begin{aligned}
\delta_0 &= |\beta_m| - \arccos\!\left[\cos\beta_m\left(1+(1-\cos\beta_m)\left(\frac{2|\beta_m|}{\pi}\right)^2\right)\right] \\
\delta &= \delta_0\left(1-10X^3+15X^4-6X^5\right), \qquad X = 2M_2-1 \\
\beta &= \beta_m - \delta
\end{aligned}
$$

## Oblique shock deviation

*Class:* {py:class}`adet.deviation.ShockLoss`

*Machine type:* Any

*Citation:* standard oblique-shock relations (e.g. Anderson, *Modern Compressible Flow*; to be verified). The shock angle uses a weak-shock approximation.

Oblique shock across a deflection $\delta$, with the entropy rise and the outlet flow angle tied to the metal angle.

$$
\begin{aligned}
\rho_1 u_1 &= \rho_0 u_0, \qquad \rho_1 u_1 w_1 = \rho_0 u_0 w_0, \qquad p_1+\rho_1u_1^2 = p_0+\rho_0u_0^2 \\
h_1+\tfrac{1}{2}u_1^2 &= h_0+\tfrac{1}{2}u_0^2 \\
\sigma &= \arcsin\frac{1}{M_0} + \frac{\gamma+1}{4}\,\frac{M_0^2}{M_0^2-1}\,\delta \\
\Delta s &= s_1 - s_0 \\
\beta_1 &= \beta_m - \delta
\end{aligned}
$$

## Chen and Baines slip

*Class:* {py:class}`adet.deviation.ChenSlip`

*Machine type:* Radial, mixed

*Citation:* Chen & Baines (1994), The aerodynamic loading of radial and mixed-flow turbines, *Int. J. Mech. Sci.* 36(1), 63–79, Eq. (8).

Rotor-inlet slip for radial and mixed-flow turbines (eddy contained within the passage).

$$
\begin{aligned}
\mu &= 1-\frac{2\sin(\pi/Z)}{\pi\left(1+\sin(\pi/Z)\right)}\,|\sin\phi|\cos\beta_m \\
W_m\tan\beta &= W_m\tan\beta_m-(1-\mu)\,U
\end{aligned}
$$

## Stanitz slip

*Class:* {py:class}`adet.deviation.StanitzSlip`

*Machine type:* Radial, mixed

*Citation:* Stanitz correlation, Eq. (4) in Chen & Baines (1994), *Int. J. Mech. Sci.* 36(1), 63–79.

Rotor-inlet slip for radial and mixed-flow turbines (original centrifugal-compressor correlation).

$$
\begin{aligned}
\mu &= 1-0.63\,\frac{\pi}{Z}\,|\sin\phi|\cos\beta_m \\
W_m\tan\beta &= W_m\tan\beta_m-(1-\mu)\,U
\end{aligned}
$$
