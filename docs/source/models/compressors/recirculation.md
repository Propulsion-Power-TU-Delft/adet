# Recirculation

Backflow at the impeller outlet at low flow rates.

| Model | Machine type |
|---|---|
| [Recirculation (Coppage)](#recirculation-coppage) | Radial |
| [Recirculation (Oh et al.)](#recirculation-oh-et-al) | Radial |

## Recirculation (Coppage)

*Class:* {py:class}`adet.losses.compressors.centrifugal.RecirculationCoppage`

*Machine type:* Radial

*Citation:* Coppage et al. (1956), WADC TR 55-257, Eq. (2.36); Galvas (1972), Eq. (B76); Oh et al. (1997) Table 5.

Recirculation loss at low flow, proportional to $\sqrt{\tan\alpha_2}$ and the diffusion factor.

$$
\Delta h_{rc} = 0.02\,\sqrt{\tan\alpha_2}\;D_f\,U_2^2
$$

with $D_f$ as in the blade-loading model and $k=0.75$ (0.60 with splitters).

## Recirculation (Oh et al.)

*Class:* {py:class}`adet.losses.compressors.centrifugal.RecirculationOh`

*Machine type:* Radial

*Citation:* Oh, Yoon & Chung (1997), An optimum set of loss models for performance prediction of centrifugal compressors, Table 5 (set E-3) and Table 6.

Recirculation loss with a hyperbolic dependence on the exit flow angle.

$$
\Delta h_{rc} = 8\times10^{-5}\,\sinh\!\left(3.5\,\alpha_2^3\right)\,(D_f\,U_2)^2
$$
