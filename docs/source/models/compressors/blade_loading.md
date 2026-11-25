# Blade loading

Diffusion along the blade surfaces, quantified by a diffusion factor.

| Model | Machine type |
|---|---|
| [Blade loading (Coppage)](#blade-loading-coppage) | Radial |

## Blade loading (Coppage)

*Class:* {py:class}`adet.losses.compressors.centrifugal.BladeLoadingCoppage`

*Machine type:* Radial

*Citation:* Coppage et al. (1956), WADC TR 55-257, Eq. (2.33) (diffusion factor), as used in Oh et al. (1997).

Enthalpy loss from the blade loading, through a diffusion factor $D_f$.

$$
\begin{aligned}
D_f &= 1-\frac{W_1}{W_{tip,0}} + \frac{k\,(h_{t,1}-h_{t,0})}{U_1^2}\,\frac{W_1/W_{tip,0}}{\dfrac{Z}{\pi}\left(1-\dfrac{r_{tip,0}}{r_{mid,1}}\right)+\dfrac{2r_{tip,0}}{r_{mid,1}}} \\
\Delta h_{bl} &= 0.05\,(D_f\,U_1)^2
\end{aligned}
$$
