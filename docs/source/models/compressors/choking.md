# Choking

Losses from sonic conditions at the throat or choke section.

| Model | Machine type |
|---|---|
| [Aungier choking](#aungier-choking) | Radial |

## Aungier choking

*Class:* {py:class}`adet.losses.compressors.centrifugal.AungierChoking`

*Machine type:* Radial

*Citation:* Aungier (2000), *Centrifugal Compressors*, ASME Press (model name; to be verified).

Choking loss from the throat-to-choke area ratio.

$$
\begin{aligned}
X &= 11-10\,\frac{A_{thr}}{A_{chk}} \\
\Delta h_{ch} &= W_0^2\,\max\!\left[0,\ \tfrac{1}{2}\left(0.05X+X^7\right)\right]
\end{aligned}
$$
