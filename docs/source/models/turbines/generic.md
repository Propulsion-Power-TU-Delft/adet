# Generic and auxiliary models

Prescribed or structural loss models without a physical mechanism, plus auxiliary relations used by other models.

| Model | Machine type |
|---|---|
| [Ideal interspace](#ideal-interspace) | Radial |

## Ideal interspace

*Class:* {py:class}`adet.losses.turbines.centripetal.IdealInterspace`

*Machine type:* Radial

*Citation:* none (ideal limit).

Angular momentum is conserved and no entropy is generated across the interspace.

$$
\begin{aligned}
r_0 V_{t,0} &= r_1 V_{t,1} \\
s_1 &= s_0
\end{aligned}
$$
