# Incidence

Mismatch between the flow angle and the blade angle at the leading edge.

| Model | Machine type |
|---|---|
| [Incidence (Van den Braembussche)](#incidence-van-den-braembussche) | Radial |
| [Incidence (Galvas)](#incidence-galvas) | Radial |

## Incidence (Van den Braembussche)

*Class:* {py:class}`adet.losses.compressors.centrifugal.IncidenceVDB`

*Machine type:* Radial

*Citation:* Van den Braembussche (model name; to be verified).

Quadratic-plus-linear incidence loss, with asymmetric coefficient for positive and negative incidence.

$$
\begin{aligned}
\Delta i &= \beta_0-\beta_{opt,0} \\
\delta_i &= 2.5+0.15\,(12.5-0.1\beta_0)\,\frac{(M_{rel}-1.2)^2}{2} \\
\Delta h_{inc} &= \left[0.833\left(\frac{\Delta i}{C_{te}\,\delta_i}\right)^2+0.1667\,\frac{\Delta i}{C_{te}\,\delta_i}\right]W_{tip,0}^2
\end{aligned}
$$

with $C_{te}=2.5$ for $i>i_{opt}$ and $C_{te}=2.0$ otherwise.

## Incidence (Galvas)

*Class:* {py:class}`adet.losses.compressors.centrifugal.IncidenceGalvas`

*Machine type:* Radial

*Citation:* Galvas (1972) (model name; to be verified).

Incidence loss from the lost tangential relative velocity.

$$
\Delta h_{inc} = \frac{k_{inc}}{2}\left(W_0\sin|\beta_{opt,0}-\beta_0|\right)^2
$$
