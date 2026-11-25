# Incidence

Mismatch between the flow angle and the blade angle at the leading edge.

| Model | Machine type |
|---|---|
| [Impeller incidence (Persky and Sauret)](#impeller-incidence-persky-and-sauret) | Radial |

## Impeller incidence (Persky and Sauret)

*Class:* {py:class}`adet.losses.turbines.centripetal.ImpellerIncidenceLoss`

*Machine type:* Radial

*Citation:* Persky and Sauret (2019), Table 1 II.1 and Eq. A.2b; Wasserbauer & Glassman (1975), NASA TN D-8063.

Incidence loss with an exponent $n$ that depends on the sign of the incidence.

$$
\begin{aligned}
\beta_{opt} &= \arctan\!\left(\frac{-1.98\,\tan\alpha_0}{Z\,(1-1.98/Z)}\right) \\
\Delta h_{inc} &= \tfrac{1}{2}W_0^2\,\left|\sin(\beta_0-\beta_{opt})\right|^{n}, \qquad n=\begin{cases}2, & \beta_0-\beta_{opt}\le 0 \\ 3, & \text{otherwise}\end{cases}
\end{aligned}
$$
