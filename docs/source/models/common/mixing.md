# Mixing and base pressure

Mixing of non-uniform wakes and jets, sudden expansion, and trailing-edge base pressure.

| Model | Machine type |
|---|---|
| [Abrupt expansion](#abrupt-expansion) | Any |

## Abrupt expansion

*Class:* {py:class}`adet.losses.shared.AbruptExpansion`

*Machine type:* Any

*Citation:* Benedict et al.; Aungier (2006), *Turbine Aerodynamics*, ASME Press, p. 81.

Entropy rise from an abrupt passage enlargement, with the pressure drop from the expansion.

$$
\begin{aligned}
g &= s\cos\beta_m \\
\Delta p_t &= \tfrac{1}{2}\rho W^2\left(\frac{g}{g-t-\delta^*}-1\right)^2 \\
\Delta s &= s_0 - s\!\left(h^{r}_0,\ p^{r}_0-\Delta p_t\right)
\end{aligned}
$$
