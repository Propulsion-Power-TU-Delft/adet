# Generic and auxiliary models

Prescribed or structural loss models without a physical mechanism, plus auxiliary relations used by other models.

| Model | Machine type |
|---|---|
| [Percentage total pressure loss](#percentage-total-pressure-loss) | Any |
| [Total pressure loss coefficient](#total-pressure-loss-coefficient) | Any |
| [Isentropic link](#isentropic-link) | Any |
| [Percentage entropy loss](#percentage-entropy-loss) | Any |

## Percentage total pressure loss

*Class:* {py:class}`adet.losses.basic.PercTotalPressureLoss`

*Machine type:* Any

*Citation:* none (generic model).

Relative total pressure reduced by a fixed loss coefficient.

$$
p^{r}_{t,1} = p^{r}_{t,0}\,(1-C)
$$

## Total pressure loss coefficient

*Class:* {py:class}`adet.losses.basic.TotalPressureLoss`

*Machine type:* Any

*Citation:* none (generic model).

Loss coefficient defined on the dynamic head at the inlet static pressure.

$$
C = \frac{p^{r}_{t,0}-p^{r}_{t,1}}{p^{r}_{t,0}-p_0}
$$

## Isentropic link

*Class:* {py:class}`adet.losses.basic.IsentropicLink`

*Machine type:* Any

*Citation:* none (structural model).

Entropy is conserved across the component.

$$
s_1 = s_0
$$

## Percentage entropy loss

*Class:* {py:class}`adet.losses.basic.PercentageEntropyLoss`

*Machine type:* Any

*Citation:* none (generic model).

Outlet entropy is a fixed multiple of the inlet entropy, as implemented.

$$
s_1 = (1+C)\,s_0
$$
