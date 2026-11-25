# Mixing and base pressure

Mixing of non-uniform wakes and jets, sudden expansion, and trailing-edge base pressure.

| Model | Machine type |
|---|---|
| [Mixing (Johnston and Dean)](#mixing-johnston-and-dean) | Radial |

## Mixing (Johnston and Dean)

*Class:* {py:class}`adet.losses.compressors.centrifugal.MixingJohnstonDean`

*Machine type:* Radial

*Citation:* Johnston and Dean (1966), Losses in vaneless diffusers of centrifugal compressors and pumps: analysis, experiment, and design.

Sudden-expansion mixing of the impeller discharge wakes, with only the meridional velocity mixed out. The wake fraction ramps between a minimum and a maximum with mass flow (wake-fraction ramp not from the paper).

$$
\begin{aligned}
\lambda_\mathrm{ramp} &= \frac{\dot m - \lambda_t\,\dot m_{ch}}{(1-\lambda_t)\,\dot m_{ch}}, \qquad \varepsilon = \varepsilon_{min}+(\varepsilon_{max}-\varepsilon_{min})\,\lambda_\mathrm{ramp} \\
k_1 &= \frac{1-\varepsilon-B}{1-\varepsilon}, \qquad k_2 = 1+\tan^2\alpha \\
\Delta h_{mix} &= \frac{k_1^2}{k_2}\,\frac{V^2}{2}
\end{aligned}
$$
