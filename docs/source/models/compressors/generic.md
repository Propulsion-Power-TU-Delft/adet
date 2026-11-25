# Generic and auxiliary models

Prescribed or structural loss models without a physical mechanism, plus auxiliary relations used by other models.

| Model | Machine type |
|---|---|
| [Casey–Rusch inlet relation](#caseyrusch-inlet-relation) | Radial |
| [Shape factor](#shape-factor) | Radial |

## Casey–Rusch inlet relation

*Class:* {py:class}`adet.losses.compressors.centrifugal.CaseyRushInletFunc`

*Machine type:* Radial

*Citation:* not verified; to be added.

Relates the tip relative flow angle to the tip relative Mach number and specific-heat ratio.

$$
\cos\beta_{tip} = \frac{\sqrt{3+\gamma M^2+2M}-\sqrt{3+\gamma M^2-2M}}{2M}
$$

## Shape factor

*Class:* {py:class}`adet.losses.compressors.centrifugal.CompressorShapeFactor`

*Machine type:* Radial

*Citation:* none (geometric definition).

Shape coefficient from the hub-to-tip radius ratio.

$$
k = 1-\left(\frac{r_{hub}}{r_{tip}}\right)^2
$$
