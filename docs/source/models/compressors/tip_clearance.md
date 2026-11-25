# Tip clearance and leakage

Flow passing over the blade tip or through the clearance gap, which does no work.

| Model | Machine type |
|---|---|
| [Tip clearance (Jansen)](#tip-clearance-jansen) | Radial |
| [Tip clearance (Brasz)](#tip-clearance-brasz) | Radial |
| [Leakage (Aungier)](#leakage-aungier) | Radial |
| [Leakage lost work](#leakage-lost-work) | Radial |

## Tip clearance (Jansen)

*Class:* {py:class}`adet.losses.compressors.centrifugal.ClearanceJansen`

*Machine type:* Radial

*Citation:* Jansen (1967) (model name; to be verified).

Enthalpy loss from tip-clearance leakage flow.

$$
\begin{aligned}
K &= \frac{\left|r_{tip,0}^2-r_{hub,0}^2\right|}{(r_1-r_{tip,0})\left(1+\rho_1/\rho_0\right)} \\
\Delta h_{cl} &= 0.6\,\frac{\bar c}{b_1}\,|V_{t,1}|\sqrt{\frac{4\pi}{b_1 Z}\,K\,|V_{t,1}|\,V_{m,0}}
\end{aligned}
$$

## Tip clearance (Brasz)

*Class:* {py:class}`adet.losses.compressors.centrifugal.ClearanceBrasz`

*Machine type:* Radial

*Citation:* Brasz (model name; full reference to be verified).

Alternative tip-clearance enthalpy loss, with the clearance added to the blade height.

$$
\Delta h_{cl} = 0.6\,\frac{\bar c\,|V_{t,1}|}{b_1+\bar c/2}\sqrt{\frac{4\pi\,|V_{t,1}|\,V_{m,0}\,K}{(b_1+\bar c/2)\,Z}}
$$

## Leakage (Aungier)

*Class:* {py:class}`adet.losses.compressors.centrifugal.LeakageAungier`

*Machine type:* Radial

*Citation:* Aungier (2000), *Centrifugal Compressors: A Strategy for Aerodynamic Design and Analysis*, ASME Press.

Enthalpy loss from the tip-clearance leakage mass flow, with a leakage velocity from the blade loading.

$$
\begin{aligned}
\Delta p_{cl} &= \frac{\dot m\,(r_1V_{t,1}-r_{tip,0}V_{t,0})}{Z\,\bar r\,\bar H\,L} \\
U_{cl} &= 0.816\sqrt{\frac{2\Delta p_{cl}}{\rho_1}} \\
\dot m_{cl} &= \rho_1 Z\,c\,L\,U_{cl} \\
\Delta h_{leak} &= \frac{\dot m_{cl}\,U_{cl}\,U_1}{2\,\dot m}
\end{aligned}
$$

## Leakage lost work

*Class:* {py:class}`adet.losses.compressors.centrifugal.LeakageLostWork`

*Machine type:* Radial

*Citation:* not verified; to be added.

Fraction of the Euler work lost to clearance leakage.

$$
\Delta h_{lost} = \kappa_{lw}\,(h_{t,1}-h_{t,0})\,\frac{\bar c}{b_1}
$$
