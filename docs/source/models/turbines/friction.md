# Wall and disk friction

Skin friction on passage and diffuser walls, disk-face windage, and interspace friction.

| Model | Machine type |
|---|---|
| [Impeller passage (Baines)](#impeller-passage-baines) | Radial |
| [Disk windage (Persky and Sauret)](#disk-windage-persky-and-sauret) | Radial |
| [Interspace friction](#interspace-friction) | Radial |

## Impeller passage (Baines)

*Class:* {py:class}`adet.losses.turbines.centripetal.ImpellerPassageLoss`

*Machine type:* Radial

*Citation:* Baines (1998), meanline prediction method for radial turbine efficiency, as cited in the source (full reference to be verified).

Passage friction and blade-loading loss, scaled by the mean relative kinetic energy.

$$
\begin{aligned}
L_h &= \frac{\pi}{4}\left[(c_{ax}-h_2/2)+(r_{2m}-r_{3s}-h_3/2)\right] \\
\Delta h_{pass} &= k_p\left(\frac{L_h}{D_h}+\frac{0.68\left(1-(r_{3m}/r_{2m})^2\right)\cos\beta_3}{h_3}\,c_{mean}\right)\frac{W_0^2+W_1^2}{2}
\end{aligned}
$$

with $D_h$ the hydraulic diameter of the exit passage, and the loss doubled when $(r_{2m}-r_{3s})/h_3<0.2$.

## Disk windage (Persky and Sauret)

*Class:* {py:class}`adet.losses.turbines.centripetal.SauretWindageLoss`

*Machine type:* Radial

*Citation:* Persky and Sauret (2019), Table 1 I.1 and Eqs. A.7, A.8c; Daily & Nece (1960) coefficients.

Rotor back-face windage loss with laminar and turbulent coefficients switched at $Re=10^5$.

$$
\begin{aligned}
Re &= \frac{\rho U r}{\mu}, \qquad C_m = C\left(\frac{\varepsilon}{r}\right)^{m}Re^{-n} \\
\Delta h_{wind} &= \frac{C_m\,\rho\,U^3\,r^2}{2\,\dot m}
\end{aligned}
$$

with $(C,m,n)=(3.7,\,0.1,\,0.5)$ for $Re<10^5$ and $(0.0102,\,0.2,\,0.2)$ otherwise. Note: the source docstring states $4\dot m$ in the denominator, the code uses $2\dot m$.

## Interspace friction

*Class:* {py:class}`adet.losses.turbines.centripetal.InterspaceLoss`

*Machine type:* Radial

*Citation:* not verified; to be added.

Angular momentum loss and entropy generation from wall friction in the interspace between stator and rotor.

$$
\begin{aligned}
Re &= \frac{\dot m}{\mu\,\bar r}, \qquad C_f = k\left(\frac{1.8\times10^5}{Re}\right)^{0.2} \\
\tau &= \tfrac{1}{2}C_f\bar\rho\bar V^2, \qquad A_w = 2\pi(r_0^2-r_1^2) \\
r_0 V_{t,0} &= r_1 V_{t,1} + \frac{\tau\,(\bar V_t/\bar V)\,A_w\,\bar r}{\dot m} \\
s_1 &= s_0 + \frac{\tau\,\bar V\,A_w}{\dot m\,\bar T}
\end{aligned}
$$
