# Wall and disk friction

Skin friction on passage and diffuser walls, disk-face windage, and interspace friction.

| Model | Machine type |
|---|---|
| [Skin friction (Jansen)](#skin-friction-jansen) | Radial |
| [Amirante vaneless diffuser momentum](#amirante-vaneless-diffuser-momentum) | Radial, mixed |
| [Stanitz vaneless diffuser momentum](#stanitz-vaneless-diffuser-momentum) | Radial, mixed |
| [Diffuser friction coefficient](#diffuser-friction-coefficient) | Radial, mixed |
| [Jansen vaneless diffuser loss](#jansen-vaneless-diffuser-loss) | Radial, mixed |
| [Exit cone friction loss](#exit-cone-friction-loss) | Radial, mixed |

## Skin friction (Jansen)

*Class:* {py:class}`adet.losses.compressors.centrifugal.SkinFrictionJansen`

*Machine type:* Radial

*Citation:* Jansen (1967) (model name; to be verified).

Enthalpy loss from wall friction in the impeller passage, using the Colebrook smooth and rough blend.

$$
\begin{aligned}
Re &= \frac{\dot m}{\bar\mu\,D_h}, \qquad \bar w = \frac{V_0+V_1+W_{tip,0}+2W_{hub,0}+3W_1}{8} \\
\Delta h_{fr} &= 2\,|C_f|\,\frac{L_h}{D_h}\,\bar w^2
\end{aligned}
$$

with $C_f$ blended between smooth and rough Colebrook values as in `ExitConeLoss`.

## Amirante vaneless diffuser momentum

*Class:* {py:class}`adet.losses.stationary.diffuser.AmiranteDiffuserMomentum`

*Machine type:* Radial, mixed

*Citation:* Amirante et al. (2015), *J. Eng. Gas Turbines Power* 137(8) (related one-dimensional vaneless diffuser model; to be verified against the implemented equations).

Angular momentum with a wall-friction term, and a polytropic temperature rise with $\eta_p = 0.85$.

$$
\begin{aligned}
r_0 V_{t,0} &= r_1 V_{t,1}\left(1+\frac{W_f}{\varepsilon_w V_1 V_0}\right) \\
T_1 &= T_0\left(\frac{p_1}{p_0}\right)^{\frac{\gamma-1}{\eta_p\gamma}}
\end{aligned}
$$

## Stanitz vaneless diffuser momentum

*Class:* {py:class}`adet.losses.stationary.diffuser.StanitzDiffuserMomentum`

*Machine type:* Radial, mixed

*Citation:* Dubitsky & Japikse (2008), Vaneless Diffuser Advanced Model, Eqs. (1)–(2), adapted from Stanitz (1952), NACA TN 2610, Eqs. (8)–(9).

Meridional and tangential momentum over one radial step of a one-zone vaneless diffuser.

$$
\begin{aligned}
\frac{p_1-p_0}{\rho_1} &= V_{m,1}(V_{m,0}-V_{m,1}) + \frac{V_{t,1}^2}{r_1}\Delta r - \frac{C_f V_1 V_{m,1}}{b_1\sin\phi_1}\Delta r \\
V_{m,1}(V_{t,1}-V_{t,0}) &= -\Delta r\left(\frac{V_{m,1}V_{t,1}}{r_1} + \frac{C_f V_1 V_{t,1}}{b_1\sin\phi_1}\right)
\end{aligned}
$$

## Diffuser friction coefficient

*Class:* {py:class}`adet.losses.stationary.diffuser.DiffuserFrictionCoefficient`

*Machine type:* Radial, mixed

*Citation:* Dubitsky & Japikse (2008), Eq. (6), attributed to Japikse (1996).

Wall skin-friction coefficient of the vaneless diffuser.

$$
C_f = k\left(\frac{1.8\times10^5}{Re}\right)^{0.2}, \qquad Re = \frac{\dot m}{\mu_0\,D_{h,0}}
$$

## Jansen vaneless diffuser loss

*Class:* {py:class}`adet.losses.stationary.diffuser.JansenDiffuserLoss`

*Machine type:* Radial, mixed

*Citation:* Jansen (1967), vaneless diffuser model (to be verified against the original).

Angular momentum balance with wall friction, and an entropy rise from the friction loss.

$$
\begin{aligned}
\frac{V_{t,0}}{V_{t,1}} &= \frac{r_1}{r_0} + \frac{2\pi C_f\rho_0 V_{t,0}\,r_1(r_1-r_0)}{\dot m} \\
\mathcal{L} &= \frac{C_f\left(1-(r_0/r_1)^{1.5}\right)V_0^2}{1.5\,h_0\cos\alpha_0} \\
s_1 &= s_0 + e^{-\mathcal{L}/R}
\end{aligned}
$$

## Exit cone friction loss

*Class:* {py:class}`adet.losses.stationary.diffuser.ExitConeLoss`

*Machine type:* Radial, mixed

*Citation:* Colebrook (1939) for the friction correlation; smooth and rough limits as implemented.

Total pressure loss from wall friction in the conical diffuser downstream of the volute.

$$
\begin{aligned}
\frac{1}{\sqrt{4C_{f,s}}} &= -2\log_{10}\frac{2.51}{Re\sqrt{4C_{f,s}}}, \qquad \frac{1}{\sqrt{4C_{f,r}}} = -2\log_{10}\frac{\varepsilon}{3.71\,D_h} \\
C_f &= C_{f,s}+\left(C_{f,r}-C_{f,s}\right)\left(1-\frac{60}{Re_e}\right), \qquad Re_e = \frac{(Re-2000)\,\varepsilon}{D_h} \\
p_{t,1} &= p_{t,0} - \frac{2\rho_0 V_0^2\,C_f\,L}{D_h}
\end{aligned}
$$
