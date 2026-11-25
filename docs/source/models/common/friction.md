# Wall and disk friction

Skin friction on passage and diffuser walls, disk-face windage, and interspace friction.

| Model | Machine type |
|---|---|
| [Disk friction (Daily and Nece)](#disk-friction-daily-and-nece) | Any |

## Disk friction (Daily and Nece)

*Class:* {py:class}`adet.losses.shared.DiskFricDailyNece`

*Machine type:* Any

*Citation:* Daily & Nece (1960), Chamber dimension effects on induced flow and frictional resistance of enclosed rotating disks, *ASME J. Basic Eng.* 82, 217–230.

Enthalpy loss from rotor-disk friction in an enclosed cavity.

$$
\begin{aligned}
\Delta h_{df} &= \frac{f_{df}\,\bar\rho\,r_1^2\,U_1^3}{4\,\dot m} \\
f_{df} &= \begin{cases} 3.7\,(c/h)^{0.1}\,Re^{-1/2}, & Re\le 3\times10^5 \\ 0.102\,(c/h)^{0.1}\,Re^{-1/5}, & Re>3\times10^5 \end{cases} \\
Re &= \frac{U_1\, r_1\, \rho_1}{\mu_1}
\end{aligned}
$$
