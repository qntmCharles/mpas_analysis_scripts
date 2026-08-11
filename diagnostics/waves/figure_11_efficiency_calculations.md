# Figure 11 wave-response efficiency calculations

## Filtered variables

Let $P$ denote precipitation rate, $H_c$ denote cloud-class occurrence, and $L$ denote outgoing longwave radiation.

The index $c$ identifies total, active, stratiform, non-precipitating, or TTL-reaching high cloud.

Let $\mathcal{F}_w$ denote the latitude projection and wavenumber-frequency filter for wave band $w$.

The filtered fields are

\[
P_f = \mathcal{F}_w[P], \qquad
H_{f,c} = \mathcal{F}_w[H_c], \qquad
L_f = \mathcal{F}_w[L].
\]

Each field has dimensions time by longitude after latitude projection.

$H_{f,c}$ is continuous and signed because filtering acts on each cloud occurrence field.

## Regression coefficients

Figure 11 uses through-origin regressions over every joint finite time-longitude sample.

The cloud-class efficiency is

\[
\beta_{H,c} =
\frac{\sum_{t,\lambda} P_f(t,\lambda)H_{f,c}(t,\lambda)}
     {\sum_{t,\lambda} P_f(t,\lambda)^2}.
\]

The original total-high-cloud coefficient is $\beta_H=\beta_{H,\mathrm{total}}$.

Its units are cloud fraction per mm day$^{-1}$.

The raw longwave regression coefficient is

\[
\beta_L =
\frac{\sum_{t,\lambda} P_f(t,\lambda)L_f(t,\lambda)}
     {\sum_{t,\lambda} P_f(t,\lambda)^2}.
\]

Convection usually produces negative $L_f$ anomalies.

Figure 11 therefore plots the positive longwave response

\[
E_L = -\beta_L.
\]

Its units are W m$^{-2}$ per mm day$^{-1}$.

The calculation does not fit an intercept or recenter the filtered samples.

The precipitation amplitude is separate from efficiency:

\[
P_{\mathrm{RMS}} =
\left(\frac{1}{N}\sum_{t,\lambda}P_f(t,\lambda)^2\right)^{1/2}.
\]

Direct experiment contrasts use

\[
\Delta\beta = \beta_{\mathrm{experiment}}-\beta_{\mathrm{reference}}.
\]

## Block-bootstrap intervals

Uncertainty uses 2,000 circular moving-block resamples along time.

Each selected time retains every longitude, which preserves within-time spatial dependence.

The block duration for wave band $w$ is

\[
B_w = \sqrt{T_{w,\min}T_{w,\max}},
\]

rounded to the nearest six-hourly time step.

Each resample concatenates circular contiguous blocks until it reaches the original record length.

The calculation recomputes the complete through-origin coefficient for every resample.

The plotted interval uses the 2.5th and 97.5th percentiles.

Experiment contrasts use identical block indices across simulations.

An interval excluding zero indicates pointwise block-resampling consistency.

It does not represent formal field significance or ensemble uncertainty.

The intraseasonal interval needs caution because the record contains few representative-period blocks.

## Scatter figures

Each cloud scatter figure plots $H_{f,c}$ against $P_f$.

Its regression line is $H_{f,c}=\beta_{H,c}P_f$.

The longwave scatter figure plots $L_f$ against $P_f$ with line $L_f=\beta_L P_f=-E_LP_f$.

Each panel displays at most 20,000 sampled points for clarity.

The regression calculation still uses every joint finite sample.

## References

- Wheeler and Kiladis (1999), *Journal of the Atmospheric Sciences*, 56, 374–399.
- Künsch (1989), *Annals of Statistics*, 17, 1217–1241. [doi:10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265)
