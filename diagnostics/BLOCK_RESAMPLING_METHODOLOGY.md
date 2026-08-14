# Selecting temporal block lengths

Temporal blocks preserve serial dependence when the workflow estimates uncertainty from model output.

The control simulation determines each block length.

The workflow then applies that length to every simulation within the same uncertainty family.

This approach avoids tuning the resampling method to each simulated response.

## Statistical evidence

The design script constructs daily control influence series for each reported statistic.

Statistic-specific series matter because the optimal length depends on the estimator, not only the source variable.

This principle follows [Bühlmann and Künsch (1999)](https://doi.org/10.1016/S0167-9473(99)00014-6).

For a centred series \(X_t-\bar X\), the script estimates the lag-\(k\) autocorrelation as

\[
\widehat{\rho}(k)=
\frac{\sum_{t=1}^{n-k}(X_t-\bar X)(X_{t+k}-\bar X)}
     {\sum_{t=1}^{n}(X_t-\bar X)^2}.
\]

It reports three descriptive dependence scales.

These are the first e-folding lag, the first nonpositive lag, and the positive integral scale

\[
\widehat{\tau}_{\mathrm{int}}
=1+2\sum_{k=1}^{k_0-1}\widehat{\rho}(k),
\]

where \(k_0\) is the first nonpositive lag.

The script also implements the corrected Politis–White plug-in selector.

The method follows [Politis and White (2004)](https://doi.org/10.1081/ETC-120028836) and its [2009 correction](https://doi.org/10.1080/07474930802459016).

Set \(r=\max(5,\lfloor\log_{10}n\rfloor)\).

The selector finds the first \(r\) consecutive autocorrelations below \(2\sqrt{\log_{10}(n)/n}\) in absolute value.

It sets \(m=2\max(m_0,1)\), where \(m_0\) starts that sequence.

Define the tapered long-run covariance and weighted covariance moment as

\[
\widehat{\sigma}^{2}=\widehat{\gamma}_0
+2\sum_{k=1}^{m}h(k/m)\widehat{\gamma}_k,
\qquad
G=2\sum_{k=1}^{m}h(k/m)k\widehat{\gamma}_k.
\]

Here, \(h\) is the flat-top taper defined by Politis and White.

It equals one for \(|x|\leq1/2\), then decreases as \(2(1-|x|)\) to zero.

The stationary and circular reference lengths are

\[
\widehat b_i=
\left(\frac{2G^2}{D_i}\right)^{1/3}n^{1/3},
\quad
D_{\mathrm{SB}}=2\widehat{\sigma}^{4},
\quad
D_{\mathrm{CB}}=\frac{4}{3}\widehat{\sigma}^{4}.
\]

Contiguous-block resampling follows [Künsch (1989)](https://doi.org/10.1214/aos/1176347265).

[Wilks (1997)](https://doi.org/10.1175/1520-0442(1997)010%3C0065:RHTFAF%3E2.0.CO;2) gives a climate-field application and discusses spatially varying persistence.

## Selection rule

Review every representative series within one uncertainty family.

Choose one rounded family length that covers important dependence scales and retains enough complete blocks.

Use candidate-length plots to inspect block counts, interval widths, and bootstrap standard errors.

Treat the plug-in estimates as guidance because production resamples precomputed, complete block statistics.

The final rounded production lengths are 10 days for circulation and energy-budget calculations.

High-cloud, jet, mean-state distribution, and thermal-wind calculations use 20 days.

Kelvin, mixed Rossby-gravity, and equatorial Rossby wave responses use 10 days.

Easterly-disturbance and eastward-intraseasonal responses use 20 days.

The ER correlation changes sign but crosses the dependence threshold near the other 10-day wave bands.

These wave lengths replace the geometric midpoint of each filtered period range.

The script supplies this evidence but does not edit `analysis_config.py`.
