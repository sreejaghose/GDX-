# EDA Summary — Train Split Only

Source: `data/processed/train.parquet` (2006-05-22 to 2013-11-18, 1,888 trading
days — see `results/split_dates.json`). Validation and holdout data were not
touched. Plots are in `results/eda/`.

## 1. Prices and cumulative returns

- `01_prices.png`, `02_cumulative_returns.png`
- GDXJ has no data before 2009-11-11 (876 rows of `NaN` in this split — it
  hadn't launched yet); its price/return series correctly starts partway
  through the panel.
- **IAU and GLD are near-duplicates**: both track spot gold and their lines
  in `02_cumulative_returns.png` sit exactly on top of each other. Cumulative
  return over the train window: IAU +89.1%, GLD +88.2%.
- Full-period cumulative returns: **GLD +88.2%, IAU +89.1%, TLT +65.9%, SPY
  +65.9%, GDX −33.7%, GDXJ −58.6%**. The two gold-miner ETFs (GDX, GDXJ)
  underperformed spot gold badly over this window despite starting the
  period more or less together with it — miners gave back all their 2010–11
  gains and then some as costs rose and gold pulled back from its 2011 peak.
- SPY shows the 2008–09 crash and recovery; GDX/GDXJ show much larger
  drawdowns and swings around the same period (annualized vol below).

## 2. Correlation matrix of daily returns

`03_correlation_matrix.png`

| | SPY | IAU | GLD | TLT | GDX | GDXJ |
|---|---|---|---|---|---|---|
| SPY | 1.00 | 0.08 | 0.08 | −0.46 | 0.37 | 0.41 |
| IAU | 0.08 | 1.00 | 1.00 | 0.05 | 0.76 | 0.77 |
| GLD | 0.08 | 1.00 | 1.00 | 0.06 | 0.76 | 0.77 |
| TLT | −0.46 | 0.05 | 0.06 | 1.00 | −0.10 | −0.12 |
| GDX | 0.37 | 0.76 | 0.76 | −0.10 | 1.00 | 0.93 |
| GDXJ | 0.41 | 0.77 | 0.77 | −0.12 | 0.93 | 1.00 |

- **IAU vs GLD: 0.995** — effectively redundant instruments (both hold
  physical gold bullion). Including both in a model as independent features
  will cause severe multicollinearity.
- **GDX vs GDXJ: 0.93** — highly related but not redundant; GDXJ (junior
  miners) is the higher-beta, noisier cousin of GDX (senior miners).
- **Gold ETFs vs miner ETFs: ~0.76–0.77** — miners are gold-linked but far
  from a pure play on the metal; roughly 40% of return variance is
  unexplained by gold moves alone (equity-market / cost / leverage factors).
- **SPY vs TLT: −0.46** — the expected stocks/bonds hedge relationship.
- **SPY vs gold (IAU/GLD): ~0.08** — near zero, consistent with gold's
  reputation as an equity diversifier over this period.
- **TLT vs gold miners: ≈ −0.10 to −0.12** — weak negative, i.e. rate-driven
  bond moves are only loosely linked to miner equities.

## 3. Autocorrelation of GDX and GDXJ daily returns (20 lags)

`04_acf_gdx_gdxj.png`

- **GDX** (n=1,887 return obs): lag-1 ACF ≈ −0.01 (not significant). 95% CI
  ≈ ±0.045. Lags 2, 5, and 15 poke just outside the band, but with 20 lags
  tested, 1 spurious "significant" hit is expected by chance alone — there's
  no evidence of a real, exploitable linear pattern.
- **GDXJ** (n=1,011 return obs, starting 2009-11-11): lag-1 ACF ≈ +0.02 (not
  significant). 95% CI ≈ ±0.062. Only lag 18 pokes outside the band, again
  consistent with noise.
- **Conclusion: daily returns for both tickers are close to serially
  uncorrelated** (weak-form efficient at the daily-return level) over the
  train window — no obvious short-horizon momentum or mean-reversion signal
  to exploit from lagged returns alone.

## 4. Rolling 60-day correlation, GDX vs GLD

`05_rolling_corr_gdx_gld.png`

- Full-sample correlation is 0.76, but it is **not stable**: the 60-day
  rolling correlation ranges from **0.46 to 0.92** (mean 0.79, std 0.075).
- Minimum (≈0.46–0.47) occurs around **August 2011**, the sharpest point of
  gold's 2011 rally into its peak — miners lagged the metal badly as the
  rally accelerated (rising costs / equity-market drag decoupling GDX from
  spot gold).
- Correlation dips again, less severely, around **mid-2013** during the
  gold sell-off, and stays consistently above ~0.85 in calmer stretches
  (e.g. late 2010, mid-2012).
- **Implication for modeling**: treating GDX/GLD correlation as a fixed
  0.76 constant would understate regime risk — the relationship weakens by
  roughly 0.3 correlation points during fast, large gold moves, which is
  exactly when a hedge or pairs strategy relying on that correlation would
  need it most.

## Practical takeaways

1. Don't feed both IAU and GLD into a model as separate features without
   addressing collinearity (drop one, or use their spread/ratio instead).
2. GDX and GDXJ are highly correlated (0.93) but not interchangeable.
   Annualized daily-return vol over train is comparable (GDX 44.9%, GDXJ
   44.0%), but GDXJ's sample only starts in Nov 2009 and so excludes the
   2008 crash that's baked into GDX's number — on a matched-date basis
   GDXJ is almost certainly the more volatile of the two, as its "junior
   miner" mandate implies.
3. No meaningful daily-return autocorrelation was found in GDX or GDXJ —
   any strategy built on lagged own-returns needs a real edge beyond what
   the raw ACF shows.
4. GDX/GLD correlation is regime-dependent, weakening sharply during large,
   fast gold moves (August 2011 low of ~0.46) — worth modeling as
   time-varying rather than a fixed constant.
