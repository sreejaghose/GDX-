"""
Builds results/GDX_Research_Report.pdf -- a full narrative summary of the
GDX next-day-direction research project, pulling together every stage
(data, EDA, timing spec, naive baseline, features, in-sample model fit,
walk-forward validation, realistic backtest, robustness checks,
statistical significance, and the frozen final config) into one document
with the key tables and charts already produced along the way.

This is a report-generation script, not part of the research pipeline --
it doesn't compute anything new, only presents what results/*.md and
results/*.csv already contain. Run once, ad hoc: python3 scripts/build_report_pdf.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, HRFlowable, KeepTogether, ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

OUT_PATH = Path("results/GDX_Research_Report.pdf")

# Palette (same reference values used throughout this project's plots)
INK_PRIMARY = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
INK_MUTED = colors.HexColor("#898781")
BLUE = colors.HexColor("#2a78d6")
GOOD = colors.HexColor("#0ca30c")
CRITICAL = colors.HexColor("#d03b3b")
GRIDLINE = colors.HexColor("#e1e0d9")
SURFACE_TINT = colors.HexColor("#f9f9f7")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", fontSize=24, leading=29, textColor=INK_PRIMARY,
                           fontName="Helvetica-Bold", spaceAfter=6, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=13, leading=17, textColor=INK_SECONDARY,
                           fontName="Helvetica", spaceAfter=4))
styles.add(ParagraphStyle(name="Meta", fontSize=9.5, leading=13, textColor=INK_MUTED, fontName="Helvetica"))
styles.add(ParagraphStyle(name="H1", fontSize=16, leading=20, textColor=INK_PRIMARY,
                           fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle(name="H2", fontSize=12.5, leading=16, textColor=INK_PRIMARY,
                           fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14.5, textColor=INK_PRIMARY,
                           fontName="Helvetica", spaceAfter=8, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="BodyBold", parent=styles["Body"], fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Caption", fontSize=8.5, leading=11, textColor=INK_MUTED,
                           fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceBefore=4, spaceAfter=14))
styles.add(ParagraphStyle(name="Callout", parent=styles["Body"], backColor=SURFACE_TINT,
                           borderColor=GRIDLINE, borderWidth=0.75, borderPadding=8, spaceAfter=10))
styles.add(ParagraphStyle(name="BulletBody", parent=styles["Body"], spaceAfter=3))

PAGE_WIDTH = letter[0] - 1.4 * inch  # usable width inside margins


def rule():
    return HRFlowable(width="100%", thickness=0.75, color=GRIDLINE, spaceBefore=2, spaceAfter=10)


def make_table(data, col_widths=None, header=True, small=False, highlight_row=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    font_size = 8 if small else 8.7
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK_PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, GRIDLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, SURFACE_TINT]),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), INK_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    if highlight_row is not None:
        style += [("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#dbe9fa"))]
    t.setStyle(TableStyle(style))
    return t


def fig(path, width=PAGE_WIDTH, caption=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    scaled_h = width * h / w
    flow = [Image(path, width=width, height=scaled_h)]
    if caption:
        flow.append(Paragraph(caption, styles["Caption"]))
    return flow


def bullets(items, style="BulletBody"):
    return ListFlowable(
        [ListItem(Paragraph(it, styles[style]), leftIndent=12, bulletColor=INK_SECONDARY) for it in items],
        bulletType="bullet", start="circle", leftIndent=10, spaceBefore=2, spaceAfter=10,
    )


def build():
    doc = SimpleDocTemplate(
        str(OUT_PATH), pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="GDX Reversal Strategy Research Report", author="Research pipeline",
    )
    story = []

    # ---------------- Title page ----------------
    story.append(Spacer(1, 1.6 * inch))
    story.append(Paragraph("GDX Trailing-Return Reversal Strategy", styles["ReportTitle"]))
    story.append(Paragraph("An end-to-end research write-up: data, EDA, model fitting, walk-forward "
                            "validation, realistic-cost backtesting, robustness checks, and statistical "
                            "significance testing on gold-miner ETF data", styles["ReportSubtitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(rule())
    story.append(Paragraph(
        "Universe: SPY, IAU, GLD, TLT, GDX, GDXJ daily close prices, 2006-05-22 to 2021-12-31 "
        "(3,932 trading days). All model development, tuning, and validation in this report use "
        "<b>train + validation only</b> (2006-05-22 to 2018-11-15, 3,146 days). The final 20% of "
        "the data (2018-11-16 to 2021-12-31, 786 days) is a reserved holdout split that has not "
        "been touched anywhere in this report.", styles["Meta"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Bottom line up front: the research process below found a config that is profitable "
        "in-sample after realistic costs, but it does not survive scrutiny -- it fails a "
        "statistical significance test, is fragile to small parameter perturbations, and its "
        "profits are concentrated in two of four sub-periods rather than persistent. It is "
        "frozen (results/FROZEN_CONFIG.json) for methodological discipline, not because it is "
        "a validated trading edge.", styles["Callout"]))
    story.append(PageBreak())

    # ---------------- Executive summary ----------------
    story.append(Paragraph("Executive Summary", styles["H1"]))
    story.append(bullets([
        "<b>Data</b>: six ETFs' daily close prices were parsed from a raw spreadsheet into a clean panel, "
        "then split chronologically into train (48%), validation (32%), and a never-touched holdout (20%).",
        "<b>EDA</b> found IAU and GLD are near-duplicate instruments (0.995 correlation), GDX/GDXJ are highly "
        "but not perfectly correlated (0.93), GDX's own daily returns show essentially no autocorrelation, "
        "and the GDX/GLD relationship is regime-dependent (60-day rolling correlation ranged 0.46-0.92).",
        "A simple, fully-invested, zero-cost <b>naive reversal rule</b> on GDX showed a large paper Sharpe "
        "(0.67) and CAGR (21.3%) on train+validation -- explicitly flagged at the time as almost certainly "
        "not investable once turnover and costs are considered.",
        "A causal <b>feature set</b> (trailing returns, volatility, RSI, cross-asset momentum) was built with "
        "programmatic zero-lookahead verification, and a <b>logistic regression</b> fit on the train split "
        "alone came back weak: AUC 0.543, 0 of 10 coefficients significant at p&lt;0.05.",
        "A <b>walk-forward grid search</b> (16 lookback x threshold combos, quarterly-refit direction, no "
        "costs) found a best Sharpe of 0.379 -- but re-running the top 5 through a <b>realistic-cost "
        "backtest</b> (real commission/slippage, fixed direction, persistent positions) left only 1 of 5 "
        "profitable, with a CAGR of just +0.44%.",
        "<b>Robustness checks</b> on that one surviving config found: only 2 of 9 nearby parameter "
        "combinations were profitable, only 2 of 4 equal sub-periods were profitable, and doubling costs "
        "flipped CAGR to -13.8%.",
        "A <b>t-test</b> found the mean daily strategy return is not statistically distinguishable from zero "
        "(p=0.578), and a <b>bootstrap</b> of trade-level Sharpe ratios produced a 5th-95th percentile "
        "interval of [-0.33, 0.64] -- straddling zero.",
        "The final parameters were <b>frozen</b> (results/FROZEN_CONFIG.json) with these limitations recorded "
        "verbatim, so a holdout evaluation -- if one is ever run -- uses an untouched, honestly-labeled config.",
    ]))
    story.append(PageBreak())

    # ---------------- 1. Data & splits ----------------
    story.append(Paragraph("1. Data and Chronological Splits", styles["H1"]))
    story.append(Paragraph(
        "The source file (<font face='Courier'>data/raw/gdx_data.xlsx</font>) contained a single sheet "
        "with no assumed structure going in -- the header row, date column, and per-ticker missing-data "
        "patterns were discovered programmatically before anything else was built. GDXJ has no data "
        "before 2009-11-11 (it hadn't launched); every downstream step treats those cells as missing, "
        "not zero.", styles["Body"]))
    story.append(Paragraph(
        "The resulting panel (3,932 rows) was split once, chronologically, by row count -- holdout carved "
        "out first as the most recent 20%, then the remaining 80% split 60/40 into train and validation:",
        styles["Body"]))
    story.append(make_table([
        ["Split", "Date range", "Rows", "Share"],
        ["Train", "2006-05-22 to 2013-11-18", "1,888", "48%"],
        ["Validation", "2013-11-19 to 2018-11-15", "1,258", "32%"],
        ["Holdout (untouched)", "2018-11-16 to 2021-12-31", "786", "20%"],
    ], col_widths=[1.6 * inch, 2.1 * inch, 0.9 * inch, 0.9 * inch]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The holdout file is literally named <font face='Courier'>DO_NOT_TOUCH_holdout.parquet</font> and "
        "has not been loaded, inspected, or fit on anywhere in this project.", styles["Body"]))

    # ---------------- 2. EDA ----------------
    story.append(Paragraph("2. Exploratory Data Analysis (Train Split Only)", styles["H1"]))
    story.append(Paragraph(
        "Daily-return correlation matrix and autocorrelation of GDX/GDXJ returns, computed on the train "
        "split only:", styles["Body"]))
    story.append(KeepTogether(fig("results/eda/03_correlation_matrix.png", width=3.6 * inch,
                                   caption="Daily return correlation matrix (train split).")))
    story.append(bullets([
        "<b>IAU vs. GLD: 0.995</b> -- near-duplicate instruments; using both as independent model features "
        "risks severe multicollinearity.",
        "<b>GDX vs. GDXJ: 0.93</b> -- highly related but not interchangeable; GDXJ is the higher-beta junior-miner cousin.",
        "<b>Gold ETFs vs. GDX/GDXJ: ~0.76</b> -- miners track gold but roughly 40% of return variance is unexplained by gold alone.",
        "<b>SPY vs. TLT: -0.46</b> -- the expected stocks/bonds hedge relationship.",
    ]))
    story.append(KeepTogether(fig("results/eda/04_acf_gdx_gdxj.png", width=PAGE_WIDTH,
                                   caption="Autocorrelation of GDX and GDXJ daily returns, 20 lags (shaded = 95% CI). "
                                           "Both are close to serially uncorrelated -- no obvious short-horizon "
                                           "momentum or mean-reversion signal from lagged returns alone.")))
    story.append(Paragraph(
        "A 60-day rolling GDX/GLD correlation ranged from 0.46 to 0.92 (mean 0.79) -- weakening sharply "
        "during fast, large gold moves (e.g. the August 2011 rally), which is exactly when a fixed-correlation "
        "assumption would be most misleading. See <font face='Courier'>results/eda_summary.md</font> for the full write-up.",
        styles["Body"]))
    story.append(PageBreak())

    # ---------------- 3. Naive baseline (pre-cost) ----------------
    story.append(Paragraph("3. Naive Reversal Baseline (No Costs) and Timing Convention", styles["H1"]))
    story.append(Paragraph(
        "Before any modeling, an explicit timing specification was written down (entry at close t+1, exit at "
        "close t+2, relative to a signal known at close t) precisely because the raw data is close-price-only "
        "-- there is no open/intraday price to enter at, so this was chosen as the earliest honest execution "
        "point and verified programmatically to have zero lookahead (signal recomputed independently from "
        "raw prices and checked to match; entry/exit dates checked to land strictly later than the signal date).",
        styles["Body"]))
    story.append(Paragraph(
        "Under that timing convention, a simple, fully-invested, <b>zero-cost</b> daily reversal rule "
        "(<font face='Courier'>-sign(prior-day return)</font>) was compared against buy-and-hold on train+validation:",
        styles["Body"]))
    story.append(make_table([
        ["Strategy", "Total return", "CAGR", "Ann. vol", "Sharpe", "Max DD"],
        ["Buy-and-hold GDX", "-44.6%", "-4.6%", "42.4%", "0.10", "-80.5%"],
        ["Buy-and-hold SPY", "178.6%", "8.6%", "19.0%", "0.53", "-55.2%"],
        ["Naive reversal on GDX", "1,007.1%", "21.3%", "42.3%", "0.67", "-68.2%"],
    ], col_widths=[1.7 * inch, 1.1 * inch, 0.8 * inch, 0.9 * inch, 0.7 * inch, 0.8 * inch], highlight_row=3))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This was flagged at the time as almost certainly not investable as shown: it trades every single "
        "day (100% daily turnover) with zero transaction costs, slippage, or borrow cost assumed. Everything "
        "from Section 6 onward exists to find out how much of this apparent edge survives realistic execution.",
        styles["Body"]))

    # ---------------- 4. Features & in-sample model ----------------
    story.append(Paragraph("4. Causal Features and In-Sample Model Fit", styles["H1"]))
    story.append(Paragraph(
        "A feature set was built from train+validation data using strictly causal rolling windows (GDX "
        "trailing returns at 5/10/20/60 days, 20-day realized volatility, RSI(14), GLD/IAU/TLT 20-day "
        "momentum, GDX/GDXJ relative momentum), with the label defined as next-day GDX return. Zero-lookahead "
        "was proven programmatically via <b>truncation invariance</b>: recomputing every feature on data cut "
        "off at any date reproduces identical values up to that date. This check was confirmed to actually "
        "catch bugs by injecting a deliberate centered-rolling-window error and watching it fail; 17 automated "
        "tests (<font face='Courier'>tests/test_no_leakage.py</font>) cover this.", styles["Body"]))
    story.append(Paragraph(
        "Fit on the <b>train split only</b>: a threshold rule on trailing return, and a logistic regression "
        "on all ten features predicting next-day direction.", styles["Body"]))
    story.append(make_table([
        ["Model", "Result"],
        ["Threshold rule (best of 4 windows)", "52.9% accuracy vs. 50.8% majority baseline (+2.1pp)"],
        ["Logistic regression", "AUC 0.543, train accuracy 53.2% vs. 50.9% baseline"],
        ["Significant coefficients", "0 of 10, at p < 0.05"],
    ], col_widths=[2.6 * inch, 3.6 * inch]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This in-sample fit was already weak and consistent with the EDA's near-zero autocorrelation finding "
        "-- GDX's own past return carried little signal for its next-day direction, even before any holdout "
        "or cost considerations.", styles["Body"]))
    story.append(PageBreak())

    # ---------------- 5. Walk-forward ----------------
    story.append(Paragraph("5. Walk-Forward Grid Search (No Costs, Adaptive Direction)", styles["H1"]))
    story.append(Paragraph(
        "A quarterly-refit, expanding-window walk-forward evaluation was run on train+validation across a "
        "grid of lookback (5/10/20/60 days) x threshold (0.5%/1%/1.5%/2%) -- 16 combinations. Direction "
        "(momentum vs. reversal) was re-picked every quarter based on which had the higher realized Sharpe "
        "in the expanding window so far; no transaction costs were modeled. Top 5 by walk-forward Sharpe:",
        styles["Body"]))
    story.append(make_table([
        ["Rank", "Lookback", "Threshold", "Walk-fwd Sharpe", "Cumulative return", "Max DD"],
        ["1", "20d", "2.0%", "0.379", "+139.3%", "-68.3%"],
        ["2", "60d", "0.5%", "0.251", "+22.4%", "-69.3%"],
        ["3", "60d", "2.0%", "0.232", "+14.4%", "-66.7%"],
        ["4", "60d", "1.5%", "0.200", "-2.9%", "-75.4%"],
        ["5", "60d", "1.0%", "0.197", "-5.5%", "-71.6%"],
    ], col_widths=[0.5 * inch, 0.9 * inch, 0.9 * inch, 1.2 * inch, 1.4 * inch, 0.9 * inch], highlight_row=1))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This grid's optimism turned out to hinge heavily on two things it didn't model: zero transaction "
        "costs, and an adaptive direction that a real deployment would need to replicate somehow. Section 6 "
        "tests exactly that.", styles["Body"]))

    # ---------------- 6. Realistic backtest ----------------
    story.append(Paragraph("6. Realistic-Cost Backtest of the Top 5 Configs", styles["H1"]))
    story.append(Paragraph(
        "A custom backtest engine (market-on-close execution, configurable commission + slippage, "
        "% -of-equity position sizing with integer share lots, pluggable exit rules) re-ran the top 5 "
        "walk-forward configs with <b>$0.005/share commission + 0.05% slippage per leg</b>, a $100,000 "
        "account, <b>fixed</b> reversal direction (not the adaptive quarterly-refit direction above), and "
        "a 1-day hold:", styles["Body"]))
    story.append(make_table([
        ["Rank", "Lookback / Threshold", "Walk-fwd Sharpe", "Backtest CAGR", "Profitable after costs?"],
        ["1", "20d / 2.0%", "0.379", "+0.44%", "Yes ($5,618 net)"],
        ["2", "60d / 0.5%", "0.251", "-18.2%", "No (-$91,866)"],
        ["3", "60d / 2.0%", "0.232", "-8.0%", "No (-$64,818)"],
        ["4", "60d / 1.5%", "0.200", "-16.4%", "No (-$89,282)"],
        ["5", "60d / 1.0%", "0.197", "-15.5%", "No (-$87,833)"],
    ], col_widths=[0.5 * inch, 1.6 * inch, 1.2 * inch, 1.1 * inch, 2.0 * inch], highlight_row=1))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Only rank 1 survived, and the reason is <b>not mainly transaction costs</b>: rank 1's gross P&L at "
        "quoted prices was ~$233K against only ~$59K of commission. The dominant effect is that the "
        "walk-forward grid earned its Sharpes under an <b>adaptive</b>, quarterly-refit direction, while this "
        "backtest uses one <b>fixed</b> direction for the whole 12.5-year period -- being on the wrong side of "
        "a volatile, trending asset for extended stretches at 100% position sizing is costly.", styles["Body"]))
    story.append(KeepTogether(fig("results/backtest/equity_curve_comparison.png", width=PAGE_WIDTH,
                                   caption="Rank-1 config: fixed 1-day hold (orange) vs. an alternative exit rule, "
                                           "exit-on-signal-change (blue). Same signal, same costs -- exit rule alone "
                                           "swings the outcome from a loss to a substantial gain.")))
    story.append(PageBreak())

    # ---------------- 7. Robustness ----------------
    story.append(Paragraph("7. Robustness Checks on the Single Surviving Config", styles["H1"]))
    story.append(Paragraph(
        "The only profitable-after-costs config (lookback=20d, threshold=2.0%) was stress-tested four ways, "
        "all on train+validation with the same costs/exit rule/direction unless a check explicitly varied one.",
        styles["Body"]))

    story.append(Paragraph("7.1 Parameter perturbation (+/-20% on lookback and threshold)", styles["H2"]))
    story.append(KeepTogether(fig("results/robustness/plot_perturbation_heatmap.png", width=3.4 * inch,
                                   caption="Sharpe across the 3x3 grid (CAGR in parens); baseline outlined. "
                                           "Only 2 of 9 nearby configs are profitable.")))
    story.append(Paragraph(
        "A config whose immediate neighbors in parameter space are mostly unprofitable is a narrow, "
        "not-very-robust local optimum rather than a broad, stable edge.", styles["Body"]))

    story.append(Paragraph("7.2 Performance by sub-period (4 equal chronological chunks)", styles["H2"]))
    story.append(KeepTogether(fig("results/robustness/plot_subperiod_equity.png", width=PAGE_WIDTH,
                                   caption="Baseline equity curve, sliced into 4 equal sub-periods and shaded by "
                                           "profitable (green) / unprofitable (red). Only 2 of 4 are profitable.")))
    story.append(make_table([
        ["Sub-period", "Dates", "CAGR", "Sharpe", "Profitable?"],
        ["1", "2006-05-22 to 2009-07-07", "+8.9%", "0.413", "Yes"],
        ["2", "2009-07-08 to 2012-08-17", "-1.4%", "0.052", "No"],
        ["3", "2012-08-20 to 2015-10-05", "-4.5%", "-0.011", "No"],
        ["4", "2015-10-06 to 2018-11-15", "+0.5%", "0.139", "Yes"],
    ], col_widths=[0.9 * inch, 2.1 * inch, 0.9 * inch, 0.9 * inch, 1.1 * inch]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("7.3 Cost sensitivity (2x commission and slippage)", styles["H2"]))
    story.append(KeepTogether(fig("results/robustness/plot_cost_sensitivity.png", width=PAGE_WIDTH,
                                   caption="Doubling costs flips CAGR from +0.44% to -13.8% and Sharpe from "
                                           "0.157 to -0.376.")))
    story.append(Paragraph(
        "7.4 Trade frequency: all 4 sub-periods pass a minimum >=1 trade/month bar comfortably (8.7-9.2/month "
        "each) -- trade frequency is not the weak point here.", styles["Body"]))
    story.append(PageBreak())

    # ---------------- 8. Significance ----------------
    story.append(Paragraph("8. Statistical Significance", styles["H1"]))
    story.append(Paragraph(
        "Two checks on the same final config, on train+validation:", styles["Body"]))
    story.append(bullets([
        "<b>T-test</b> of mean daily strategy return vs. zero: t = 0.556, two-sided p = 0.578 -- "
        "<b>not significant</b> at the 5% level.",
        "<b>Bootstrap</b> of the 1,384 trade-level returns (1,000 resamples, seed=42), each iteration's "
        "Sharpe annualized by sqrt(trades/year): point-estimate Sharpe 0.157, but the 5th/95th percentile "
        "interval is <b>[-0.33, 0.64]</b> -- straddling zero.",
    ]))
    story.append(KeepTogether(fig("results/significance/plot_bootstrap_sharpe.png", width=5.2 * inch,
                                   caption="Bootstrap distribution of trade-level Sharpe (1,000 resamples). "
                                           "Zero sits well inside the bulk of the distribution, not in a tail.")))
    story.append(Paragraph(
        "Both findings restate, in standard statistical form, what the walk-forward optimism, the thin "
        "backtest margin, and the robustness fragility already suggested: this is not a validated edge.",
        styles["Body"]))

    # ---------------- 9. Frozen config ----------------
    story.append(Paragraph("9. Frozen Final Configuration", styles["H1"]))
    story.append(Paragraph(
        "The exact parameters selected from this in-sample process were written to "
        "<font face='Courier'>results/FROZEN_CONFIG.json</font>, with instructions not to modify or re-tune "
        "them further:", styles["Body"]))
    story.append(make_table([
        ["Parameter", "Value"],
        ["Ticker", "GDX"],
        ["Signal", "Reversal on 20-day trailing return, 2.0% deadzone threshold"],
        ["Direction", "Fixed reversal (not adaptively refit)"],
        ["Exit rule", "1-day hold (exit_after_n_days(0)) -- market-on-close, entry t+1 / exit t+2"],
        ["Position sizing", "100% of equity, integer share lots"],
        ["Cost assumptions", "$0.005/share commission + 0.05% slippage per leg"],
        ["Reference account size", "$100,000 (in-sample evaluation only, not a deployment constraint)"],
    ], col_widths=[2.1 * inch, 4.1 * inch]))
    story.append(Spacer(1, 10))

    # ---------------- 10. Conclusion ----------------
    story.append(Paragraph("10. Conclusion and Recommendations", styles["H1"]))
    story.append(Paragraph(
        "Every stage of this research process pointed the same direction: an initially promising signal "
        "(a zero-cost naive reversal rule, then a walk-forward grid search) shrank steadily as more realism "
        "was added -- real costs, a fixed rather than adaptive direction, parameter neighborhoods, "
        "sub-period consistency, and finally formal significance testing. What's left is a config with a "
        "CAGR of well under 1%, a Sharpe not distinguishable from zero, and profits concentrated in two "
        "specific multi-year windows rather than persistent.", styles["Body"]))
    story.append(Paragraph("Recommendations:", styles["H2"]))
    story.append(bullets([
        "Do not treat the frozen config as a trading strategy ready for capital -- treat it as the "
        "end product of a disciplined process that happens to have found a thin, fragile, statistically "
        "insignificant result.",
        "If a holdout evaluation is run, it should be a single, final look (per the reserved 786-day "
        "split) -- not a new round of tuning against holdout performance, which would undo the entire "
        "point of having kept it untouched.",
        "The adaptive quarterly direction-refit mechanism (walk-forward) meaningfully outperformed the "
        "fixed-direction deployment in-sample; if this line of research continues, building that "
        "adaptive mechanism into the live/holdout-facing engine (rather than fixing direction once) is "
        "the most promising unexplored lever.",
        "IAU/GLD collinearity should be resolved (drop one, or use a spread/ratio feature) before any "
        "further feature-based modeling.",
    ]))
    story.append(Spacer(1, 16))
    story.append(rule())
    story.append(Paragraph(
        "Full detail, code, and every intermediate artifact referenced above are in the project repository "
        "under <font face='Courier'>results/</font> and <font face='Courier'>src/</font>: "
        "<font face='Courier'>eda_summary.md, baselines.md, timing_spec.md, naive_baseline/, naive_models/, "
        "walk_forward_grid.csv, backtest_results.csv, backtest_results_notes.md, robustness_report.md, "
        "significance_report.md, FROZEN_CONFIG.json</font>.", styles["Meta"]))

    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
