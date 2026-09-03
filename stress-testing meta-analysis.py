# Stress-Testing Meta-Analysis: Reproducible Technical-Report Pipeline
# Intended for Google Colab / standard Python environments.
#
# This script demonstrates:
#   1. Input validation
#   2. Hedges g effect-size calculation
#   3. Fixed-effect pooling
#   4. DerSimonian-Laird and Paule-Mandel tau^2 estimation
#   5. Random-effects pooling
#   6. Explicit conservative HKSJ-style CI adjustment
#   7. Independently calculated random-effects prediction interval
#   8. Leave-one-out influence analysis
#   9. Automated identification of the most influential study
#  10. Egger-type regression for small-study effects
#  11. Automated robustness summary
#  12. Machine-readable output files
#  13. Dynamically scaled forest plot
#
# IMPORTANT:
# The included dataset is synthetic/illustrative. Replace Section 1 with
# appropriately extracted real study-level data for an actual meta-analysis.

import math
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt


# =============================================================================
# 1. INPUT DATA
# =============================================================================

data = pd.DataFrame([
    ["Study 01", 48, 45, 70.8, 11.2, 75.1, 10.8],
    ["Study 02", 62, 59, 64.2, 12.0, 69.5, 11.4],
    ["Study 03", 36, 34, 81.4, 13.1, 76.2, 12.7],
    ["Study 04", 55, 58, 72.0, 9.8, 74.5, 10.2],
    ["Study 05", 28, 31, 66.1, 11.9, 73.0, 12.1],
    ["Study 06", 80, 78, 59.8, 13.4, 66.5, 13.0],
    ["Study 07", 44, 46, 77.2, 10.7, 71.1, 11.6],
    ["Study 08", 95, 93, 68.9, 12.6, 72.0, 12.0],
    ["Study 09", 33, 35, 74.8, 10.4, 79.1, 11.2],
    ["Study 10", 51, 49, 61.5, 9.5, 68.2, 10.1],
    ["Study 11", 40, 42, 71.9, 12.2, 69.8, 11.4],
    ["Study 12", 70, 68, 67.4, 11.7, 73.6, 12.5],
], columns=["study", "n_t", "n_c", "mean_t", "sd_t", "mean_c", "sd_c"])


# =============================================================================
# 2. INPUT VALIDATION
# =============================================================================

required = ["study", "n_t", "n_c", "mean_t", "sd_t", "mean_c", "sd_c"]
missing = [c for c in required if c not in data.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if data.empty:
    raise ValueError("The input dataset is empty.")

if data["study"].duplicated().any():
    raise ValueError("Study identifiers must be unique.")

if data[required].isna().any().any():
    raise ValueError("Input data contain missing values.")

for col in ["n_t", "n_c"]:
    if (data[col] <= 0).any() or (data[col] % 1 != 0).any():
        raise ValueError(f"{col} must contain positive integers.")

for col in ["sd_t", "sd_c"]:
    if (data[col] <= 0).any():
        raise ValueError(f"{col} must contain strictly positive values.")

if len(data) < 2:
    raise ValueError("At least two studies are required for meta-analysis.")


# =============================================================================
# 3. HEDGES g AND APPROXIMATE SAMPLING VARIANCE
# =============================================================================

def add_hedges_g(row):
    nt = int(row.n_t)
    nc = int(row.n_c)
    df = nt + nc - 2

    if df <= 0:
        raise ValueError(f"Invalid degrees of freedom for {row.study}.")

    sp = math.sqrt(
        ((nt - 1) * row.sd_t**2 + (nc - 1) * row.sd_c**2) / df
    )

    d = (row.mean_t - row.mean_c) / sp
    J = 1 - 3 / (4 * df - 1)
    g = J * d

    # Conventional approximate sampling variance for Hedges g.
    var_g = (nt + nc) / (nt * nc) + g**2 / (2 * df)

    return pd.Series({
        "yi": g,
        "vi": var_g,
        "se": math.sqrt(var_g)
    })


effect_sizes = data.apply(add_hedges_g, axis=1)
data = pd.concat([data, effect_sizes], axis=1)

y = data["yi"].to_numpy(dtype=float)
v = data["vi"].to_numpy(dtype=float)


# =============================================================================
# 4. FIXED-EFFECT MODEL
# =============================================================================

def pool_fixed(y, v):
    w = 1 / v
    mu = np.sum(w * y) / np.sum(w)
    se = math.sqrt(1 / np.sum(w))

    q = np.sum(w * (y - mu) ** 2)
    df = len(y) - 1

    crit = stats.norm.ppf(0.975)
    ci = (mu - crit * se, mu + crit * se)

    return {
        "mu": mu,
        "se": se,
        "q": q,
        "q_df": df,
        "q_p": stats.chi2.sf(q, df),
        "ci": ci,
    }


# =============================================================================
# 5. HETEROGENEITY ESTIMATORS
# =============================================================================

def tau2_der_simonian_laird(y, v):
    """DerSimonian-Laird estimator, truncated at zero."""
    w = 1 / v
    mu = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mu) ** 2)
    df = len(y) - 1

    c = np.sum(w) - np.sum(w**2) / np.sum(w)

    if c <= 0:
        return 0.0

    return max(0.0, (q - df) / c)


def tau2_paule_mandel(y, v):
    """Paule-Mandel estimator obtained by solving Q(tau^2) = k - 1."""
    k = len(y)

    def q_minus_df(tau2):
        w = 1 / (v + tau2)
        mu = np.sum(w * y) / np.sum(w)
        return np.sum(w * (y - mu) ** 2) - (k - 1)

    if q_minus_df(0.0) <= 0:
        return 0.0

    hi = max(np.var(y, ddof=1), np.max(v), 1.0)

    while q_minus_df(hi) > 0:
        hi *= 2

    return optimize.brentq(q_minus_df, 0.0, hi)


# =============================================================================
# 6. RANDOM-EFFECTS POOLING
# =============================================================================

def pool_random(y, v, tau2, hksj=False):
    """
    Random-effects pooled estimate.

    If hksj=True, a conservative HKSJ-style variance adjustment is used:
        HK factor = max(1, Q / df)

    The pooled-effect CI uses the corresponding t critical value.

    The prediction interval is calculated independently from the CI standard
    error. It uses:
        prediction SE = sqrt(1 / sum(weights) + tau^2)

    Thus, the HKSJ-adjusted CI standard error is NOT reused for the prediction
    interval.
    """
    w = 1 / (v + tau2)
    mu = np.sum(w * y) / np.sum(w)

    q = np.sum(w * (y - mu) ** 2)
    k = len(y)
    df = k - 1

    if hksj:
        # Conservative HKSJ-style variance adjustment.
        hk_factor = max(1.0, q / df)
        se = math.sqrt(hk_factor / np.sum(w))
        crit = stats.t.ppf(0.975, df)
    else:
        hk_factor = 1.0
        se = math.sqrt(1 / np.sum(w))
        crit = stats.norm.ppf(0.975)

    ci = (mu - crit * se, mu + crit * se)

    # Prediction interval calculated independently from the CI SE.
    pred_se = math.sqrt(1 / np.sum(w) + tau2)
    pred_df = max(1, k - 2)
    pred_crit = stats.t.ppf(0.975, pred_df)
    pi = (
        mu - pred_crit * pred_se,
        mu + pred_crit * pred_se
    )

    return {
        "mu": mu,
        "se": se,
        "tau2": tau2,
        "q": q,
        "q_df": df,
        "q_p": stats.chi2.sf(q, df),
        "ci": ci,
        "pi": pi,
        "pred_se": pred_se,
        "pred_df": pred_df,
        "weights": w,
        "hk_factor": hk_factor,
    }


# =============================================================================
# 7. PRIMARY AND SENSITIVITY MODELS
# =============================================================================

fixed = pool_fixed(y, v)

tau_dl = tau2_der_simonian_laird(y, v)
tau_pm = tau2_paule_mandel(y, v)

primary = pool_random(y, v, tau_dl, hksj=False)
hksj = pool_random(y, v, tau_dl, hksj=True)
pm = pool_random(y, v, tau_pm, hksj=False)


# =============================================================================
# 8. LEAVE-ONE-OUT INFLUENCE STRESS TEST
# =============================================================================

loo = []

for i, study in enumerate(data["study"]):
    mask = np.arange(len(data)) != i

    tau = tau2_der_simonian_laird(y[mask], v[mask])
    result = pool_random(y[mask], v[mask], tau, hksj=False)

    change = result["mu"] - primary["mu"]

    loo.append({
        "study": study,
        "pooled_g_without_study": result["mu"],
        "ci_lo": result["ci"][0],
        "ci_hi": result["ci"][1],
        "tau2": result["tau2"],
        "change_from_full": change,
        "absolute_change": abs(change),
    })

loo = pd.DataFrame(loo)

most_influential = loo.loc[loo["absolute_change"].idxmax()]

print("\nMOST INFLUENTIAL STUDY")
print("----------------------")
print(
    "Study with largest absolute change:",
    most_influential["study"]
)
print(
    "Absolute change:",
    f'{most_influential["absolute_change"]:.2f}'
)
print(
    "Exact absolute change:",
    f'{most_influential["absolute_change"]:.6f}'
)


# =============================================================================
# 9. SMALL-STUDY EFFECTS: EGGER-TYPE REGRESSION
# =============================================================================

def egger_type_regression(dataframe):
    """
    Egger-type regression of standardized effect size on precision.

    Model:
        yi / SE_i = intercept + slope * (1 / SE_i)

    The p-value reported here is specifically for the INTERCEPT, not the slope.
    A significant intercept is a signal compatible with small-study effects;
    it is not, by itself, proof of publication bias.
    """
    precision = 1 / dataframe["se"].to_numpy(dtype=float)
    standardized = (
        dataframe["yi"].to_numpy(dtype=float)
        / dataframe["se"].to_numpy(dtype=float)
    )

    n = len(precision)

    if n < 3:
        raise ValueError("Egger-type regression requires at least 3 studies.")

    regression = stats.linregress(precision, standardized)

    xbar = np.mean(precision)
    ybar = np.mean(standardized)
    sxx = np.sum((precision - xbar) ** 2)

    if sxx <= 0:
        raise ValueError("Precision values have zero variance.")

    fitted = regression.intercept + regression.slope * precision
    residuals = standardized - fitted

    residual_df = n - 2
    s2 = np.sum(residuals ** 2) / residual_df

    intercept_se = math.sqrt(
        s2 * (1 / n + (xbar**2 / sxx))
    )

    intercept_t = regression.intercept / intercept_se
    intercept_p = 2 * stats.t.sf(abs(intercept_t), residual_df)

    return {
        "intercept": regression.intercept,
        "intercept_se": intercept_se,
        "intercept_t": intercept_t,
        "intercept_p": intercept_p,
        "slope": regression.slope,
        "slope_p": regression.pvalue,
        "n_studies": n,
    }


if len(data) >= 10:
    egger = egger_type_regression(data)

    print("\nEGGER-TYPE REGRESSION")
    print("---------------------")
    print(
        "Egger-type regression intercept:",
        f'{egger["intercept"]:.4f}'
    )
    print(
        "Intercept p-value:",
        f'{egger["intercept_p"]:.4f}'
    )
    print(
        "Interpretation: treat the intercept test as a sensitivity signal "
        "for small-study effects, not as a diagnosis of publication bias."
    )
else:
    egger = None
    print(
        "\nSmall-study effects test suppressed: fewer than 10 studies."
    )


# =============================================================================
# 10. AUTOMATED ROBUSTNESS SUMMARY
# =============================================================================

# Direction stability:
# "Stable" means every leave-one-out pooled estimate retains the same sign
# as the primary pooled estimate.
primary_direction = np.sign(primary["mu"])
loo_directions = np.sign(loo["pooled_g_without_study"])
direction_stable = np.all(loo_directions == primary_direction)

# Precision stability:
# Compare the width of the HKSJ-adjusted CI with the conventional Wald CI.
wald_width = primary["ci"][1] - primary["ci"][0]
hksj_width = hksj["ci"][1] - hksj["ci"][0]
precision_stable = hksj_width >= wald_width

# Influence stability:
# A practical, explicitly stated threshold of 0.10 Hedges g is used.
# This threshold is a reporting convention for this demonstration, not a
# universal statistical cutoff.
influence_threshold = 0.10
max_absolute_change = loo["absolute_change"].max()
influence_stable = max_absolute_change <= influence_threshold

# Heterogeneity-estimator sensitivity:
# A difference <= 0.10 Hedges g is treated as stable for this demonstration.
pooled_difference_dl_pm = abs(primary["mu"] - pm["mu"])
heterogeneity_threshold = 0.10
heterogeneity_stable = pooled_difference_dl_pm <= heterogeneity_threshold

# Small-study effects status.
if egger is None:
    small_study_summary = "Not assessed (<10 studies)"
elif egger["intercept_p"] < 0.05:
    small_study_summary = (
        "Egger-type intercept statistically significant; "
        "interpret as a sensitivity signal, not proof of publication bias"
    )
else:
    small_study_summary = (
        "Egger-type intercept not statistically significant; "
        "no clear small-study signal detected"
    )

robustness_summary = pd.DataFrame([
    {
        "domain": "Direction stability",
        "assessment": "Stable" if direction_stable else "Changed",
        "detail": (
            "All leave-one-out estimates retained the direction of the "
            "primary pooled estimate."
            if direction_stable
            else
            "At least one leave-one-out estimate changed the direction "
            "of the pooled estimate."
        ),
    },
    {
        "domain": "Precision stability",
        "assessment": "Stable" if precision_stable else "Changed",
        "detail": (
            f"HKSJ CI width = {hksj_width:.3f}; "
            f"Wald CI width = {wald_width:.3f}."
        ),
    },
    {
        "domain": "Influence stability",
        "assessment": "Stable" if influence_stable else "Potential influence",
        "detail": (
            f"Maximum absolute leave-one-out change = "
            f"{max_absolute_change:.3f}; demonstration threshold = "
            f"{influence_threshold:.2f} Hedges g."
        ),
    },
    {
        "domain": "Most influential study",
        "assessment": str(most_influential["study"]),
        "detail": (
            f"Largest absolute change = "
            f'{most_influential["absolute_change"]:.3f} Hedges g.'
        ),
    },
    {
        "domain": "Heterogeneity estimator sensitivity",
        "assessment": "Stable" if heterogeneity_stable else "Changed",
        "detail": (
            f"DL pooled g = {primary['mu']:.3f}; "
            f"PM pooled g = {pm['mu']:.3f}; "
            f"absolute difference = {pooled_difference_dl_pm:.3f}; "
            f"demonstration threshold = {heterogeneity_threshold:.2f}."
        ),
    },
    {
        "domain": "Small-study effects",
        "assessment": "Egger-type regression performed" if egger is not None
                    else "Not assessed",
        "detail": small_study_summary,
    },
])

print("\nROBUSTNESS SUMMARY")
print("------------------")
for _, row in robustness_summary.iterrows():
    print(f'{row["domain"]}: {row["assessment"]}')
    print(f'  {row["detail"]}')

print("\nMOST INFLUENTIAL STUDY")
print("----------------------")
print(
    f'{most_influential["study"]} produced the largest absolute change '
    f'of {most_influential["absolute_change"]:.2f} Hedges g.'
)


# =============================================================================
# 11. REPRODUCIBLE STATISTICAL OUTPUT
# =============================================================================

print("\nMODEL RESULTS")
print("-------------")
print(f"Number of studies: {len(data)}")

print("\nFixed-effect model:")
print(f"  Pooled g = {fixed['mu']:.4f}")
print(f"  95% CI = ({fixed['ci'][0]:.4f}, {fixed['ci'][1]:.4f})")
print(f"  Q = {fixed['q']:.4f}, p = {fixed['q_p']:.4f}")

print("\nDerSimonian-Laird:")
print(f"  tau^2 = {tau_dl:.6f}")

print("\nPaule-Mandel:")
print(f"  tau^2 = {tau_pm:.6f}")

print("\nPrimary random-effects model (DL):")
print(f"  Pooled g = {primary['mu']:.4f}")
print(f"  95% Wald CI = ({primary['ci'][0]:.4f}, {primary['ci'][1]:.4f})")
print(
    f"  Prediction interval = "
    f"({primary['pi'][0]:.4f}, {primary['pi'][1]:.4f})"
)
print(f"  Prediction SE = {primary['pred_se']:.4f}")
print(f"  Prediction df = {primary['pred_df']}")

print("\nHKSJ-adjusted primary model:")
print(f"  Pooled g = {hksj['mu']:.4f}")
print(f"  95% HKSJ-style CI = ({hksj['ci'][0]:.4f}, {hksj['ci'][1]:.4f})")
print(f"  HKSJ factor = {hksj['hk_factor']:.4f}")

print("\nPaule-Mandel sensitivity model:")
print(f"  Pooled g = {pm['mu']:.4f}")
print(f"  95% Wald CI = ({pm['ci'][0]:.4f}, {pm['ci'][1]:.4f})")
print(f"  tau^2 = {pm['tau2']:.6f}")

print("\nLeave-one-out pooled range:")
print(
    f"  {loo['pooled_g_without_study'].min():.4f} to "
    f"{loo['pooled_g_without_study'].max():.4f}"
)


# =============================================================================
# 12. SAVE MACHINE-READABLE OUTPUTS
# =============================================================================

data.to_csv("meta_effect_sizes.csv", index=False)
loo.to_csv("meta_leave_one_out.csv", index=False)
robustness_summary.to_csv("meta_robustness_summary.csv", index=False)

# A compact model-results table for auditability.
model_results = pd.DataFrame([
    {
        "model": "Fixed effect",
        "pooled_g": fixed["mu"],
        "ci_low": fixed["ci"][0],
        "ci_high": fixed["ci"][1],
        "tau2": np.nan,
        "prediction_low": np.nan,
        "prediction_high": np.nan,
    },
    {
        "model": "Random effects - DL",
        "pooled_g": primary["mu"],
        "ci_low": primary["ci"][0],
        "ci_high": primary["ci"][1],
        "tau2": primary["tau2"],
        "prediction_low": primary["pi"][0],
        "prediction_high": primary["pi"][1],
    },
    {
        "model": "Random effects - DL + HKSJ-style CI",
        "pooled_g": hksj["mu"],
        "ci_low": hksj["ci"][0],
        "ci_high": hksj["ci"][1],
        "tau2": hksj["tau2"],
        "prediction_low": hksj["pi"][0],
        "prediction_high": hksj["pi"][1],
    },
    {
        "model": "Random effects - Paule-Mandel",
        "pooled_g": pm["mu"],
        "ci_low": pm["ci"][0],
        "ci_high": pm["ci"][1],
        "tau2": pm["tau2"],
        "prediction_low": pm["pi"][0],
        "prediction_high": pm["pi"][1],
    },
])

model_results.to_csv("meta_model_results.csv", index=False)


# =============================================================================
# 13. DYNAMIC FOREST PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

ypos = np.arange(len(data), 0, -1)

# Study-level 95% CIs.
study_lo = data["yi"] - 1.96 * data["se"]
study_hi = data["yi"] + 1.96 * data["se"]

ax.errorbar(
    data["yi"],
    ypos,
    xerr=1.96 * data["se"],
    fmt="o",
    capsize=3,
    linestyle="none",
)

ax.axvline(0, linewidth=1)

# Pooled-effect diamond.
py = len(data) + 1.5
lo, hi = primary["ci"]
m = primary["mu"]

dx = [lo, m, hi, m, lo]
dy = [py, py + 0.25, py, py - 0.25, py]

ax.fill(dx, dy, alpha=0.35)
ax.plot([lo, hi], [py, py], linewidth=1.5)

# Determine plot limits dynamically from study CIs and pooled estimates.
all_x = np.concatenate([
    study_lo.to_numpy(),
    study_hi.to_numpy(),
    np.array([
        primary["ci"][0],
        primary["ci"][1],
        primary["pi"][0],
        primary["pi"][1],
    ]),
])

xmin = np.min(all_x)
xmax = np.max(all_x)

data_range = xmax - xmin

if data_range <= 0:
    data_range = 1.0

margin = max(0.10 * data_range, 0.20)

plot_xmin = xmin - margin
plot_xmax = xmax + margin

ax.set_xlim(plot_xmin, plot_xmax)

# Study labels are placed just outside the left plotting boundary.
label_x = plot_xmin + 0.01 * data_range

for y0, study in zip(ypos, data["study"]):
    ax.text(
        label_x,
        y0,
        study,
        ha="left",
        va="center",
        fontsize=8,
    )

ax.text(
    label_x,
    py,
    "Random-effects pooled",
    ha="left",
    va="center",
    fontsize=8,
    fontweight="bold",
)

ax.set_ylim(0.5, len(data) + 2.5)
ax.set_yticks([])

ax.set_xlabel("Hedges g")
ax.set_title("Forest plot of the illustrative 12-study dataset")

fig.tight_layout()
fig.savefig(
    "Figure_2_forest.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("\nFILES SAVED")
print("-----------")
print("meta_effect_sizes.csv")
print("meta_leave_one_out.csv")
print("meta_robustness_summary.csv")
print("meta_model_results.csv")
print("Figure_2_forest.png")

print("\nPipeline completed successfully.")
