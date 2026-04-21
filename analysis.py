"""
LLM Accessibility Research - Data Analysis & Visualization Script
Install with: pip3 install pandas matplotlib scipy numpy openpyxl
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")


# Configurations

EXCEL_FILE = "llm_accessibility_results.xlsx"   # must be in same folder as this script
OUTPUT_FOLDER = "charts"                          # folder where charts will be saved
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Column names
COL_MODEL       = "Model"
COL_PROMPT      = "Prompt"
COL_DASHBOARD   = "Dashboard_ID"
COL_ISSUE       = "Issue_Type"
COL_FP          = "False_Positives"
COL_DETECT      = "Detection_Accuracy"
COL_IMPL        = "Implementation_Accuracy"
COL_REASON      = "Code_Reasoning"
COL_VIOLATION   = "Violation_Presence"
COL_TOTAL       = "Total_Score"

METRICS = [COL_FP, COL_DETECT, COL_IMPL, COL_REASON, COL_VIOLATION]
METRIC_LABELS = ["False\nPositives", "Detection\nAccuracy", "Implementation\nAccuracy",
                 "Code\nReasoning", "Violation\nPresence"]
MAX_SCORE = 15  # perfect score = 3 per metric × 5 metrics
PERFECT_SCORE = 15  # 5 metrics × 3 = 15

MODEL_COLORS = {"GPT": "#4C9BE8", "Claude": "#E87B4C", "Gemini": "#4CAF7D"}
PROMPT_COLORS = {"P1": "#7B68EE", "P2": "#FF8C69"}


# Load the data
def load_data():
    print(f"\Loading data from '{EXCEL_FILE}'...")
    try:
        df = pd.read_excel(EXCEL_FILE)
    except FileNotFoundError:
        print(f"\nERROR: Could not find '{EXCEL_FILE}'.")
        print("   Make sure the Excel file is in the SAME folder as this script.")
        exit()

    # Auto-calculate Total_Score if missing or all zeros
    if COL_TOTAL not in df.columns or df[COL_TOTAL].sum() == 0:
        df[COL_TOTAL] = df[METRICS].sum(axis=1)
        print("   Total_Score was recalculated from individual metric columns.")

    print(f"   Loaded {len(df)} rows across {df[COL_MODEL].nunique()} models.\n")
    return df



# 1. Model Performance/Comparison

def analyze_model_performance(df):
    print("=" * 55)
    print("1. MODEL PERFORMANCE COMPARISON")
    print("=" * 55)

    # Mean and SD of total score per model
    model_stats = df.groupby(COL_MODEL)[COL_TOTAL].agg(["mean", "std", "count"])
    model_stats.columns = ["Mean Total Score", "Std Dev", "N"]
    model_stats = model_stats.round(2)
    print("\nMean Total Score per Model:")
    print(model_stats.to_string())

    # Mean per metric per model
    metric_means = df.groupby(COL_MODEL)[METRICS].mean().round(2)
    print("\nMean Score per Metric per Model:")
    print(metric_means.to_string())

    # One-way ANOVA
    groups = [group[COL_TOTAL].values for _, group in df.groupby(COL_MODEL)]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nOne-Way ANOVA: F = {f_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("   Statistically significant difference between models (p < 0.05)")
    else:
        print("   No statistically significant difference between models (p ≥ 0.05)")

    return model_stats, metric_means


# 2. Prompt Comparison
def analyze_prompt_comparison(df):
    print("\n" + "=" * 55)
    print("2. PROMPT COMPARISON")
    print("=" * 55)

    prompt_means = df.groupby(COL_PROMPT)[COL_TOTAL].agg(["mean", "std", "count"]).round(2)
    prompt_means.columns = ["Mean Total Score", "Std Dev", "N"]
    print("\nMean Total Score per Prompt:")
    print(prompt_means.to_string())

    # % difference
    means = prompt_means["Mean Total Score"]
    if "P1" in means and "P2" in means:
        pct_diff = ((means["P2"] - means["P1"]) / means["P1"]) * 100
        print(f"\nP2 vs P1 % Change: {pct_diff:+.1f}%")

    # Paired t-test: match by Model + Dashboard_ID
    p1 = df[df[COL_PROMPT] == "P1"].sort_values([COL_MODEL, COL_DASHBOARD])[COL_TOTAL].values
    p2 = df[df[COL_PROMPT] == "P2"].sort_values([COL_MODEL, COL_DASHBOARD])[COL_TOTAL].values
    min_len = min(len(p1), len(p2))
    t_stat, p_val = stats.ttest_rel(p1[:min_len], p2[:min_len])
    print(f"\nPaired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("   Statistically significant difference between prompts (p < 0.05)")
    else:
        print("   No statistically significant difference between prompts (p ≥ 0.05)")

    return prompt_means



# 3. Success Rate
def analyze_success_rate(df):
    print("\n" + "=" * 55)
    print("3. SUCCESS RATE  (Total Score = 15 = perfect)")
    print("=" * 55)

    df["Success"] = df[COL_TOTAL] == PERFECT_SCORE

    # Per model
    model_success = df.groupby(COL_MODEL)["Success"].agg(["sum", "count"])
    model_success["Percentage"] = (model_success["sum"] / model_success["count"] * 100).round(1)
    model_success.columns = ["Successes", "Total Runs", "Success %"]
    print("\nSuccess Rate per Model:")
    print(model_success.to_string())

    # Per prompt
    prompt_success = df.groupby(COL_PROMPT)["Success"].agg(["sum", "count"])
    prompt_success["Percentage"] = (prompt_success["sum"] / prompt_success["count"] * 100).round(1)
    prompt_success.columns = ["Successes", "Total Runs", "Success %"]
    print("\nSuccess Rate per Prompt:")
    print(prompt_success.to_string())


# 4. Error Analysis
def analyze_errors(df):
    print("\n" + "=" * 55)
    print("4. ERROR ANALYSIS  (Score 0 or 1 = failure)")
    print("=" * 55)

    error_df = pd.DataFrame()
    for metric in METRICS:
        fail_counts = df[df[metric] <= 1].groupby(COL_MODEL)[metric].count()
        error_df[metric] = fail_counts

    error_df = error_df.fillna(0).astype(int)
    error_df.columns = METRIC_LABELS
    print("\nFailure Count per Metric per Model (score ≤ 1):")
    print(error_df.to_string())

    return error_df


# 5. Issue-Type Breakdown
def analyze_issue_type(df):
    print("\n" + "=" * 55)
    print("5. ISSUE-TYPE BREAKDOWN")
    print("=" * 55)

    issue_model = df.groupby([COL_ISSUE, COL_MODEL])[COL_TOTAL].mean().round(2).unstack()
    print("\nMean Total Score by Issue Type and Model:")
    print(issue_model.to_string())

    return issue_model


# 6. Violation Comparison
def analyze_violations(df):
    print("\n" + "=" * 55)
    print("6. COMPARISON OF VIOLATIONS")
    print("=" * 55)

    vio_stats = df.groupby(COL_ISSUE)[COL_TOTAL].agg(["mean", "std", "count"]).round(2)
    vio_stats.columns = ["Mean Total Score", "Std Dev", "N"]
    print("\nMean Total Score per Violation Type:")
    print(vio_stats.to_string())

    # Independent t-test between the two violation types
    issue_types = df[COL_ISSUE].unique()
    if len(issue_types) == 2:
        g1 = df[df[COL_ISSUE] == issue_types[0]][COL_TOTAL].values
        g2 = df[df[COL_ISSUE] == issue_types[1]][COL_TOTAL].values
        t_stat, p_val = stats.ttest_ind(g1, g2)
        print(f"\nIndependent t-test ({issue_types[0]} vs {issue_types[1]}): t = {t_stat:.3f}, p = {p_val:.4f}")
        if p_val < 0.05:
            print("   Significant difference in difficulty between violation types (p < 0.05)")
        else:
            print("   No significant difference in difficulty between violation types (p ≥ 0.05)")
    else:
        print(f"\n   Found {len(issue_types)} issue types: {list(issue_types)}")
        print("   Independent t-test requires exactly 2 groups - skipping.")



# Visualizations

def save(fig, filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"   Saved → {path}")
    plt.close(fig)


# Chart 1: Avg Score per Model (Bar Chart)
def chart_model_comparison(df):
    print("\n Generating Chart 1: Model Comparison Bar Chart ")
    model_means = df.groupby(COL_MODEL)[COL_TOTAL].mean()
    model_stds  = df.groupby(COL_MODEL)[COL_TOTAL].std()
    models = model_means.index.tolist()
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, model_means.values, color=colors,
                  yerr=model_stds.values, capsize=6, edgecolor="white", linewidth=1.2, width=0.5)

    for bar, val in zip(bars, model_means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Average Total Score per Model", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Total Score (out of 15)", fontsize=12)
    ax.set_ylim(0, PERFECT_SCORE + 2)
    ax.axhline(PERFECT_SCORE, color="gray", linestyle="--", linewidth=1, label="Perfect Score (15)")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "chart1_model_comparison.png")


# Chart 2: Prompt Comparison (Bar Chart)
def chart_prompt_comparison(df):
    print(" Generating Chart 2: Prompt Comparison Bar Chart ")
    prompt_model = df.groupby([COL_MODEL, COL_PROMPT])[COL_TOTAL].mean().unstack()
    models = prompt_model.index.tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, prompt_model.get("P1", [0]*len(models)),
                   width, label="Prompt 1", color=PROMPT_COLORS["P1"], edgecolor="white")
    bars2 = ax.bar(x + width / 2, prompt_model.get("P2", [0]*len(models)),
                   width, label="Prompt 2", color=PROMPT_COLORS["P2"], edgecolor="white")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Prompt 1 vs Prompt 2 — Average Score per Model", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean Total Score (out of 15)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, PERFECT_SCORE + 2)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "chart2_prompt_comparison.png")


# Chart 3: Scatter Plot — Model vs Prompt
def chart_scatter_model_prompt(df):
    print(" Generating Chart 3: Scatter Plot (Model vs Prompt) ")
    fig, ax = plt.subplots(figsize=(9, 6))

    prompt_map = {"P1": 1, "P2": 2}
    jitter_x = 0.07
    jitter_y = 0.15

    for model, group in df.groupby(COL_MODEL):
        color = MODEL_COLORS.get(model, "#888888")
        x_vals = group[COL_PROMPT].map(prompt_map) + np.random.uniform(-jitter_x, jitter_x, len(group))
        y_vals = group[COL_TOTAL] + np.random.uniform(-jitter_y, jitter_y, len(group))
        ax.scatter(x_vals, y_vals, label=model, color=color, alpha=0.75, s=70, edgecolors="white", linewidth=0.5)

    # Add mean lines per model per prompt
    for model, group in df.groupby(COL_MODEL):
        color = MODEL_COLORS.get(model, "#888888")
        means = group.groupby(COL_PROMPT)[COL_TOTAL].mean()
        if "P1" in means and "P2" in means:
            ax.plot([1, 2], [means["P1"], means["P2"]], color=color,
                    linewidth=2, linestyle="--", alpha=0.8)

    ax.set_title("Total Score by Model and Prompt", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Prompt", fontsize=12)
    ax.set_ylabel("Total Score (out of 15)", fontsize=12)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Prompt 1", "Prompt 2"], fontsize=11)
    ax.set_ylim(-1, PERFECT_SCORE + 1)
    ax.legend(title="Model", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "chart3_scatter_model_prompt.png")


# Chart 4: Spider Charts per Model
def chart_spider(df):
    print(" Generating Chart 4: Spider Charts per Model ")
    models = df[COL_MODEL].unique()
    n_metrics = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5),
                             subplot_kw=dict(polar=True))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        values = df[df[COL_MODEL] == model][METRICS].mean().tolist()
        values += values[:1]  # close the loop
        color = MODEL_COLORS.get(model, "#888888")

        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(METRIC_LABELS, fontsize=9)
        ax.set_ylim(0, 3)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["1", "2", "3"], fontsize=7, color="gray")
        ax.set_title(model, fontsize=13, fontweight="bold", pad=14)
        ax.spines["polar"].set_visible(True)

    fig.suptitle("Metric Breakdown per Model (Spider Chart)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "chart4_spider_charts.png")


# Chart 5: Stacked Bar Chart — Error Types per Model
def chart_stacked_errors(df, error_df):
    print(" Generating Chart 5: Stacked Bar Chart (Error Types per Model) ")
    models = error_df.index.tolist()
    metric_cols = error_df.columns.tolist()
    x = np.arange(len(models))

    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(metric_cols))]

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(models))

    for i, (metric, color) in enumerate(zip(metric_cols, colors)):
        vals = error_df[metric].values
        bars = ax.bar(x, vals, bottom=bottom, label=metric, color=color, edgecolor="white", linewidth=0.8)

        # Label inside each segment if tall enough
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottom[j] + val / 2,
                        str(int(val)), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        bottom += vals

    ax.set_title("Error Frequency per Model\n(Score ≤ 1 = Failure)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Number of Failures", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "chart5_stacked_errors.png")



# Main function to run all analyses and generate charts
def main():

    df = load_data()

    # Run all analyses
    model_stats, metric_means = analyze_model_performance(df)
    analyze_prompt_comparison(df)
    analyze_success_rate(df)
    error_df = analyze_errors(df)
    analyze_issue_type(df)
    analyze_violations(df)

    # Generate all charts
    print("\n" + "=" * 55)
    print("GENERATING CHARTS")
    print("=" * 55)
    chart_model_comparison(df)
    chart_prompt_comparison(df)
    chart_scatter_model_prompt(df)
    chart_spider(df)
    chart_stacked_errors(df, error_df)

    print("\nAll done! Charts saved to the '{}' folder.\n".format(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()