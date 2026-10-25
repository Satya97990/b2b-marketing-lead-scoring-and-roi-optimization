"""
Module: visualization.py
Description:
    Generates 5 executive-ready static analytical charts saved in `reports/figures/`
    for inclusion in GitHub README, slide decks, and management reports.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean corporate styling (IBM-inspired blue/slate aesthetic)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

IBM_PALETTE = ["#0F62FE", "#393939", "#009D9A", "#8A3FFC", "#EE5396", "#FF832B", "#24A148"]


def generate_all_figures(
    scored_df: pd.DataFrame,
    training_results: dict,
    optimization_results: dict,
    channel_summary: pd.DataFrame,
    output_dir: str = "reports/figures"
):
    """Generates and saves all 5 visual analytics figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Conversion Rate by Acquisition Channel & Buyer Persona
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
    ch_order = scored_df.groupby("Lead_Origin_Channel")["Converted"].mean().sort_values(ascending=True).index
    ch_means = scored_df.groupby("Lead_Origin_Channel")["Converted"].mean().loc[ch_order] * 100

    axes[0].barh(ch_order, ch_means, color="#0F62FE", alpha=0.9, height=0.6)
    for i, v in enumerate(ch_means):
        axes[0].text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold", color="#161616")
    axes[0].set_title("Lead Conversion Rate by Marketing Channel", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Conversion Rate (%)")
    axes[0].set_xlim(0, max(ch_means) * 1.18)
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    bp_order = scored_df.groupby("Buyer_Persona")["Converted"].mean().sort_values(ascending=True).index
    bp_means = scored_df.groupby("Buyer_Persona")["Converted"].mean().loc[bp_order] * 100
    axes[1].barh(bp_order, bp_means, color="#009D9A", alpha=0.9, height=0.6)
    for i, v in enumerate(bp_means):
        axes[1].text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold", color="#161616")
    axes[1].set_title("Lead Conversion Rate by Buyer Group Persona", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Conversion Rate (%)")
    axes[1].set_xlim(0, max(bp_means) * 1.18)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_conversion_funnel_and_channels.png"), bbox_inches="tight")
    plt.close()

    # Figure 2: ROC Curves & Precision-Recall Curves for Candidate Models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    colors = {"Logistic Regression": "#8A3FFC", "Random Forest": "#009D9A", "Gradient Boosting (HistGB)": "#0F62FE"}
    for name, cdata in training_results["curves_data"].items():
        auc_val = training_results["metrics_summary"][name]["ROC_AUC"]
        pr_val = training_results["metrics_summary"][name]["PR_AUC"]
        axes[0].plot(cdata["fpr"], cdata["tpr"], label=f"{name} (AUC = {auc_val:.3f})", color=colors[name], linewidth=2.2)
        axes[1].plot(cdata["rec_curve"], cdata["prec_curve"], label=f"{name} (PR-AUC = {pr_val:.3f})", color=colors[name], linewidth=2.2)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Random Baseline (0.500)")
    axes[0].set_title("ROC Curve Comparison (Test Set)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(linestyle="--", alpha=0.4)

    axes[1].set_title("Precision-Recall Curve Comparison (Test Set)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Proportion of True Buyers Captured)")
    axes[1].set_ylabel("Precision (Hit Rate of Sales Outreach)")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_model_roc_and_pr_curves.png"), bbox_inches="tight")
    plt.close()

    # Figure 3: Permutation Feature Importance (Top 12 Drivers of Lead Conversion)
    fi_df = training_results["feature_importance_df"].head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.barh(fi_df["Feature"], fi_df["Importance_Mean"], xerr=fi_df["Importance_Std"], color="#0F62FE", alpha=0.88, capsize=3)
    ax.set_title("Top 12 Predictive Drivers of Lead Conversion (Permutation Importance)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in ROC-AUC when Feature is Permuted")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_feature_importance_drivers.png"), bbox_inches="tight")
    plt.close()

    # Figure 4: Lead Score Distribution & Profit-Maximizing Threshold Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    sns.kdeplot(
        data=scored_df[scored_df["Converted"] == 0], x="Lead_Score", fill=True,
        color="#8D8D8D", alpha=0.35, label="Non-Converted Leads (0)", ax=axes[0]
    )
    sns.kdeplot(
        data=scored_df[scored_df["Converted"] == 1], x="Lead_Score", fill=True,
        color="#0F62FE", alpha=0.45, label="Converted Buyers (1)", ax=axes[0]
    )
    opt_score = optimization_results["optimal_threshold"] * 100
    axes[0].axvline(opt_score, color="#DA1E28", linestyle="--", linewidth=2, label=f"Optimal Cutoff ({opt_score:.0f})")
    axes[0].set_title("Lead Score Density by Actual Conversion Outcome", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predictive Lead Score (0 - 100)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True)
    axes[0].grid(linestyle="--", alpha=0.4)

    t_curve = optimization_results["threshold_curve"]
    axes[1].plot(t_curve["Threshold"] * 100, t_curve["Net_Profit_USD"] / 1e6, color="#009D9A", linewidth=2.5)
    axes[1].axvline(opt_score, color="#DA1E28", linestyle="--", linewidth=2, label=f"Max Profit Score = {opt_score:.0f}")
    axes[1].set_title("Expected Net Profit vs. Lead Score Threshold", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Minimum Lead Score Threshold for SDR Sales Outreach")
    axes[1].set_ylabel("Expected Net Profit ($ Millions USD)")
    axes[1].legend(frameon=True)
    axes[1].grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_lead_score_distribution_and_tiers.png"), bbox_inches="tight")
    plt.close()

    # Figure 5: Paid Media Budget Reallocation & Projected Pipeline Lift
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=180)
    ch_plot = channel_summary.sort_values(by="Historical_ROAS", ascending=True)
    y_pos = np.arange(len(ch_plot))
    bar_h = 0.36

    axes[0].barh(y_pos - bar_h/2, ch_plot["Total_Current_Spend_USD"] / 1e3, height=bar_h, label="Current Spend ($K)", color="#8D8D8D")
    axes[0].barh(y_pos + bar_h/2, ch_plot["Optimized_Budget_USD"] / 1e3, height=bar_h, label="Optimized Spend ($K)", color="#0F62FE")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ch_plot["Lead_Origin_Channel"])
    axes[0].set_xlabel("Marketing Spend ($ Thousands USD)")
    axes[0].set_title("Current vs. AI-Optimized Channel Budget Allocation", fontsize=12, fontweight="bold")
    axes[0].legend(frameon=True)
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    colors_delta = ["#24A148" if x >= 0 else "#DA1E28" for x in ch_plot["Budget_Delta_Pct"]]
    axes[1].barh(y_pos, ch_plot["Budget_Delta_Pct"], color=colors_delta, height=0.55)
    axes[1].axvline(0, color="#393939", linewidth=1)
    for i, v in enumerate(ch_plot["Budget_Delta_Pct"]):
        offset = 1.0 if v >= 0 else -5.5
        axes[1].text(v + offset, i, f"{v:+.1f}%", va="center", fontsize=9, fontweight="bold")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ch_plot["Lead_Origin_Channel"])
    axes[1].set_xlabel("Recommended Budget Shift (%)")
    axes[1].set_title("Prescribed Budget Reallocation by Channel ROAS", fontsize=12, fontweight="bold")
    axes[1].set_xlim(min(ch_plot["Budget_Delta_Pct"]) - 10, max(ch_plot["Budget_Delta_Pct"]) + 10)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_budget_optimization_roi_simulation.png"), bbox_inches="tight")
    plt.close()
    print(f"[INFO] All 5 analytical figures saved to -> {output_dir}/")
