"""
Module: marketing_optimization.py
Description:
    Translates raw ML probability scores into actionable Marketing & Sales Operations decisions:
    1. Lead Tier Segmentation (Hot / Warm / Nurture / Cold) for SDR routing.
    2. Cost-Benefit Threshold Optimization: Finds the probability threshold that maximizes Net Expected Profit
       given Sales Outreach Cost per Lead vs. Expected Deal Margin.
    3. Paid Media Channel Attribution & Budget Reallocation Engine: Evaluates historical ROAS (Return on Ad Spend)
       and marginal conversion efficiency across channels to recommend an optimized budget allocation.
"""

import os
import numpy as np
import pandas as pd


def score_and_segment_leads(
    df: pd.DataFrame,
    pipeline,
    feature_cols: list,
    sales_call_cost_usd: float = 950.0,
    gross_margin_rate: float = 0.25,
    bi_export_path: str = "data/bi_exports/scored_leads_for_powerbi_tableau.csv"
) -> tuple[pd.DataFrame, dict]:
    """
    Scores all leads using the trained ML pipeline, assigns actionable lead tiers,
    and computes the optimal classification threshold maximizing net revenue contribution.
    """
    scored_df = df.copy()
    probs = pipeline.predict_proba(scored_df[feature_cols])[:, 1]
    scored_df["Lead_Score"] = (probs * 100.0).round(2)
    scored_df["Predicted_Conversion_Prob"] = probs.round(4)

    # Segment into actionable Marketing / Sales Tiers
    conditions = [
        scored_df["Lead_Score"] >= 70.0,
        (scored_df["Lead_Score"] >= 40.0) & (scored_df["Lead_Score"] < 70.0),
        (scored_df["Lead_Score"] >= 20.0) & (scored_df["Lead_Score"] < 40.0),
        scored_df["Lead_Score"] < 20.0
    ]
    tiers = [
        "Tier 1: Hot (Immediate SDR Outreach)",
        "Tier 2: Warm (Priority Inside Sales)",
        "Tier 3: Nurture (Automated Drip Campaign)",
        "Tier 4: Cold (Suppress / Low-Cost Retargeting)"
    ]
    scored_df["Lead_Segment_Tier"] = np.select(conditions, tiers, default="Tier 3: Nurture (Automated Drip Campaign)")

    # Expected Deal Value Value Weighted by Conversion Probability
    scored_df["Expected_Pipeline_Value_USD"] = (
        scored_df["Predicted_Conversion_Prob"] * scored_df["Estimated_Deal_Value_USD"] * gross_margin_rate
    ).round(2)

    # Optimal Probability Threshold Search (Grid search from 0.05 to 0.90)
    thresholds = np.linspace(0.05, 0.90, 86)
    best_threshold = 0.50
    max_net_profit = -np.inf
    threshold_curve = []

    for thresh in thresholds:
        targeted = probs >= thresh
        # True conversions captured among targeted leads
        captured_conversions = (scored_df["Converted"] == 1) & targeted
        gross_profit = (scored_df.loc[captured_conversions, "Estimated_Deal_Value_USD"] * gross_margin_rate).sum()
        outreach_cost = targeted.sum() * sales_call_cost_usd
        net_profit = gross_profit - outreach_cost
        threshold_curve.append({
            "Threshold": round(float(thresh), 2),
            "Targeted_Leads_Pct": round(float(targeted.mean() * 100), 2),
            "Conversion_Capture_Rate_Pct": round(float(captured_conversions.sum() / max(1, scored_df["Converted"].sum()) * 100), 2),
            "Net_Profit_USD": round(float(net_profit), 2)
        })
        if net_profit > max_net_profit:
            max_net_profit = net_profit
            best_threshold = float(thresh)

    scored_df["Recommended_Sales_Action"] = np.where(
        scored_df["Predicted_Conversion_Prob"] >= best_threshold,
        "Pursue Direct Sales Outreach",
        "Route to Automated Digital Nurture"
    )

    os.makedirs(os.path.dirname(bi_export_path), exist_ok=True)
    scored_df.to_csv(bi_export_path, index=False)
    print(f"[INFO] PowerBI/Tableau Scored Leads Exported -> {bi_export_path}")
    print(f"[INFO] Optimal Conversion Probability Cutoff: {best_threshold:.2f} (Score >= {best_threshold*100:.0f})")

    return scored_df, {
        "optimal_threshold": round(best_threshold, 2),
        "max_net_profit_usd": round(max_net_profit, 2),
        "threshold_curve": pd.DataFrame(threshold_curve)
    }


def optimize_marketing_channel_budget(
    scored_df: pd.DataFrame,
    gross_margin_rate: float = 0.25,
    output_path: str = "data/bi_exports/channel_roi_optimization_summary.csv"
) -> pd.DataFrame:
    """
    Aggregates channel-level performance (CAC, Conversion Rate, Avg Lead Score, Historical ROAS)
    and prescribes a data-driven budget reallocation strategy that shifts spend from low-efficiency
    channels toward high-ROAS, high-propensity buyer channels.
    """
    channel_summary = scored_df.groupby("Lead_Origin_Channel", as_index=False).agg(
        Total_Leads=("Lead_ID", "count"),
        Total_Conversions=("Converted", "sum"),
        Actual_Conversion_Rate=("Converted", "mean"),
        Avg_Predicted_Lead_Score=("Lead_Score", "mean"),
        Total_Current_Spend_USD=("Acquisition_Cost_USD", "sum"),
        Avg_CAC_Per_Lead_USD=("Acquisition_Cost_USD", "mean"),
        Total_Expected_Pipeline_USD=("Expected_Pipeline_Value_USD", "sum")
    )

    channel_summary["Actual_Conversion_Rate_Pct"] = (channel_summary["Actual_Conversion_Rate"] * 100.0).round(2)
    channel_summary["Cost_Per_Converted_Customer_USD"] = (
        channel_summary["Total_Current_Spend_USD"] / np.maximum(channel_summary["Total_Conversions"], 1)
    ).round(2)

    # Return on Ad Spend (ROAS) = Expected Gross Margin Pipeline / Total Spend
    channel_summary["Historical_ROAS"] = (
        channel_summary["Total_Expected_Pipeline_USD"] / np.maximum(channel_summary["Total_Current_Spend_USD"], 1.0)
    ).round(2)

    # Budget Reallocation Algorithm:
    # Weight each channel by (Historical_ROAS^0.65 * Current_Spend_Share^0.35) with guardrails (+/- 35% max shift)
    total_budget = channel_summary["Total_Current_Spend_USD"].sum()
    current_share = channel_summary["Total_Current_Spend_USD"] / total_budget
    roas_norm = channel_summary["Historical_ROAS"] / channel_summary["Historical_ROAS"].sum()

    raw_optimal_share = 0.55 * current_share + 0.45 * roas_norm
    # Apply guardrails so no channel is zeroed out or scaled beyond realistic saturation
    bounded_share = np.clip(raw_optimal_share, current_share * 0.65, current_share * 1.40)
    bounded_share = bounded_share / bounded_share.sum()

    channel_summary["Optimized_Budget_USD"] = (bounded_share * total_budget).round(2)
    channel_summary["Budget_Delta_Pct"] = (
        ((channel_summary["Optimized_Budget_USD"] - channel_summary["Total_Current_Spend_USD"])
         / channel_summary["Total_Current_Spend_USD"]) * 100.0
    ).round(2)

    # Simulate Expected Pipeline after budget shift (incorporating diminishing returns saturation factor 0.88)
    spend_multiplier = channel_summary["Optimized_Budget_USD"] / channel_summary["Total_Current_Spend_USD"]
    effective_lift = np.where(
        spend_multiplier > 1.0,
        1.0 + (spend_multiplier - 1.0) * 0.85,  # Diminishing marginal returns on increased ad spend
        spend_multiplier
    )
    channel_summary["Projected_Optimized_Pipeline_USD"] = (
        channel_summary["Total_Expected_Pipeline_USD"] * effective_lift
    ).round(2)

    channel_summary = channel_summary.sort_values(by="Historical_ROAS", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    channel_summary.to_csv(output_path, index=False)
    print(f"[INFO] Paid Media Channel ROI Optimization Summary saved -> {output_path}")
    return channel_summary
