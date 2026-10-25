"""
Application: app.py
Description:
    Interactive Streamlit Executive Dashboard for Marketing Data Science & Demand Optimization.
    Includes:
    - Tab 1: Executive Funnel & Buyer Group Analytics
    - Tab 2: Paid Media Attribution & AI Budget Reallocation Simulator
    - Tab 3: Real-Time Interactive B2B Lead Scoring Calculator (Live Model Inference)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="B2B Marketing Lead Scoring & ROI Optimizer",
    page_icon="📊",
    layout="wide"
)

# Ensure artifacts exist; if running fresh on cloud, run main pipeline automatically
if not os.path.exists("data/bi_exports/scored_leads_for_powerbi_tableau.csv"):
    from main import main as run_pipeline
    run_pipeline()


@st.cache_data
def load_data():
    scored_df = pd.read_csv("data/bi_exports/scored_leads_for_powerbi_tableau.csv")
    channel_df = pd.read_csv("data/bi_exports/channel_roi_optimization_summary.csv")
    with open("models/model_metrics.json", "r") as f:
        metrics = json.load(f)
    return scored_df, channel_df, metrics


@st.cache_resource
def load_model():
    return joblib.load("models/lead_scoring_pipeline.pkl")


scored_df, channel_df, metrics = load_data()
pipeline = load_model()

st.title("📊 B2B Marketing Lead Scoring & Paid Media ROI Optimization")
st.markdown(
    "**Enterprise Marketing Data Science Suite** — Orchestrating performance across paid media tactics, "
    "web demand capture, buyer groups, and sales lead prioritization."
)

# Top-level KPI Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
best_model_name = metrics["best_model"]
best_auc = metrics["models"][best_model_name]["ROC_AUC"]
total_leads = len(scored_df)
overall_conv = scored_df["Converted"].mean() * 100
hot_leads_count = (scored_df["Lead_Score"] >= 70).sum()
cur_pipe = channel_df["Total_Expected_Pipeline_USD"].sum()
opt_pipe = channel_df["Projected_Optimized_Pipeline_USD"].sum()
roas_lift = ((opt_pipe - cur_pipe) / cur_pipe) * 100

col1.metric("Total Marketing Leads", f"{total_leads:,}")
col2.metric("Baseline Conversion Rate", f"{overall_conv:.1f}%")
col3.metric(f"Model ROC-AUC ({best_model_name})", f"{best_auc:.3f}")
col4.metric("Tier 1 'Hot' Leads Identified", f"{hot_leads_count:,}")
col5.metric("Projected Budget Optimization Lift", f"+{roas_lift:.1f}%")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "🎯 Lead Tiers & Buyer Group Intelligence",
    "💰 Paid Media Attribution & Budget Optimizer",
    "⚡ Live Lead Scoring Simulator"
])

with tab1:
    st.subheader("Lead Segmentation & Model Performance Summary")
    c1, c2 = st.columns([1.2, 1])

    with c1:
        tier_summary = scored_df.groupby("Lead_Segment_Tier", as_index=False).agg(
            Leads_Count=("Lead_ID", "count"),
            Actual_Conversion_Rate_Pct=("Converted", lambda s: round(s.mean() * 100, 2)),
            Avg_Lead_Score=("Lead_Score", lambda s: round(s.mean(), 1)),
            Expected_Pipeline_USD=("Expected_Pipeline_Value_USD", lambda s: round(s.sum(), 0))
        ).sort_values(by="Avg_Lead_Score", ascending=False)
        st.dataframe(tier_summary, use_container_width=True, hide_index=True)

    with c2:
        model_comp_df = pd.DataFrame(metrics["models"]).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(model_comp_df, use_container_width=True, hide_index=True)

    st.markdown("### Visual Diagnostic Reports")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.image("reports/figures/02_model_roc_and_pr_curves.png", use_container_width=True)
    with r1c2:
        st.image("reports/figures/04_lead_score_distribution_and_tiers.png", use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.image("reports/figures/01_conversion_funnel_and_channels.png", use_container_width=True)
    with r2c2:
        st.image("reports/figures/03_feature_importance_drivers.png", use_container_width=True)

with tab2:
    st.subheader("Paid Media Channel Efficiency & Budget Reallocation")
    st.markdown(
        "Evaluates historical Customer Acquisition Cost (CAC) and Return on Ad Spend (ROAS) across tactics, "
        "prescribing an optimal budget shift with diminishing-return guardrails."
    )
    st.dataframe(
        channel_df[[
            "Lead_Origin_Channel", "Total_Leads", "Actual_Conversion_Rate_Pct",
            "Avg_CAC_Per_Lead_USD", "Historical_ROAS", "Total_Current_Spend_USD",
            "Optimized_Budget_USD", "Budget_Delta_Pct", "Projected_Optimized_Pipeline_USD"
        ]],
        use_container_width=True,
        hide_index=True
    )
    st.image("reports/figures/05_budget_optimization_roi_simulation.png", use_container_width=True)

with tab3:
    st.subheader("Interactive Real-Time B2B Lead Scoring Calculator")
    st.markdown("Adjust buyer group firmographics and digital engagement signals to score a prospect in real time.")

    f1, f2, f3 = st.columns(3)
    with f1:
        origin = st.selectbox("Acquisition Channel", sorted(scored_df["Lead_Origin_Channel"].unique()))
        persona = st.selectbox("Buyer Group Persona", sorted(scored_df["Buyer_Persona"].unique()))
        comp_size = st.selectbox("Company Size", sorted(scored_df["Company_Size"].unique()))
        industry = st.selectbox("Industry Vertical", sorted(scored_df["Industry_Vertical"].unique()))

    with f2:
        visits = st.slider("Total Website Visits", 0, 25, 6)
        time_sec = st.slider("Total Time on Website (Seconds)", 0, 3600, 750)
        page_views = st.slider("Page Views per Visit", 0.0, 12.0, 3.5, 0.5)
        email_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.45, 0.05)

    with f3:
        demo = st.checkbox("Requested Product Demo", value=True)
        pricing = st.checkbox("Visited Pricing Page", value=True)
        whitepaper = st.checkbox("Downloaded Technical Whitepaper", value=False)
        activity = st.selectbox("Last Notable CRM Activity", sorted(scored_df["Last_Notable_Activity"].unique()))
        deal_val = st.number_input("Estimated Deal Size (ARR in USD)", value=28000, step=1000)
        cac_val = st.number_input("Acquisition Cost (USD)", value=95, step=5)

    # Construct single-row feature DataFrame matching training schema
    avg_time = round(time_sec / visits, 2) if visits > 0 else 0.0
    intent_cnt = int(demo) + int(pricing) + int(whitepaper)
    eng_idx = round(0.45 * (time_sec / 3600.0) + 0.25 * (visits / 25.0) + 0.30 * email_rate, 4)
    val_cac = round(deal_val / max(cac_val, 1.0), 2)

    input_df = pd.DataFrame([{
        "Lead_Origin_Channel": origin,
        "Buyer_Persona": persona,
        "Company_Size": comp_size,
        "Industry_Vertical": industry,
        "Last_Notable_Activity": activity,
        "Acquisition_Cost_USD": float(cac_val),
        "Total_Web_Visits": float(visits),
        "Total_Time_Spent_Sec": float(time_sec),
        "Page_Views_Per_Visit": float(page_views),
        "Downloaded_Whitepaper": int(whitepaper),
        "Pricing_Page_Visited": int(pricing),
        "Requested_Product_Demo": int(demo),
        "Email_Open_Rate": float(email_rate),
        "Avg_Time_Per_Visit_Sec": avg_time,
        "High_Intent_Action_Count": intent_cnt,
        "Digital_Engagement_Index": eng_idx,
        "Deal_Value_to_CAC_Ratio": val_cac
    }])

    pred_prob = float(pipeline.predict_proba(input_df)[0, 1])
    lead_score = round(pred_prob * 100, 1)

    if lead_score >= 70:
        tier_label = "🔥 Tier 1: Hot Lead (Immediate SDR Outreach)"
    elif lead_score >= 40:
        tier_label = "⚡ Tier 2: Warm Lead (Priority Inside Sales)"
    elif lead_score >= 20:
        tier_label = "📩 Tier 3: Nurture Lead (Automated Drip Campaign)"
    else:
        tier_label = "❄️ Tier 4: Cold Lead (Low-Cost Retargeting)"

    st.markdown("---")
    r_col1, r_col2, r_col3 = st.columns(3)
    r_col1.metric("Predicted Lead Score (0-100)", f"{lead_score}")
    r_col2.metric("Conversion Probability", f"{pred_prob * 100:.1f}%")
    r_col3.metric("Recommended Routing Tier", tier_label)
