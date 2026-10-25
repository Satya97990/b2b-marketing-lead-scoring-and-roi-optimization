"""
Module: data_generator.py
Description:
    Generates a realistic, enterprise-scale B2B Marketing & Web Demand Lead Dataset (10,000 records)
    modeled after the public Kaggle "Lead Scoring Dataset" (X Education) and enterprise B2B SaaS/Cloud
    marketing funnels (such as IBM Hybrid Cloud & AI buyer journeys).

    NOTE ON EXTERNAL DATA:
    If you download the public Kaggle Lead Scoring dataset (`Lead Scoring.csv` from
    https://www.kaggle.com/datasets/ashydv/leads-dataset), you can place it in `data/raw/`
    and adapt the column mapping dictionary below. By default, this script synthesizes
    a rich, realistic 10,000-row dataset complete with real-world imperfections (missing values,
    "Select" placeholder strings, skewed web session metrics, and multi-channel attribution costs)
    so the repository runs 100% out-of-the-box.
"""

import os
import numpy as np
import pandas as pd


def generate_b2b_marketing_dataset(n_samples: int = 10000, random_state: int = 42, output_path: str = "data/raw/b2b_marketing_leads_raw.csv") -> pd.DataFrame:
    """
    Synthesizes a realistic B2B marketing lead dataset spanning three core pillars:
    1. Early-stage Paid Media & Acquisition Channel attribution
    2. Web Presence & Digital Demand Capture metrics
    3. Buyer Group Demographics, Firmographics & CRM Lead Management signals
    """
    rng = np.random.default_rng(random_state)

    # 1. Unique Lead Identifiers
    lead_ids = [f"LD-{100000 + i}" for i in range(n_samples)]

    # 2. Acquisition Channels (Early-Stage Audience Targeting)
    channels = [
        "Paid Search (SEM)",
        "Paid Social (LinkedIn Ads)",
        "Organic Search (SEO)",
        "Direct Web Traffic",
        "Webinar & Virtual Events",
        "Partner & Affiliate Referral",
        "Email Marketing Campaign"
    ]
    channel_probs = [0.24, 0.20, 0.22, 0.12, 0.10, 0.05, 0.07]
    lead_origin = rng.choice(channels, size=n_samples, p=channel_probs)

    # Average Customer Acquisition Cost (CAC / Ad Spend per Lead) by channel in USD
    channel_base_cost = {
        "Paid Search (SEM)": 95.0,
        "Paid Social (LinkedIn Ads)": 125.0,
        "Organic Search (SEO)": 30.0,
        "Direct Web Traffic": 20.0,
        "Webinar & Virtual Events": 80.0,
        "Partner & Affiliate Referral": 60.0,
        "Email Marketing Campaign": 25.0
    }
    acquisition_cost = np.array([
        max(5.0, rng.normal(channel_base_cost[ch], channel_base_cost[ch] * 0.18))
        for ch in lead_origin
    ]).round(2)

    # 3. Buyer Group & Firmographic Attributes
    buyer_personas = [
        "C-Suite / Executive (CIO, CTO)",
        "VP / Director of IT & Analytics",
        "Data Scientist / AI Engineer",
        "Enterprise Architect",
        "Procurement / Operations Manager",
        "Student / Academic Researcher",
        "Select"  # Realistic CRM placeholder when user skips dropdown
    ]
    persona_probs = [0.08, 0.18, 0.26, 0.16, 0.12, 0.08, 0.12]
    buyer_group = rng.choice(buyer_personas, size=n_samples, p=persona_probs)

    company_sizes = [
        "Enterprise (5000+ Employees)",
        "Large Mid-Market (1000-4999)",
        "Mid-Market (250-999)",
        "SMB (1-249)",
        "Select"
    ]
    size_probs = [0.22, 0.25, 0.25, 0.18, 0.10]
    company_size = rng.choice(company_sizes, size=n_samples, p=size_probs)

    target_industries = [
        "BFSI (Banking & Financial Services)",
        "Technology & Cloud Software",
        "Healthcare & Life Sciences",
        "Manufacturing & Automotive",
        "Retail & E-Commerce",
        "Telecommunications"
    ]
    industry = rng.choice(target_industries, size=n_samples, p=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10])

    # 4. Web Presence & Demand Capture Behavioral Metrics
    total_visits = rng.poisson(lam=4.0, size=n_samples) + rng.choice([0, 2, 5, 10], size=n_samples, p=[0.55, 0.25, 0.15, 0.05])
    base_time = rng.gamma(shape=2.0, scale=220.0, size=n_samples)
    total_time_spent_sec = np.clip(base_time + total_visits * 30.0, 0, 4800).round(0)

    page_views_per_visit = np.where(
        total_visits > 0,
        np.clip(rng.normal(loc=2.6, scale=1.1, size=n_samples), 1.0, 14.0).round(2),
        0.0
    )

    # High-intent content interactions (Binary signals: 1 = Yes, 0 = No)
    downloaded_whitepaper = rng.binomial(1, p=np.clip(0.12 + 0.025 * total_visits, 0.04, 0.65))
    requested_product_demo = rng.binomial(1, p=np.clip(0.04 + (total_time_spent_sec / 6500.0) * 0.30, 0.01, 0.55))
    pricing_page_visited = rng.binomial(1, p=np.clip(0.10 + 0.035 * page_views_per_visit, 0.04, 0.65))
    email_opened_rate = np.clip(rng.beta(a=2.2, b=3.8, size=n_samples), 0.0, 1.0).round(3)

    # Last Notable Activity in CRM
    activities = [
        "Product Demo Requested",
        "Pricing Page Visited",
        "Whitepaper Downloaded",
        "Email Link Clicked",
        "Email Opened",
        "Page Visited on Website",
        "Unreachable / Bounced"
    ]
    last_activity = []
    for i in range(n_samples):
        if requested_product_demo[i] == 1 and rng.random() < 0.72:
            last_activity.append("Product Demo Requested")
        elif pricing_page_visited[i] == 1 and rng.random() < 0.50:
            last_activity.append("Pricing Page Visited")
        elif downloaded_whitepaper[i] == 1 and rng.random() < 0.45:
            last_activity.append("Whitepaper Downloaded")
        else:
            last_activity.append(rng.choice(activities[3:], p=[0.25, 0.35, 0.25, 0.15]))
    last_activity = np.array(last_activity)

    # 5. Construct True Underlying Conversion Log-Odds (Realistic ~28% B2B Conversion Rate + Non-Linear Interactions)
    logit = np.full(n_samples, -3.55)

    # Channel effects
    channel_weights = {
        "Webinar & Virtual Events": 0.70,
        "Partner & Affiliate Referral": 0.65,
        "Organic Search (SEO)": 0.35,
        "Paid Social (LinkedIn Ads)": 0.30,
        "Paid Search (SEM)": 0.20,
        "Direct Web Traffic": 0.15,
        "Email Marketing Campaign": -0.25
    }
    logit += np.array([channel_weights[ch] for ch in lead_origin])

    # Buyer Persona effects
    persona_weights = {
        "C-Suite / Executive (CIO, CTO)": 0.90,
        "VP / Director of IT & Analytics": 0.80,
        "Enterprise Architect": 0.45,
        "Data Scientist / AI Engineer": 0.20,
        "Procurement / Operations Manager": 0.10,
        "Student / Academic Researcher": -1.60,
        "Select": -0.75
    }
    logit += np.array([persona_weights[p] for p in buyer_group])

    # Company Size effects
    size_weights = {
        "Enterprise (5000+ Employees)": 0.65,
        "Large Mid-Market (1000-4999)": 0.45,
        "Mid-Market (250-999)": 0.15,
        "SMB (1-249)": -0.35,
        "Select": -0.50
    }
    logit += np.array([size_weights[s] for s in company_size])

    # Non-linear behavioral threshold & interaction effects (where tree ensembles excel over linear models)
    # 1) Sweet-spot dwell time threshold (> 600s and > 3 visits triggers strong buying signal)
    deep_session_signal = np.where((total_time_spent_sec > 600) & (total_visits >= 4), 1.45, -0.35)
    logit += deep_session_signal + 0.0006 * total_time_spent_sec

    # 2) High-intent synergy: Demo Request + Pricing Page or High-Level Persona
    senior_decision_maker = np.isin(buyer_group, ["C-Suite / Executive (CIO, CTO)", "VP / Director of IT & Analytics", "Enterprise Architect"])
    enterprise_org = np.isin(company_size, ["Enterprise (5000+ Employees)", "Large Mid-Market (1000-4999)"])

    logit += 1.25 * requested_product_demo
    logit += 0.65 * pricing_page_visited
    logit += 0.40 * downloaded_whitepaper
    logit += 0.85 * (senior_decision_maker & (pricing_page_visited == 1)).astype(float)
    logit += 0.95 * (enterprise_org & (requested_product_demo == 1)).astype(float)
    logit += 0.70 * ((email_opened_rate > 0.45) & (downloaded_whitepaper == 1)).astype(float)
    logit -= 1.20 * (last_activity == "Unreachable / Bounced").astype(float)

    # Add moderate idiosyncratic noise
    logit += rng.normal(0, 0.55, size=n_samples)

    conversion_prob = 1.0 / (1.0 + np.exp(-logit))
    converted = rng.binomial(1, p=conversion_prob)

    # Estimated Deal Value (ARR in USD) if converted, based on company size
    deal_base = {
        "Enterprise (5000+ Employees)": 24000,
        "Large Mid-Market (1000-4999)": 16000,
        "Mid-Market (250-999)": 10000,
        "SMB (1-249)": 5000,
        "Select": 8000
    }
    estimated_deal_value = np.array([
        max(2000.0, rng.normal(deal_base[s], deal_base[s] * 0.15))
        for s in company_size
    ]).round(0)

    df = pd.DataFrame({
        "Lead_ID": lead_ids,
        "Lead_Origin_Channel": lead_origin,
        "Acquisition_Cost_USD": acquisition_cost,
        "Buyer_Persona": buyer_group,
        "Company_Size": company_size,
        "Industry_Vertical": industry,
        "Total_Web_Visits": total_visits.astype(float),
        "Total_Time_Spent_Sec": total_time_spent_sec,
        "Page_Views_Per_Visit": page_views_per_visit,
        "Downloaded_Whitepaper": downloaded_whitepaper,
        "Pricing_Page_Visited": pricing_page_visited,
        "Requested_Product_Demo": requested_product_demo,
        "Email_Open_Rate": email_opened_rate,
        "Last_Notable_Activity": last_activity,
        "Estimated_Deal_Value_USD": estimated_deal_value,
        "Converted": converted
    })

    # Inject realistic missing values (NaNs) to demonstrate data cleaning in preprocessing.py
    missing_visits_idx = rng.choice(n_samples, size=int(0.025 * n_samples), replace=False)
    missing_pageviews_idx = rng.choice(n_samples, size=int(0.03 * n_samples), replace=False)
    missing_persona_idx = rng.choice(n_samples, size=int(0.04 * n_samples), replace=False)

    df.loc[missing_visits_idx, "Total_Web_Visits"] = np.nan
    df.loc[missing_pageviews_idx, "Page_Views_Per_Visit"] = np.nan
    df.loc[missing_persona_idx, "Buyer_Persona"] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Generated {n_samples:,} B2B marketing lead records -> {output_path}")
    print(f"[INFO] Overall Baseline Conversion Rate: {df['Converted'].mean() * 100:.2f}%")
    return df


if __name__ == "__main__":
    generate_b2b_marketing_dataset()
