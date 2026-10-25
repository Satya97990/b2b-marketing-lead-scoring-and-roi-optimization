"""
Module: preprocessing.py
Description:
    Handles end-to-end data cleaning, missing value treatment, CRM placeholder ("Select")
    resolution, outlier capping, and Marketing Domain Feature Engineering.
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_and_engineer_features(df: pd.DataFrame, output_path: str = "data/processed/b2b_leads_cleaned_featured.csv") -> pd.DataFrame:
    """
    Cleans raw CRM/Web data and creates domain-specific marketing analytics features.

    Key Steps:
    1. Replace CRM placeholder strings ('Select') and NaNs in categorical columns with 'Unknown / Unspecified'.
    2. Impute missing numerical web metrics (Total_Web_Visits, Page_Views_Per_Visit) using median values.
    3. Cap extreme right-tail outliers at the 99th percentile (Winsorization) to stabilize linear models.
    4. Engineer high-signal B2B marketing features:
       - Avg_Time_Per_Visit_Sec: Measures depth of engagement per session.
       - High_Intent_Action_Count: Composite score of whitepaper download, pricing page visit, and demo request.
       - Digital_Engagement_Index: Weighted composite index combining web time, visits, and email engagement.
       - Expected_Value_To_CAC_Ratio: Ratio of potential deal size to acquisition cost.

    Parameters:
        df (pd.DataFrame): Raw marketing dataframe.
        output_path (str): Path to save the cleaned and feature-engineered dataset.

    Returns:
        pd.DataFrame: Cleaned and feature-engineered DataFrame.
    """
    data = df.copy()

    # 1. Handle CRM 'Select' placeholders and missing categorical values
    categorical_cols = ["Lead_Origin_Channel", "Buyer_Persona", "Company_Size", "Industry_Vertical", "Last_Notable_Activity"]
    for col in categorical_cols:
        data[col] = data[col].replace("Select", np.nan)
        data[col] = data[col].fillna("Unknown / Unspecified")

    # 2. Impute missing numerical values with median
    numerical_impute_cols = ["Total_Web_Visits", "Page_Views_Per_Visit", "Total_Time_Spent_Sec", "Email_Open_Rate"]
    for col in numerical_impute_cols:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)

    # 3. Winsorize (cap) extreme outliers at 99th percentile for web session metrics
    for col in ["Total_Web_Visits", "Page_Views_Per_Visit", "Total_Time_Spent_Sec"]:
        upper_cap = data[col].quantile(0.99)
        data[col] = np.clip(data[col], a_min=0.0, a_max=upper_cap)

    # 4. Feature Engineering (Marketing Domain Signals)
    # 4a. Average Time Per Visit (in seconds)
    data["Avg_Time_Per_Visit_Sec"] = np.where(
        data["Total_Web_Visits"] > 0,
        (data["Total_Time_Spent_Sec"] / data["Total_Web_Visits"]).round(2),
        0.0
    )

    # 4b. High-Intent Action Count (0 to 3 scale)
    data["High_Intent_Action_Count"] = (
        data["Downloaded_Whitepaper"]
        + data["Pricing_Page_Visited"]
        + data["Requested_Product_Demo"]
    )

    # 4c. Digital Engagement Index (Normalized composite engagement score)
    norm_time = data["Total_Time_Spent_Sec"] / (data["Total_Time_Spent_Sec"].max() + 1e-5)
    norm_visits = data["Total_Web_Visits"] / (data["Total_Web_Visits"].max() + 1e-5)
    data["Digital_Engagement_Index"] = (
        0.45 * norm_time
        + 0.25 * norm_visits
        + 0.30 * data["Email_Open_Rate"]
    ).round(4)

    # 4d. Deal Value to Acquisition Cost Ratio
    data["Deal_Value_to_CAC_Ratio"] = (
        data["Estimated_Deal_Value_USD"] / np.maximum(data["Acquisition_Cost_USD"], 1.0)
    ).round(2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"[INFO] Cleaned & feature-engineered dataset saved -> {output_path}")
    return data


def build_sklearn_preprocessor(categorical_features: list, numerical_features: list) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer pipeline that standardizes numerical features
    and one-hot encodes categorical variables safely for production inference.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown / Unspecified")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor
