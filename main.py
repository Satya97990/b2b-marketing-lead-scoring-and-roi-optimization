"""
Script: main.py
Description:
    Master orchestrator script that executes the complete B2B Marketing Lead Scoring &
    Budget Optimization pipeline from start to end:
    1. Generates (or loads) raw B2B marketing lead data (`data/raw/b2b_marketing_leads_raw.csv`).
    2. Cleans dirty records and engineers marketing domain features (`data/processed/`).
    3. Trains & benchmarks Logistic Regression, Random Forest, and HistGradientBoosting models (`models/`).
    4. Scores leads, segments buyers into tiers, optimizes outreach threshold, and reallocates ad spend (`data/bi_exports/`).
    5. Produces 5 executive charts in `reports/figures/`.
"""

from src.data_generator import generate_b2b_marketing_dataset
from src.preprocessing import clean_and_engineer_features
from src.model_training import train_and_evaluate_models, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.marketing_optimization import score_and_segment_leads, optimize_marketing_channel_budget
from src.visualization import generate_all_figures


def main():
    print("=" * 80)
    print("B2B MARKETING LEAD SCORING & PAID MEDIA ROI OPTIMIZATION PIPELINE")
    print("=" * 80)

    # Step 1: Data Acquisition / Generation
    raw_df = generate_b2b_marketing_dataset(n_samples=10000, random_state=42)

    # Step 2: Data Cleaning & Domain Feature Engineering
    clean_df = clean_and_engineer_features(raw_df)

    # Step 3: Predictive Modeling & Evaluation
    training_results = train_and_evaluate_models(clean_df)

    # Step 4: Lead Segmentation, Optimal Thresholding & Budget Optimization
    all_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    scored_df, optimization_results = score_and_segment_leads(
        clean_df,
        pipeline=training_results["best_pipeline"],
        feature_cols=all_features
    )
    channel_summary = optimize_marketing_channel_budget(scored_df)

    # Step 5: Generate Static Dashboard Figures
    generate_all_figures(
        scored_df=scored_df,
        training_results=training_results,
        optimization_results=optimization_results,
        channel_summary=channel_summary
    )

    cur_pipe = channel_summary["Total_Expected_Pipeline_USD"].sum()
    opt_pipe = channel_summary["Projected_Optimized_Pipeline_USD"].sum()
    lift_pct = ((opt_pipe - cur_pipe) / cur_pipe) * 100.0

    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE - EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"Winning Model               : {training_results['best_model_name']}")
    print(f"Test ROC-AUC Score          : {training_results['metrics_summary'][training_results['best_model_name']]['ROC_AUC']:.4f}")
    print(f"Optimal Outreach Cutoff     : Lead Score >= {optimization_results['optimal_threshold']*100:.0f}")
    print(f"Max Expected Net Profit     : ${optimization_results['max_net_profit_usd']:,.2f}")
    print(f"Projected Pipeline ROAS Lift: +{lift_pct:.2f}% (${opt_pipe - cur_pipe:,.2f} incremental pipeline)")
    print("=" * 80)


if __name__ == "__main__":
    main()
