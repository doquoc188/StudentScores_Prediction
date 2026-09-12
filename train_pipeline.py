import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_preprocessing import prepare_data
from src.model_training import (
    evaluate_models_cv,
    tune_best_model,
    evaluate_on_test,
    save_model_artifacts
)
from src.evaluation import (
    plot_model_comparison,
    plot_residual_analysis,
    plot_feature_importance
)

def run_training_pipeline(data_path: str = "StudentScore.xls"):
    print("=" * 60)
    print("🚀 STARTING STUDENT SCORES PREDICTION TRAINING PIPELINE 🚀")
    print("=" * 60)
    
    # Step 1: Load and Split Data
    print("\n📂 Step 1: Loading dataset and creating Train/Test split...")
    X_train, X_test, y_train, y_test = prepare_data(file_path=data_path, target_col="math score")
    print(f"   Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Step 2: Benchmark Candidate Models
    print("\n📊 Step 2: Cross-Validating Candidate Machine Learning Models...")
    cv_results = evaluate_models_cv(X_train, y_train, cv=5)
    print("\n--- 🏆 Model Benchmarking Summary (5-Fold CV R²) ---")
    print(cv_results.to_string(index=False))
    
    best_model_name = cv_results.iloc[0]["Model"]
    best_cv_r2 = cv_results.iloc[0]["CV_R2_Mean"]
    print(f"\n🥇 Winning Baseline Model: {best_model_name} (CV R²: {best_cv_r2:.4f})")
    
    # Step 3: Hyperparameter Tuning
    print(f"\n🔧 Step 3: Hyperparameter Tuning for Best Model '{best_model_name}'...")
    best_pipeline = tune_best_model(X_train, y_train, best_model_name=best_model_name)
    
    # Step 4: Evaluate on Holdout Test Set
    print("\n🎯 Step 4: Evaluating Best Model on Holdout Test Set...")
    test_metrics = evaluate_on_test(best_pipeline, X_test, y_test)
    print("\n--- 📈 Holdout Test Set Metrics ---")
    for metric_name, val in test_metrics.items():
        print(f"   - {metric_name}: {val:.4f}")
        
    # Step 5: Generate & Save Plots
    print("\n🎨 Step 5: Generating Visualization Artifacts...")
    comp_plot = plot_model_comparison(cv_results)
    print(f"   - Saved: {comp_plot}")
    
    y_pred = best_pipeline.predict(X_test)
    res_plot = plot_residual_analysis(y_test, y_pred)
    print(f"   - Saved: {res_plot}")
    
    fi_plot = plot_feature_importance(best_pipeline)
    if fi_plot:
        print(f"   - Saved: {fi_plot}")
        
    # Step 6: Serialize Artifacts
    print("\n💾 Step 6: Saving Trained Model & Metrics Artifacts...")
    metrics_log = {
        "best_model_name": best_model_name,
        "cv_benchmark_summary": cv_results.to_dict(orient="records"),
        "test_set_metrics": test_metrics
    }
    model_file, json_file = save_model_artifacts(best_pipeline, metrics_log)
    print(f"   - Model Pipeline saved to: {model_file}")
    print(f"   - Metrics Summary saved to: {json_file}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY! ✅")
    print("=" * 60)

if __name__ == "__main__":
    run_training_pipeline()
