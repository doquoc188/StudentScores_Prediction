import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sns.set_theme(style="whitegrid")

def plot_model_comparison(results_df: pd.DataFrame, output_dir: str = "plots") -> str:
    """Generate and save model comparison bar plot."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot CV R2
    palette = sns.color_palette("viridis", len(results_df))
    bars = sns.barplot(
        data=results_df,
        x="Model",
        y="CV_R2_Mean",
        ax=ax1,
        hue="Model",
        palette=palette,
        legend=False
    )
    
    ax1.set_title("Model Comparison - 5-Fold Cross Validation $R^2$ Score", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("ML Model", fontsize=12)
    ax1.set_ylabel("CV $R^2$ Score", fontsize=12)
    ax1.set_ylim(0.75, 1.0)
    
    for bar in bars.patches:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.annotate(f"{height:.4f}",
                        (bar.get_x() + bar.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, fontweight="bold", xytext=(0, 3),
                        textcoords="offset points")
            
    plt.xticks(rotation=15)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_residual_analysis(y_test: pd.Series, y_pred: np.ndarray, output_dir: str = "plots") -> str:
    """Generate Actual vs Predicted and Residual Distribution plots."""
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="#1f77b4", ax=ax1, s=60)
    min_val = min(y_test.min(), y_pred.min()) - 2
    max_val = max(y_test.max(), y_pred.max()) + 2
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction (y=x)")
    ax1.set_title("Actual vs Predicted Math Scores", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Actual Math Score", fontsize=11)
    ax1.set_ylabel("Predicted Math Score", fontsize=11)
    ax1.legend(loc="upper left")
    
    # Plot 2: Residual distribution
    sns.histplot(residuals, kde=True, color="#2ca02c", ax=ax2, bins=25)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_title("Residuals Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Residual Error", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "residual_analysis.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_feature_importance(pipeline, feature_names: List[str] = None, output_dir: str = "plots") -> str:
    """Extract and plot feature importances if supported by the model."""
    os.makedirs(output_dir, exist_ok=True)
    
    model = pipeline.named_steps["model"]
    
    if not hasattr(model, "feature_importances_"):
        print("Model does not provide feature_importances_. Skipping feature importance plot.")
        return ""
        
    importances = model.feature_importances_
    
    # Get feature names from ColumnTransformer preprocessor if available
    preprocessor = pipeline.named_steps["preprocessor"]
    
    try:
        num_cols = ["reading score", "writing score", "reading_writing_avg", "reading_writing_diff", "reading_writing_ratio"]
        nom_cols = preprocessor.named_transformers_["nom"].named_steps["onehot"].get_feature_names_out(
            ["gender", "race/ethnicity", "lunch", "test preparation course"]
        ).tolist()
        ord_cols = ["parental level of education"]
        all_features = num_cols + nom_cols + ord_cols
    except Exception:
        all_features = [f"Feature_{i}" for i in range(len(importances))]
        
    if len(all_features) != len(importances):
        all_features = [f"Feature_{i}" for i in range(len(importances))]
        
    df_imp = pd.DataFrame({"Feature": all_features, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x="Importance", y="Feature", palette="Blues_r")
    plt.title("Top Feature Importances (Best ML Model)", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path
