import json
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor
)
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

from src.data_preprocessing import create_full_pipeline

# Check optional advanced ML libraries
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_candidate_models() -> Dict[str, Any]:
    """Instantiate a dictionary of candidate ML regressors."""
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=42)
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
        
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(iterations=200, learning_rate=0.05, verbose=0, random_seed=42)
        
    return models


def evaluate_models_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5
) -> pd.DataFrame:
    """Perform K-Fold Cross Validation across all candidate models."""
    candidate_models = get_candidate_models()
    results = []
    
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error"
    }
    
    for name, estimator in candidate_models.items():
        pipeline = create_full_pipeline(estimator)
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        r2_mean = np.mean(cv_res["test_r2"])
        r2_std = np.std(cv_res["test_r2"])
        mae_mean = -np.mean(cv_res["test_mae"])
        rmse_mean = np.mean(np.sqrt(-cv_res["test_mse"]))
        
        results.append({
            "Model": name,
            "CV_R2_Mean": r2_mean,
            "CV_R2_Std": r2_std,
            "CV_MAE": mae_mean,
            "CV_RMSE": rmse_mean
        })
        
    results_df = pd.DataFrame(results).sort_values(by="CV_R2_Mean", ascending=False).reset_index(drop=True)
    return results_df


def create_stacking_ensemble(top_estimators: list) -> StackingRegressor:
    """Build a Stacking Regressor combining top estimators with Ridge meta-learner."""
    estimators = [(name, est) for name, est in top_estimators]
    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    )
    return stacking_reg


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_model_name: str = "RandomForest"
) -> Pipeline:
    """Hyperparameter tuning for the selected top model using GridSearchCV."""
    
    if best_model_name == "RandomForest":
        base_model = RandomForestRegressor(random_state=42)
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
    elif best_model_name == "ExtraTrees":
        base_model = ExtraTreesRegressor(random_state=42)
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        }
    elif best_model_name == "GradientBoosting":
        base_model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 5]
        }
    else:
        base_model = RandomForestRegressor(random_state=42)
        param_grid = {
            "model__n_estimators": [100, 200, 300]
        }
        
    pipeline = create_full_pipeline(base_model)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params for {best_model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Compute performance metrics on holdout test set."""
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100.0
    
    return {
        "R2": float(r2),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }


def save_model_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, Any],
    model_dir: str = "models"
) -> Tuple[str, str]:
    """Save trained pipeline and metrics summary to disk."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.joblib")
    metrics_path = os.path.join(model_dir, "metrics_summary.json")
    
    joblib.dump(pipeline, model_path)
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    return model_path, metrics_path
