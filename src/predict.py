import os
import joblib
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Any

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.joblib")

class StudentScorePredictor:
    """Inference engine for Student Score Prediction."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Please run train_pipeline.py first to train and serialize the model."
            )
        self.pipeline = joblib.load(self.model_path)

    def validate_input(self, df: pd.DataFrame):
        """Validate input DataFrame schema and value ranges."""
        required_cols = [
            "gender", "race/ethnicity", "parental level of education",
            "lunch", "test preparation course", "reading score", "writing score"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required feature column: '{col}'")
                
        # Validate score ranges
        for score_col in ["reading score", "writing score"]:
            if (df[score_col] < 0).any() or (df[score_col] > 100).any():
                raise ValueError(f"Score values in '{score_col}' must be between 0 and 100.")

    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict math score for a single student input dictionary."""
        df = pd.DataFrame([input_dict])
        self.validate_input(df)
        pred = self.pipeline.predict(df)[0]
        clipped_pred = float(np.clip(pred, 0, 100))
        
        return {
            "predicted_math_score": round(clipped_pred, 2),
            "rounded_math_score": int(round(clipped_pred)),
            "reading_score": input_dict.get("reading score"),
            "writing_score": input_dict.get("writing score")
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict math scores for a batch DataFrame."""
        self.validate_input(df)
        preds = self.pipeline.predict(df)
        clipped_preds = np.clip(preds, 0, 100)
        
        df_out = df.copy()
        df_out["predicted_math_score"] = np.round(clipped_preds, 2)
        df_out["rounded_math_score"] = np.round(clipped_preds).astype(int)
        return df_out
