import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# Define exact logical hierarchy for ordinal education level
EDUCATION_ORDER = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]

NUMERICAL_FEATURES = ["reading score", "writing score"]
NOMINAL_FEATURES = ["gender", "race/ethnicity", "lunch", "test preparation course"]
ORDINAL_FEATURES = ["parental level of education"]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn Transformer for student score feature engineering."""
    
    def __init__(self, add_ratios: bool = True):
        self.add_ratios = add_ratios
        self.feature_names_out_: List[str] = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        # Ensure numeric columns are present
        if "reading score" in X_df.columns and "writing score" in X_df.columns:
            reading = X_df["reading score"].astype(float)
            writing = X_df["writing score"].astype(float)
            
            X_df["reading_writing_avg"] = (reading + writing) / 2.0
            X_df["reading_writing_diff"] = reading - writing
            
            if self.add_ratios:
                X_df["reading_writing_ratio"] = (reading + 1.0) / (writing + 1.0)
                
        self.feature_names_out_ = list(X_df.columns)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


def load_data(file_path: str = "StudentScore.xls") -> pd.DataFrame:
    """Load dataset from CSV or XLS format."""
    df = pd.read_csv(file_path)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Construct sklearn ColumnTransformer for numeric, nominal, and ordinal columns."""
    
    # Numerical pipeline: impute -> feature engineering -> scale
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Nominal pipeline: impute -> one-hot encoding
    nom_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Ordinal pipeline: impute -> ordinal encoding with strict logical hierarchy
    ord_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[EDUCATION_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, ["reading score", "writing score", "reading_writing_avg", "reading_writing_diff", "reading_writing_ratio"]),
            ("nom", nom_pipeline, NOMINAL_FEATURES),
            ("ord", ord_pipeline, ORDINAL_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor


def create_full_pipeline(model_estimator) -> Pipeline:
    """Build a end-to-end sklearn Pipeline with feature engineering, preprocessing, and model."""
    full_pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer(add_ratios=True)),
        ("preprocessor", build_preprocessor()),
        ("model", model_estimator)
    ])
    return full_pipeline


def prepare_data(
    file_path: str = "StudentScore.xls",
    target_col: str = "math score",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, split into train and test sets."""
    df = load_data(file_path)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
