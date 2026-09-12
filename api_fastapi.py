import os
import sys
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException

sys.path.insert(0, os.path.dirname(__file__))

from src.predict import StudentScorePredictor
from src.data_preprocessing import EDUCATION_ORDER

app = FastAPI(
    title="Student Score Prediction REST API",
    description="Production Machine Learning REST API for predicting student math scores.",
    version="2.0.0"
)

# Load predictor
try:
    predictor = StudentScorePredictor()
except Exception as e:
    predictor = None


class StudentInput(BaseModel):
    gender: str = Field(..., example="female")
    race_ethnicity: str = Field(..., alias="race/ethnicity", example="group B")
    parental_level_of_education: str = Field(..., alias="parental level of education", example="bachelor's degree")
    lunch: str = Field(..., example="standard")
    test_preparation_course: str = Field(..., alias="test preparation course", example="none")
    reading_score: float = Field(..., alias="reading score", ge=0, le=100, example=72.0)
    writing_score: float = Field(..., alias="writing score", ge=0, le=100, example=74.0)

    class Config:
        allow_population_by_field_name = True

    @validator("gender")
    def validate_gender(cls, v):
        if v not in ["female", "male"]:
            raise ValueError("gender must be 'female' or 'male'")
        return v

    @validator("parental_level_of_education")
    def validate_edu(cls, v):
        if v not in EDUCATION_ORDER:
            raise ValueError(f"parental level of education must be one of {EDUCATION_ORDER}")
        return v


class PredictionResponse(BaseModel):
    predicted_math_score: float
    rounded_math_score: int
    reading_score: float
    writing_score: float


@app.get("/health", summary="API Health Check")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.get("/model-info", summary="Model Metadata and Metrics")
def model_info():
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "metrics_summary.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics summary not found. Please train the model first.")
    with open(metrics_path, "r") as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse, summary="Predict Math Score for Single Student")
def predict_single_student(student: StudentInput):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model predictor is not initialized.")
    try:
        input_data = student.dict(by_alias=True)
        result = predictor.predict_single(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-batch", summary="Batch Predict Math Scores")
def predict_batch_students(students: List[StudentInput]):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model predictor is not initialized.")
    try:
        input_list = [s.dict(by_alias=True) for s in students]
        df = pd.DataFrame(input_list)
        df_out = predictor.predict_batch(df)
        return df_out.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
