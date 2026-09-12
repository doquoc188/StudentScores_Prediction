# 🎓 Student Scores Prediction v2.0 - Production ML System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20Ensembles-orange)
![UI](https://img.shields.io/badge/Web%20UI-Streamlit-red)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

## 📋 Giới thiệu Dự án (Project Overview)

**Student Scores Prediction v2.0** là hệ thống Machine Learning end-to-end hoàn chỉnh được nâng cấp nhằm dự đoán chính xác điểm toán của học sinh dựa trên các yếu tố nhân khẩu học, trình độ học vấn của phụ huynh, và kết quả các bài kiểm tra đọc/viết.

Hệ thống được thiết kế theo chuẩn **Production Machine Learning**:
- 🛠️ **Data Preprocessing & Feature Engineering:** Sửa toàn bộ lỗi logic phân loại (Ordinal/Nominal), bổ sung đặc trưng tương tác (`reading_writing_avg`, `reading_writing_diff`, `reading_writing_ratio`).
- 🤖 **Multi-Model Benchmarking:** So sánh tự động 8+ thuật toán ML (Ridge, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, Stacking Ensemble) qua 5-Fold Cross Validation.
- 💾 **Model Serialization:** Lưu trữ Pipeline tối ưu dưới dạng `.joblib` và ghi log chỉ số đánh giá (`metrics_summary.json`).
- 🎨 **Visual Analytics:** Tự động tạo biểu đồ so sánh mô hình, phân tích phần dư (residual analysis), và tầm quan trọng đặc trưng (feature importances).
- 🌐 **Interactive Streamlit Web App:** Giao diện trực quan cho phép dự đoán thời gian thực, xem dashboard phân tích và EDA.
- ⚡ **FastAPI REST API Service:** Cung cấp các RESTful Endpoints (`/predict`, `/predict-batch`, `/health`, `/model-info`).
- 🧪 **Pytest Suite:** Bộ kiểm thử tự động toàn bộ pipeline xử lý dữ liệu và suy luận mô hình.

---

## 🏗️ Kiến trúc Hệ thống (System Architecture)

```text
D:\ML\StudentScores_Prediction\
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Feature engineering & Sklearn ColumnTransformer
│   ├── model_training.py          # Benchmark ML models, CV & Hyperparameter tuning
│   ├── evaluation.py              # Visualizations (Residuals, Feature Importances)
│   └── predict.py                 # StudentScorePredictor inference class
├── models/
│   ├── best_model.joblib          # Serialized trained model & preprocessor
│   └── metrics_summary.json       # Logged metrics & benchmark scores
├── plots/
│   ├── model_comparison.png       # Barplot CV R² scores
│   ├── residual_analysis.png      # Actual vs Predicted & Error distribution
│   └── feature_importance.png     # Relative feature importance plot
├── tests/
│   ├── test_preprocessing.py      # Unit tests for preprocessing
│   └── test_prediction.py         # Unit tests for inference pipeline
├── train_pipeline.py              # CLI Entrypoint for model training & evaluation
├── app_streamlit.py               # Streamlit Web Application
├── api_fastapi.py                 # FastAPI REST Endpoints server
├── StudentScores_Prediction.py    # Legacy backward compatibility entrypoint
├── requirements.txt               # Dependencies list
└── README.md                      # Documentation
```

---

## 📊 Kết quả Benchmarking (Model Leaderboard)

| Model Algorithm | CV $R^2$ Mean | CV MAE | CV RMSE | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Ridge Regression** | **0.8688** | **4.36 pts** | **5.42 pts** | 🏆 **Best Model** |
| **Gradient Boosting** | 0.8468 | 4.74 pts | 5.87 pts | Baseline |
| **Random Forest** | 0.8390 | 4.83 pts | 6.00 pts | Baseline |
| **HistGradientBoosting** | 0.8346 | 4.87 pts | 6.09 pts | Baseline |
| **Extra Trees** | 0.8314 | 4.93 pts | 6.14 pts | Baseline |

### Holdout Test Set Performance (Winning Model):
- **$R^2$ Score:** `0.8397` (Mô hình giải thích ~84% biến thiên của điểm Toán)
- **MAE:** `4.83 điểm`
- **RMSE:** `6.25 điểm`

---

## 🚀 Hướng dẫn Sử dụng (Quick Start Guide)

### 1. Cài đặt Thư viện
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện Mô hình & Tạo Artifacts (Train Pipeline)
Chạy script chính để tự động train, so sánh mô hình, lưu model và xuất biểu đồ:
```bash
python train_pipeline.py
```
*(Hoặc chạy qua file legacy `python StudentScores_Prediction.py`)*

### 3. Chạy ứng dụng Web Dashboard (Streamlit UI)
```bash
streamlit run app_streamlit.py
```
👉 Truy cập giao diện tại: `http://localhost:8501`

### 4. Chạy REST API Server (FastAPI)
```bash
uvicorn api_fastapi:app --reload --port 8000
```
👉 Xem tài liệu Swagger API tại: `http://localhost:8000/docs`

### 5. Chạy Kiểm thử (Pytest Unit Tests)
```bash
pytest tests/
```

---

## 📡 REST API Examples

### `POST /predict`
**Request Body:**
```json
{
  "gender": "female",
  "race/ethnicity": "group B",
  "parental level of education": "bachelor's degree",
  "lunch": "standard",
  "test preparation course": "none",
  "reading score": 72.0,
  "writing score": 74.0
}
```

**Response:**
```json
{
  "predicted_math_score": 67.45,
  "rounded_math_score": 67,
  "reading_score": 72.0,
  "writing_score": 74.0
}
```

---

## 📝 Giấy phép & Tác giả

- **Tác giả:** Đỗ Trọng Quốc
- **Email:** dotrongquoc1808@gmail.com
- **License:** MIT License
