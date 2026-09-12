import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# 1. Đọc dữ liệu
df = pd.read_csv("StudentScore.xls")

# 2. Kỹ thuật đặc trưng (Feature Engineering)
df["reading_writing_avg"] = (df["reading score"] + df["writing score"]) / 2.0
df["reading_writing_diff"] = df["reading score"] - df["writing score"]

# 3. Tách target
target = "math score"
x = df.drop(target, axis=1)
y = df[target]

# 4. Chia tập dữ liệu Train / Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Pipeline xử lý đặc trưng số (Numerical Features)
num_transform = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scalar", StandardScaler())
])

# 6. Pipeline xử lý đặc trưng thứ tự (Ordinal Features) - Thứ tự chuẩn về trình độ học vấn
Encoder_List = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]

ord_transform = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[Encoder_List],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

# 7. Pipeline xử lý đặc trưng danh nghĩa (Nominal Features) - Giới tính, ăn trưa, chủng tộc, khóa học
nom_transform = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 8. Tích hợp ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num_processor", num_transform, ["writing score", "reading score", "reading_writing_avg", "reading_writing_diff"]),
    ("ord_processor", ord_transform, ["parental level of education"]),
    ("nom_processor", nom_transform, ["gender", "race/ethnicity", "lunch", "test preparation course"])
])

# 9. Đặt Pipeline huấn luyện mô hình
reg = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", Ridge())
])

# 10. GridSearchCV tìm kiếm siêu tham số tối ưu
params = {
    'preprocessing__num_processor__imputer__strategy': ["mean", "median"],
    'model__alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
}

Grid_Search_CV = GridSearchCV(estimator=reg, param_grid=params, cv=5, scoring="r2", verbose=1, n_jobs=-1)
Grid_Search_CV.fit(x_train, y_train)

# 11. Đánh giá kết quả trên tập Test
best_model = Grid_Search_CV.best_estimator_
y_predict = best_model.predict(x_test)

print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
print("Best params:", Grid_Search_CV.best_params_)
print("MSE: {:.4f}".format(mean_squared_error(y_test, y_predict)))
print("RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_predict))))
print("MAE: {:.4f}".format(mean_absolute_error(y_test, y_predict)))
print("r2_score: {:.4f}".format(r2_score(y_test, y_predict)))

# 12. Lưu model để tái sử dụng
joblib.dump(best_model, "best_model.pkl")
print("✅ Đã lưu mô hình tối ưu vào file 'best_model.pkl' thành công!")
