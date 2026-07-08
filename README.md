# Student Scores Prediction

## 📋 Giới thiệu Dự án

Dự án này áp dụng Machine Learning để dự đoán điểm toán của học sinh dựa trên các đặc điểm cá nhân và học tập. Sử dụng **Random Forest Regressor** kết hợp với **GridSearchCV** để tìm kiếm các siêu tham số tối ưu.

## 🎯 Mục tiêu

Xây dựng một mô hình có khả năng dự đoán chính xác điểm toán của học sinh dựa trên:
- Điểm đọc và viết
- Chủng tộc/Dân tộc
- Trình độ học vấn của phụ huynh
- Giới tính
- Kiểu ăn trưa
- Khóa học chuẩn bị thi

## 📊 Dữ liệu

**Tệp dữ liệu:** `StudentScore.xls` (định dạng CSV)

### Các đặc trưng:
- **Đặc trưng số:**
  - `writing score` - Điểm viết
  - `reading score` - Điểm đọc

- **Đặc trưng phân loại:**
  - `race/ethnicity` - Chủng tộc/Dân tộc
  - `parental level of education` - Trình độ học vấn phụ huynh (high school, some high school, some college, associate's degree, bachelor's degree, master's degree)
  - `gender` - Giới tính
  - `lunch` - Kiểu ăn trưa
  - `test preparation course` - Khóa học chuẩn bị thi

- **Target:** `math score` - Điểm toán

## 🏗️ Kiến trúc Mô hình

### Pipeline Xử lý

Dự án sử dụng `ColumnTransformer` để xử lý các loại dữ liệu khác nhau:

1. **Xử lý Đặc trưng Số (`num_processor`)**
   - SimpleImputer (chiến lược: mean/median)
   - StandardScaler

2. **Xử lý Đặc trưng Danh mục Danh nghĩa (`nom_processor`)**
   - SimpleImputer (chiến lược: most_frequent)
   - OneHotEncoder

3. **Xử lý Đặc trưng Danh mục Thứ tự (`ord_processor`)**
   - SimpleImputer (chiến lược: most_frequent)
   - OrdinalEncoder (với danh sách các hạng mục được sắp xếp)

### Mô hình

- **Thuật toán:** Random Forest Regressor
- **Phương pháp Tuning:** GridSearchCV (5-Fold Cross Validation)

### Siêu tham số tìm kiếm

```python
params = {
    'preprocessing__num_processor__imputer__strategy': ["mean", "median"],
    'model__n_estimators': [100, 200, 300],
    'model__criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"],
}
```

**Tổng số kết hợp:** 2 × 3 × 4 = 24 kết hợp, được đánh giá qua 5-fold CV = **120 lần huấn luyện**

## 📦 Cài đặt

### Yêu cầu Hệ thống
- Python 3.7+
- pip hoặc conda

### Các Thư viện Cần Thiết

```bash
pip install pandas scikit-learn
```

Hoặc cài dùng file requirements:

```bash
pip install -r requirements.txt
```

### Danh sách Thư viện

```
pandas>=1.0.0
scikit-learn>=0.24.0
```

## 🚀 Cách Sử dụng

### 1. Chuẩn bị Dữ liệu
Đảm bảo tệp `StudentScore.xls` nằm trong cùng thư mục với script.

### 2. Chạy Script

```bash
python StudentScores_Prediction.py
```

### 3. Kết quả Đầu ra

Script sẽ in ra:
- **Best params:** Các siêu tham số tốt nhất được tìm thấy
- **MSE:** Mean Squared Error trên tập test
- **MAE:** Mean Absolute Error trên tập test
- **r2_score:** R² Score (hệ số xác định)

**Ví dụ Kết quả:**
```
Best params: {'model__criterion': 'squared_error', 'model__n_estimators': 200, 'preprocessing__num_processor__imputer__strategy': 'median'}
MSE: 25.34
MAE: 3.45
r2_score: 0.8765
```

## 📁 Cấu trúc Thư mục

```
StudentScores_Prediction/
├── README.md                          # Tệp này
├── StudentScores_Prediction.py        # Script chính
├── StudentScore.xls                   # Dữ liệu (định dạng CSV)
├── Score_report.html                  # Báo cáo phân tích (tùy chọn)
└── requirements.txt                   # Danh sách thư viện
```

## 📈 Kết quả Mong đợi

### Chỉ số Hiệu suất
- **R² Score:** Thường trên 0.85 (mô hình giải thích 85%+ phương sai)
- **MAE:** Khoảng 2-4 điểm
- **MSE:** Khoảng 20-30

### Ưu điểm Random Forest
- Xử lý tốt các đặc trưng không tuyến tính
- Ít nhạy cảm với dữ liệu ngoại lệ
- Cung cấp tầm quan trọng đặc trưng

## 🔧 Tuỳ chỉnh

### Thay đổi Tỉ lệ Train/Test
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Thay đổi test_size (mặc định 0.2 = 20%)
```

### Thêm Siêu tham số Mới
```python
params = {
    'preprocessing__num_processor__imputer__strategy': ["mean", "median"],
    'model__n_estimators': [100, 200, 300, 400],  # Thêm 400
    'model__criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    'model__max_depth': [10, 15, 20],  # Thêm tham số mới
}
```

### Thay đổi Giá trị CV
```python
Grid_Search_CV = GridSearchCV(estimator=reg, param_grid=params, cv=10, scoring="r2", verbose=2)
# Thay đổi cv=10 cho 10-fold cross validation
```

## 📊 Phân tích Dữ liệu (Tùy chọn)

Dự án có sẵn code để tạo báo cáo phân tích chi tiết (hiện đã comment):

```python
# profile = ProfileReport(df, title="Score Report", explorative=True)
# profile.to_file("Score_report.html")
```

Để sử dụng, cài đặt: `pip install ydata-profiling`, sau đó uncomment các dòng trên.

## 🐛 Khắc phục Sự cố

### ModuleNotFoundError
**Lỗi:** `No module named 'ydata_profiling'`
**Giải pháp:** Cài đặt thư viện hoặc comment out dòng import

### ValueError: Excel file format cannot be determined
**Lỗi:** Không thể đọc file xls
**Giải pháp:** Đảm bảo `StudentScore.xls` là file CSV hợp lệ

### ImportError: Missing optional dependency 'xlrd'
**Lỗi:** Thiếu thư viện xlrd
**Giải pháp:** `pip install xlrd`

## 📝 Ghi chú Kỹ thuật

### Chiến lược Xử lý Giá trị Khuyết

- **Đặc trưng số:** Impute bằng trung bình (mean) hoặc trung vị (median)
- **Đặc trưng phân loại:** Impute bằng giá trị xuất hiện nhiều nhất (most_frequent)

### Encode Phân loại

- **Danh mục Thứ tự (Ordinal):** Trình độ giáo dục được encode theo thứ tự:
  ```
  1. high school
  2. some high school
  3. some college
  4. associate's degree
  5. bachelor's degree
  6. master's degree
  ```

- **Danh mục Danh nghĩa (Nominal):** race/ethnicity được encode dùng One-Hot Encoding

### Chia Dữ liệu

- **Train:** 80% dữ liệu
- **Test:** 20% dữ liệu
- **Random State:** 42 (để dễ tái tạo)

## 🔄 Quy trình Làm việc

1. **Tải dữ liệu** → Đọc file CSV
2. **Tách Target** → Tách `math score` từ các đặc trưng
3. **Chia dữ liệu** → 80% train, 20% test
4. **Xây dựng Pipeline** → Xử lý và mô hình
5. **Tuning Hyperparameter** → GridSearchCV với 5-fold CV
6. **Dự đoán** → Dự đoán trên tập test
7. **Đánh giá** → Tính MSE, MAE, R² Score

## 📚 Tài liệu Tham khảo

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/YourFeature`)
3. Commit thay đổi (`git commit -m 'Add YourFeature'`)
4. Push lên branch (`git push origin feature/YourFeature`)
5. Tạo Pull Request

## 📄 Giấy phép

Dự án này được cấp phép dưới giấy phép MIT. Xem tệp [LICENSE](LICENSE) để biết thêm chi tiết.

## 👤 Tác giả

- **Tên:** Tùy chỉnh
- **GitHub:** [Your GitHub Profile](https://github.com/yourusername)
- **Email:** your.email@example.com

## 📞 Liên hệ & Hỗ trợ

Nếu bạn có bất kỳ câu hỏi hoặc gặp vấn đề, vui lòng:
- Tạo một Issue trên GitHub
- Gửi email đến: your.email@example.com

## 📝 Lịch sử Thay đổi

### v1.0 (2026-07-08)
- Khởi tạo dự án
- Xây dựng pipeline xử lý dữ liệu
- Triển khai GridSearchCV cho tuning hyperparameter
- Hoàn thành đánh giá mô hình

## ⭐ Nếu Dự án Hữu ích

Nếu dự án này giúp ích cho bạn, vui lòng cho nó một ⭐ Star!

---

**Cập nhật lần cuối:** 8 July 2026
