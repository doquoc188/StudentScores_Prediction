import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("StudentScore.xls")

# profile = ProfileReport(df, title="Score Report", explorative=True)
# profile.to_file("Score_report.html")

# print(df.columns)
# print(df["parental level of education"].unique())
target = "math score"
x = df.drop(target, axis =1 )
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)

num_transform = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scalar", StandardScaler())
])


Encoder_List = ["high school","some high school","some college", "associate's degree","bachelor's degree","master's degree"]
gender_value = x_train["gender"].unique()
lunch_value = x_train["lunch"].unique()
test_preparation_value = x_train["test preparation course"].unique()

ord_transform = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories= [Encoder_List, gender_value, lunch_value, test_preparation_value]))
])

nom_transform = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder())
])
# result = nom_transform.fit_transform(x_train[["race/ethnicity", "gender", "lunch", "test preparation course"]])

preprocessor = ColumnTransformer(transformers=[
    ("num_processor", num_transform, ["writing score", "reading score"]),
    ("nom_processor", nom_transform, ["race/ethnicity"]),
    ("ord_processor", ord_transform, ["parental level of education","gender", "lunch", "test preparation course"])
])

reg = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor())
])

# reg.fit(x_train, y_train)
# y_predict = reg.predict(x_test)

params = {
    'preprocessing__num_processor__imputer__strategy': ["mean", "median"],
    'model__n_estimators': [100, 200, 300],
    'model__criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    
}
Grid_Search_CV = GridSearchCV(estimator=reg, param_grid= params, cv = 5, scoring= "r2" ,verbose= 2)
Grid_Search_CV.fit(x_train, y_train)

best_model = Grid_Search_CV.best_estimator_
y_predict = best_model.predict(x_test)

print("Best params:", Grid_Search_CV.best_params_)
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("r2_score: {}".format(r2_score(y_test, y_predict)))

