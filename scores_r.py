import pandas as pd
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from lazypredict.Supervised import LazyRegressor

data = pd.read_csv("StudentScore.xls")

target = "writing score"

# profile = ProfileReport(data, title = "Scores Report", explorative=True)
# profile.to_file("scores_report.html")

# Print the correlation for 3 columns
# print(data[["math score", "reading score", "writing score"]].corr())

x = data.drop(target, axis = 1)
y = data[target]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

# Create a Pipline to handle missing value by using median (most used because avoid outliers) and preprocessing data for numerical feature
num_transformer = Pipeline(steps= [("impute", SimpleImputer(strategy="median")),
                                   ("Standard Scaler", StandardScaler()),
])

# Create a Pipline to handle missing value by using most frequent (aka mod) with  encrypt ordinal feature
education_levels = ["some high school", "high school", "some college", "associate's degree",
                    "bachelor's degree", "master's degree"]
# Get unique value from genders column
gender_value = data["gender"].unique()
lunch_value = data["lunch"].unique()
test_preparation_course_value = data["test preparation course"].unique()

# Get unique value from
ordinal_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                      ("encoder", OrdinalEncoder(categories=[education_levels, gender_value, lunch_value, test_preparation_course_value]))
                                      ])

# Create Pipeline to handle missing value by using most frequent (aka mod) with one hot coding
nominal_transformer = Pipeline(steps= [("impute", SimpleImputer(strategy="most_frequent")),
                                       ("encoder", OneHotEncoder())
                                       ])

ult_transformers = ColumnTransformer([
                        ("number transformer", num_transformer, ["math score", "reading score"]),
                        ("ordinal transformer", ordinal_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
                        ("nominal transformer", nominal_transformer, ["race/ethnicity"])
                        ])


# # Print train set before and after preprocessing
# for i, j in zip(x_train["race/ethnicity"].values, output):
#     print(f"Before: {i}. After: {j}")

#print(x_train[["math score", "reading score"]].values)

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)

model = Pipeline(steps=[
                ("preprocessors", ult_transformers),
                ("regressor", RandomForestRegressor())
])

model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

print(f"MAE: {mean_absolute_error(y_test, y_predicted)}")
print(f"MSE: {mean_squared_error(y_test, y_predicted)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_predicted)}")
print(f"R^2: {r2_score(y_test, y_predicted)}")
