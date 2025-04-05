from statistics import median

import pandas as pd
#from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
data = pd.read_csv("StudentScore.xls")

target = "writing score"

# profile = ProfileReport(data, title = "Scores Report", explorative=True)
# profile.to_file("scores_report.html")

# Print the correlation for 3 columns
print(data[["math score", "reading score", "writing score"]].corr())


x = data.drop(target, axis = 1)
y = data[target]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

# Create a Pipline to handle missing value by using median (most used because avoid outliers) and preprocessing data for numerical feature
# num_transformer = Pipeline(steps= [("impute", SimpleImputer(strategy="median")),
#                                    ("Standard Scaler", StandardScaler()),
# ])

# Create a Pipline to handle missing value by using most frequent with  encrypt ordinal feature
education_levels = ["some high school", "high school", "some college", "associate's degree",
                    "bachelor's degree", "master's degree"]
ordinal_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                      ("Scaler", OrdinalEncoder(categories=[education_levels]))
                                      ])

output = ordinal_transformer.fit_transform(x_train[["parental level of education"]])

# Print train set before and after preprocessing
for i, j in zip(x_train["parental level of education"].values, output):
    print(f"Before: {i}. After: {j}")

#print(x_train[["math score", "reading score"]].values)


