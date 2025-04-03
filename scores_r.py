import pandas as pd
#from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split
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





