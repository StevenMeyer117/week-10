import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle
import numpy as np

# Load dataset
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# ---------------------------
# MODEL 1 (Linear Regression)
# ---------------------------

df1 = df[["100g_USD", "rating"]].dropna()

X1 = df1[["100g_USD"]]
y1 = df1["rating"]

model1 = LinearRegression()
model1.fit(X1, y1)

with open("model_1.pickle", "wb") as f:
    pickle.dump(model1, f)

print("model_1.pickle created")

# ---------------------------
# MODEL 2 (Decision Tree)
# ---------------------------

def roast_category(roast):
    if pd.isna(roast):
        return np.nan
    
    roast = roast.lower()
    
    if "light" in roast:
        return 1
    elif "medium" in roast:
        return 2
    elif "dark" in roast:
        return 3
    else:
        return np.nan

df["roast_cat"] = df["roast"].apply(roast_category)

df2 = df[["100g_USD", "roast_cat", "rating"]].dropna()

X2 = df2[["100g_USD", "roast_cat"]]
y2 = df2["rating"]

model2 = DecisionTreeRegressor()
model2.fit(X2, y2)

with open("model_2.pickle", "wb") as f:
    pickle.dump(model2, f)

print("model_2.pickle created")