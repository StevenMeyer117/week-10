import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
import numpy as np

# Load dataset
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Function to map roast → number
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

# Create numeric roast column
df["roast_cat"] = df["roast"].apply(roast_category)

# Keep needed columns
df = df[["100g_USD", "roast_cat", "rating"]].dropna()

# Features and target
X = df[["100g_USD", "roast_cat"]]
y = df["rating"]

# Train Decision Tree
model = DecisionTreeRegressor()
model.fit(X, y)

# Save model
with open("model_2.pickle", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model_2.pickle")