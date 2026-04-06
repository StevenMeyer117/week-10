import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def load_data(url):
    """Load dataset from URL."""
    return pd.read_csv(url)


def train_linear_model(df):
    """
    Train Linear Regression model using 100g_USD to predict rating.
    Returns trained model.
    """
    df_clean = df[["100g_USD", "rating"]].dropna()

    x = df_clean[["100g_USD"]]
    y = df_clean["rating"]

    model = LinearRegression()
    model.fit(x, y)

    return model


def roast_category(roast):
    """
    Convert roast category into numeric values.
    Light -> 1, Medium -> 2, Dark -> 3.
    """
    if pd.isna(roast):
        return np.nan

    roast = roast.lower()

    if "light" in roast:
        return 1
    if "medium" in roast:
        return 2
    if "dark" in roast:
        return 3

    return np.nan


def train_tree_model(df):
    """
    Train Decision Tree model using 100g_USD and roast_cat.
    Returns trained model.
    """
    df["roast_cat"] = df["roast"].apply(roast_category)

    df_clean = df[["100g_USD", "roast_cat", "rating"]].dropna()

    x = df_clean[["100g_USD", "roast_cat"]]
    y = df_clean["rating"]

    model = DecisionTreeRegressor()
    model.fit(x, y)

    return model


def save_model(model, filename):
    """Save trained model to pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def main():
    """Main execution function."""
    url = (
        "https://raw.githubusercontent.com/leontoddjohnson/"
        "datasets/main/data/coffee_analysis.csv"
    )

    df = load_data(url)

    # Train and save model 1
    model_1 = train_linear_model(df)
    save_model(model_1, "model_1.pickle")

    # Train and save model 2
    model_2 = train_tree_model(df)
    save_model(model_2, "model_2.pickle")

    print("model_1.pickle and model_2.pickle created")


if __name__ == "__main__":
    main()