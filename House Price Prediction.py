import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------- 1. Load and explore
def load_data(path=None, target=None):
    if path:
        df = pd.read_csv(path)
        return df, target
    from sklearn.datasets import fetch_california_housing
    df = fetch_california_housing(as_frame=True).frame
    return df, "MedHouseVal"


def explore(df, target):
    print("Shape:", df.shape)
    print("\nFirst rows:\n", df.head())
    print("\nSummary statistics:\n", df.describe().T.round(2))
    print("\nMissing values per column:\n", df.isna().sum().to_string())

    df.hist(bins=40, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig("distributions.png", dpi=120)
    plt.close()

    corr = df.corr(numeric_only=True)[target].drop(target).sort_values()
    print(f"\nCorrelation with {target}:\n", corr.round(3).to_string())
    corr.plot.barh(title=f"Correlation with {target}")
    plt.tight_layout()
    plt.savefig("correlations.png", dpi=120)
    plt.close()


# ------------------------------------------------------ 2. Feature engineering
def add_features(df):
    """Example engineered features (only added if the needed columns exist)."""
    df = df.copy()
    if {"AveRooms", "AveBedrms"} <= set(df.columns):
        df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
    if {"AveRooms", "AveOccup"} <= set(df.columns):
        df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to a CSV file")
    parser.add_argument("--target", help="target column name (required with --data)")
    args = parser.parse_args()
    if args.data and not args.target:
        parser.error("--target is required when using --data")

    df, target = load_data(args.data, args.target)
    explore(df, target)

    df = add_features(df)
    df = df.select_dtypes(include=np.number)
    df = df.dropna(subset=[target])

    X, y = df.drop(columns=[target]), df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print("\n=== Test-set performance ===")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}   (typical error, same units as the target)")
    print(f"MAE : {mean_absolute_error(y_test, pred):.4f}")
    print(f"R^2 : {r2_score(y_test, pred):.4f}   (share of variance explained)")

    baseline = np.full_like(y_test, y_train.mean(), dtype=float)
    print(f"Baseline (predict mean) MSE: {mean_squared_error(y_test, baseline):.4f}")

    coefs = pd.Series(model.named_steps["reg"].coef_, index=X.columns).sort_values()
    print("\nFeature coefficients (per 1 std dev):\n", coefs.round(3).to_string())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.3, s=8)
    lim = [y_test.min(), y_test.max()]
    plt.plot(lim, lim, "r--", label="perfect prediction")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.legend()
    plt.title("Predicted vs actual")
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=120)
    print("\nSaved plots: distributions.png, correlations.png, predicted_vs_actual.png")


if __name__ == "__main__":
    main()