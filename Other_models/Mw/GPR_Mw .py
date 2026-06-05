import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (mean_squared_error,mean_absolute_error,r2_score)

FEATURES = [ 'R1', 'R2', 'R3','Metal_NPA','d_cp-M','d_B-M','E_els','E_rep','LUMO','1-octene','Al/M','T']
TARGET = 'Mw'

DATA_DIR = 'data_Mw.csv'

TRAIN_FILE = os.path.join(DATA_DIR, 'train_data_Mw.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_data_Mw.csv')


def load_data():

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    return X_train, y_train, X_test, y_test


def preprocess_data(
        X_train,
        X_test,
        y_train,
        y_test
):

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y_train = pd.to_numeric(
        y_train,
        errors="coerce"
    )

    y_test = pd.to_numeric(
        y_test,
        errors="coerce"
    )

    imputer = SimpleImputer(
        strategy="mean"
    )

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )

    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    train_mask = ~y_train.isnull()
    test_mask = ~y_test.isnull()

    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask]

    X_test = X_test.loc[test_mask]
    y_test = y_test.loc[test_mask]

    return (
        X_train,
        y_train.values.ravel(),
        X_test,
        y_test.values.ravel()
    )


def gpr_cv(
        alpha,
        length_scale,
        nu
):

    kernel = Matern(
        length_scale=length_scale,
        nu=nu
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
    )

    score = cross_val_score(
        model,
        X_train_final,
        y_train_final,
        cv=cv,
        scoring="neg_mean_squared_error"
    )

    return score.mean()


def optimize_gpr():

    pbounds = {
        "alpha": (1e-6, 1e-1),
        "length_scale": (1e-2, 1e2),
        "nu": (0.5, 2.5)
    }

    optimizer = BayesianOptimization(
        f=gpr_cv,
        pbounds=pbounds,
        allow_duplicate_points=True
    )

    optimizer.maximize(
        init_points=5,
        n_iter=20
    )

    return optimizer.max


def train_model(best_params):

    kernel = Matern(
        length_scale=best_params["length_scale"],
        nu=best_params["nu"]
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=best_params["alpha"],
    )

    model.fit(
        X_train_final,
        y_train_final
    )

    return model


def evaluate(model):

    train_pred = model.predict(
        X_train_final
    )

    test_pred = model.predict(
        X_test_final
    )

    train_mse = mean_squared_error(
        y_train_final,
        train_pred
    )

    test_mse = mean_squared_error(
        y_test_final,
        test_pred
    )

    print("\n=========== Metrics ===========")

    print(
        f"Train MSE = {train_mse:.4f}"
    )

    print(
        f"Test MSE = {test_mse:.4f}"
    )

    print(
        f"Train RMSE = {np.sqrt(train_mse):.4f}"
    )

    print(
        f"Test RMSE = {np.sqrt(test_mse):.4f}"
    )

    print(
        f"Train MAE = "
        f"{mean_absolute_error(y_train_final, train_pred):.4f}"
    )

    print(
        f"Test MAE = "
        f"{mean_absolute_error(y_test_final, test_pred):.4f}"
    )

    print(
        f"Train R² = "
        f"{r2_score(y_train_final, train_pred):.4f}"
    )

    print(
        f"Test R² = "
        f"{r2_score(y_test_final, test_pred):.4f}"
    )

    return train_pred, test_pred


def plot_results(
        y_true,
        y_pred,
        title,
        filename
):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.8,
        s=70
    )

    mn = min(
        y_true.min(),
        y_pred.min()
    )

    mx = max(
        y_true.max(),
        y_pred.max()
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
        "r--",
        lw=2
    )

    plt.xlabel("Actual Mw")
    plt.ylabel("Predicted Mw")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=400
    )

    plt.show()


if __name__ == "__main__":


    X_train, y_train, X_test, y_test = load_data()

    (
        X_train_final,
        y_train_final,
        X_test_final,
        y_test_final
    ) = preprocess_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    best_result = optimize_gpr()

    print("\nBest Parameters:")
    print(best_result)

    model = train_model(
        best_result["params"]
    )

    train_pred, test_pred = evaluate(
        model
    )

    joblib.dump(
        model,
        "GPR_Mw.pkl"
    )

    plot_results(
        y_train_final,
        train_pred,
        "Training Set",
        "Mw_GPR_train.png"
    )

    plot_results(
        y_test_final,
        test_pred,
        "Test Set",
        "Mw_GPR_test.png"
    )