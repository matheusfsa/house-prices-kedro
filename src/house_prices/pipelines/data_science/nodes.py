import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV


def get_model(
    X: np.ndarray,
    y: np.ndarray,
    kfolds: int,
    n_estimators: List[int],
    n_jobs: int,
    random_state: int
) -> BaseEstimator:

    kf = KFold(kfolds, shuffle=True, random_state=random_state).get_n_splits(X)

    model = RandomForestRegressor(random_state=random_state)
    cv = GridSearchCV(model, {"n_estimators": n_estimators}, cv=kf, n_jobs=n_jobs, verbose=2)
    cv.fit(X, y)

    return cv.best_estimator_


def evaluate_model(
    model: BaseEstimator, X: np.ndarray, y: np.ndarray, kfold: int
) -> None:
    

    kf = KFold(kfold, shuffle=True, random_state=42).get_n_splits(X)
    rmse = np.sqrt(
        -cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf)
    )

    log = logging.getLogger(__name__)
    log.info("RMSE on train set: {}".format(rmse))


def predict(
    model: BaseEstimator, X_test_df: pd.DataFrame, X: np.ndarray, y: np.ndarray, X_test: np.ndarray
) -> pd.DataFrame:

    model.fit(X, y)
    y_pred = model.predict(X_test)
    sub = pd.DataFrame()
    sub["Id"] = X_test_df.Id
    sub["SalePrice"] = np.floor(np.expm1(y_pred))
    return sub
