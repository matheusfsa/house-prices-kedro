from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def cleaning_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[str, Any]:

    all_data = pd.concat(
        [df_train.drop(["Id", "SalePrice"], axis=1), df_test.drop(["Id"], axis=1)]
    )
    notna_columns = [
        "Alley",
        "MasVnrType",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "BsmtFinType1",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
    ]
    all_data[notna_columns] = all_data[notna_columns].fillna("Not")
    all_data["LotFrontage"] = all_data["LotFrontage"].fillna(
        all_data["LotFrontage"].mean()
    )
    all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(
        all_data["GarageYrBlt"].median()
    )
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(
        all_data["MasVnrArea"].median()
    )
    all_data["Electrical"] = all_data["Electrical"].fillna("SBrkr")
    all_data["MSSubClass"] = all_data["MSSubClass"].apply(str)
    all_data["YrSold"] = all_data["YrSold"].astype(str)
    all_data["MoSold"] = all_data["MoSold"].astype(str)

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics = []
    for i in all_data.columns:
        if all_data[i].dtype in numeric_dtypes:
            numerics.append(i)

    all_data.update(all_data[numerics].fillna(0))



    objects = []
    for i in all_data.columns:
        if all_data[i].dtype == object:
            objects.append(i)
    all_data.update(all_data[objects].fillna('None'))

    df_train[all_data.columns] = all_data.iloc[: df_train.shape[0], :]
    df_test[all_data.columns] = all_data.iloc[df_train.shape[0] :, :]
    df_train = df_train.drop(["Utilities", "Street", "PoolQC"], axis=1)
    df_test = df_test.drop(["Utilities", "Street", "PoolQC"], axis=1)
    return {"df_train": df_train, "df_test": df_test}


def remove_outliers(df_train: pd.DataFrame) -> pd.DataFrame:

    Q1 = df_train["SalePrice"].quantile(0.25)
    Q3 = df_train["SalePrice"].quantile(0.75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    outliers = df_train[
        (df_train["SalePrice"] < Q1 - outlier_step)
        | (df_train["SalePrice"] > Q3 + outlier_step)
    ].index
    df_train = df_train.drop(outliers, axis=0)
    print("\t{} samples was removed".format(len(outliers)))
    return df_train


def feature_engineering(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> Dict[str, Any]:

    all_data = pd.concat(
        [df_train.drop(["Id", "SalePrice"], axis=1), df_test.drop(["Id"], axis=1)]
    )

    all_data["haspool"] = all_data["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    all_data["has2ndfloor"] = all_data["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
    all_data["hasgarage"] = all_data["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
    all_data["hasbsmt"] = all_data["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
    all_data["hasfireplace"] = all_data["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)

    all_data = pd.get_dummies(all_data)
    scaler = MinMaxScaler()
    scaler.fit(all_data)
    X = all_data.iloc[: df_train.shape[0], :]
    y = np.log1p(df_train["SalePrice"])
    X_test = all_data.iloc[df_train.shape[0] :, :]
    return {
        "x_train": scaler.transform(X.to_numpy()),
        "y_train": y.to_numpy().reshape(-1,),
        "x_test": scaler.transform(X_test.to_numpy())}
