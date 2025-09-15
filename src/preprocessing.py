"""
Funciones de preprocesado para notebooks.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

def build_preprocessor(
    num_features: List[str],
    cat_features: List[str],
    scaler: str = "standard",
    impute_strategy_num: str = "median",
    impute_strategy_cat: str = "most_frequent",
    onehot_drop: Optional[str] = "if_binary",
) -> ColumnTransformer:
    """
    Construye un ColumnTransformer estándar para numéricas y categóricas.
    """
    if scaler == "standard":
        scaler_step = StandardScaler()
    elif scaler == "minmax":
        scaler_step = MinMaxScaler()
    elif scaler is None:
        scaler_step = "passthrough"
    else:
        raise ValueError(f"Scaler no soportado: {scaler}")

    num_pipeline = [
        ("imputer", SimpleImputer(strategy=impute_strategy_num)),
        ("scaler", scaler_step),
    ]
    if scaler_step == "passthrough":
        num_pipeline = [("imputer", SimpleImputer(strategy=impute_strategy_num))]

    cat_pipeline = [
        ("imputer", SimpleImputer(strategy=impute_strategy_cat)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=onehot_drop, sparse_output=False)),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", __to_pipeline(num_pipeline), num_features or []),
            ("cat", __to_pipeline(cat_pipeline), cat_features or []),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def __to_pipeline(steps):
    from sklearn.pipeline import Pipeline
    return Pipeline(steps)
