"""
Funciones de entrenamiento para notebooks.
"""
from typing import Dict, Optional, Any
import os
import joblib
from sklearn.model_selection import GridSearchCV

def train_with_gridsearch(
    pipeline,
    X_train,
    y_train,
    param_grid: Dict[str, list],
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    refit: bool = True,
    verbose: int = 1,
):
    """
    Entrena un GridSearchCV sobre un pipeline dado y lo ajusta.
    Devuelve el objeto GridSearchCV ya entrenado (search).
    """
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)
    return gs

def save_model(model_obj: Any, path: str = "../models/final_model.pkl") -> str:
    """
    Guarda el objeto modelo (pipeline o GridSearchCV) en un .pkl.
    Crea el directorio si no existe. Devuelve la ruta final.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model_obj, path)
    return path
