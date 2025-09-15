import os
import io
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🧪 Inferencia — Modelo Final", layout="wide")
st.title("🧪 Inferencia — Modelo Final Guardado")
st.caption("Este front-end carga el mejor modelo guardado en `models/final_model.pkl` (o un GridSearch con `refit=True`).")


# --------------------------
# Utilidades
# --------------------------
def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def get_inference_estimator(model_obj):
    # GridSearchCV/RandomizedSearchCV con refit=True delegan a best_estimator_
    if hasattr(model_obj, "best_estimator_") and getattr(model_obj, "best_estimator_", None) is not None:
        return model_obj  # usar directamente: predict/predict_proba funcionan
    return model_obj


def infer_expected_columns(model_obj) -> Optional[List[str]]:
    """
    Intenta deducir columnas de entrada:
    1) Si es Pipeline y tiene un paso tipo ColumnTransformer (propiedad .transformers_) → columnas declaradas.
    2) Si el estimador expone feature_names_in_ → usar eso.
    """
    est = model_obj
    if hasattr(est, "best_estimator_") and est.best_estimator_ is not None:
        est = est.best_estimator_

    # Caso 1: buscar ColumnTransformer en pipeline
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(est, Pipeline):
            for _, step in est.named_steps.items():
                if hasattr(step, "transformers_"):  # ColumnTransformer u objetos compatibles
                    cols = []
                    try:
                        for name, trans, cols_sel in step.transformers_:
                            if cols_sel is None or cols_sel == "drop":
                                continue
                            if isinstance(cols_sel, (list, tuple, np.ndarray)):
                                cols.extend(list(cols_sel))
                    except Exception:
                        pass
                    if cols:
                        seen = set()
                        ordered = [c for c in cols if not (c in seen or seen.add(c))]
                        return ordered
    except Exception:
        pass

    # Caso 2: feature_names_in_
    try:
        if hasattr(est, "feature_names_in_"):
            cols = list(est.feature_names_in_)
            if cols:
                return cols
    except Exception:
        pass

    return None


def coerce_schema(df_input: pd.DataFrame, schema_cols: List[str]) -> pd.DataFrame:
    df = df_input.copy()
    for c in schema_cols:
        if c not in df.columns:
            df[c] = np.nan
    extra = [c for c in df.columns if c not in schema_cols]
    if extra:
        df = df.drop(columns=extra)
    return df[schema_cols]


def predict_any(model_obj, X_df: pd.DataFrame, threshold: float = 0.5):
    est = get_inference_estimator(model_obj)

    # Preferimos predict_proba si existe
    score = None
    proba = None

    if hasattr(est, "predict_proba"):
        try:
            proba = est.predict_proba(X_df)
            if proba is not None and hasattr(proba, "shape") and proba.shape[1] >= 2:
                score = proba[:, 1]
            else:
                score = np.asarray(proba).reshape(-1)
        except Exception:
            proba = None
            score = None

    if score is None and hasattr(est, "decision_function"):
        try:
            dfun = est.decision_function(X_df)
            score = 1.0 / (1.0 + np.exp(-np.asarray(dfun)))  # sigmoid
        except Exception:
            pass

    if score is None and hasattr(est, "predict"):
        pred = est.predict(X_df)
        score = np.asarray(pred).astype(float)

    if score is None:
        raise RuntimeError("El estimador no implementa predict/predict_proba/decision_function.")

    pred = (np.asarray(score, dtype=float) >= float(threshold)).astype(int)
    return pred, score


# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("⚙️ Configuración")

default_candidates = [
    "../models/final_model.pkl",  # si lanzas desde app/
    "models/final_model.pkl",     # si lanzas desde raíz
    "./final_model.pkl",          # fallback
]
default_model_path = first_existing(default_candidates) or "../models/final_model.pkl"

model_path = st.sidebar.text_input("Ruta del modelo final (.pkl)", default_model_path)
threshold = st.sidebar.slider("Umbral de clasificación", 0.0, 1.0, 0.5, 0.01)
data_mode = st.sidebar.radio("Fuente de datos", ["Subir CSV", "Introducir a mano"], index=0)
sep_choice = st.sidebar.selectbox("Separador CSV", [",", ";", "\\t"], index=0)


# --------------------------
# Carga modelo
# --------------------------
if not os.path.exists(model_path):
    st.error(f"No encuentro el modelo en: {model_path}")
    st.stop()

try:
    model_obj = load_model(model_path)
except Exception as e:
    st.error(f"Error cargando el modelo: {e}")
    st.stop()

schema_cols = infer_expected_columns(model_obj)
if schema_cols is None:
    st.warning(
        "No pude inferir las columnas esperadas del modelo. "
        "Asegúrate de subir un CSV con el mismo esquema que el entrenamiento."
    )

with st.expander("ℹ️ Información del modelo", expanded=False):
    st.write(type(model_obj).__name__)
    if hasattr(model_obj, "best_params_"):
        st.json({"best_params_": getattr(model_obj, "best_params_", {})})
    if hasattr(model_obj, "best_score_"):
        st.write("best_score_:", getattr(model_obj, "best_score_", None))
    if hasattr(model_obj, "classes_"):
        st.write("classes_:", getattr(model_obj, "classes_", None))


# --------------------------
# Entrada de datos
# --------------------------
def get_input_dataframe():
    if data_mode == "Subir CSV":
        up = st.file_uploader(
            "Sube tu CSV **con las columnas de entrenamiento** (sin el target)",
            type=["csv"]
        )
        if up is None:
            return None
        try:
            sep = "\t" if sep_choice == "\\t" else sep_choice
            df = pd.read_csv(up, sep=sep)
            return df
        except Exception as e:
            st.error(f"No pude leer el CSV: {e}")
            return None
    else:
        if schema_cols is None:
            st.info("Como no se pudo inferir el esquema, pega un CSV mínimo (con cabecera) en el cuadro de abajo.")
            csv_text = st.text_area("Pega aquí un CSV mínimo (con cabecera)", height=180, value="")
            if not csv_text.strip():
                return None
            try:
                df = pd.read_csv(io.StringIO(csv_text))
                return df
            except Exception as e:
                st.error(f"CSV inválido: {e}")
                return None
        else:
            template = pd.DataFrame([{c: "" for c in schema_cols}])
            edited = st.data_editor(template, num_rows="dynamic", use_container_width=True, height=320)
            df = edited.replace("", np.nan)
            if df.dropna(how="all").empty:
                return None
            return df


df_input = get_input_dataframe()
if df_input is None:
    st.stop()

if schema_cols is not None:
    df_features = coerce_schema(df_input, schema_cols)
else:
    df_features = df_input.copy()

st.write("### Vista previa de entrada")
st.dataframe(df_features.head(10), use_container_width=True)

if st.button("🚀 Predecir"):
    try:
        y_pred, y_score = predict_any(model_obj, df_features, threshold=threshold)

        out = df_features.copy()
        out["score"] = y_score
        out["pred"] = y_pred

        st.success("✅ Predicción completada")
        st.dataframe(out.head(50), use_container_width=True)

        csv_buf = io.StringIO()
        out.to_csv(csv_buf, index=False)
        st.download_button(
            "💾 Descargar predicciones (CSV)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="predicciones.csv",
            mime="text/csv",
        )

        st.write("### Resumen")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total filas", len(out))
        with c2:
            st.metric("Positivos (1)", int(out["pred"].sum()))

        st.write("### Histograma de scores")
        st.bar_chart(pd.DataFrame({"score": out["score"]})["score"])
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")
        st.exception(e)
