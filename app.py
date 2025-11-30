import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

# ===================== CONFIGURACIÓN BÁSICA =====================

st.set_page_config(
    page_title="Riesgo de Crédito - Default de Tarjeta",
    layout="wide"
)

st.title("Modelo Predictivo de Riesgo de Crédito (Default de Tarjeta de Crédito)")
st.markdown(
    "Aplicación en Streamlit basada en el dataset **Default of Credit Card Clients**."
)

# ===================== PARÁMETROS GLOBALES =====================

TARGET_COL = "default.payment.next.month"

# mismas features que en el notebook
FEATURES = [
    "meses_con_atraso",
    "PAY_0",
    "meses_sin_pago",
    "LIMIT_BAL",
    "pago_promedio_6m",
    "volatilidad_facturacion"
]

# Umbral óptimo obtenido en el notebook
THRESHOLD = 0.298

# ===================== ESTADO DE SESIÓN =====================

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "y_pred" not in st.session_state:
    st.session_state.y_pred = None

if "proba" not in st.session_state:
    st.session_state.proba = None

# ===================== FUNCIONES AUXILIARES =====================

def load_default_data():
    """Carga un dataset por defecto si existe data_processed.csv."""
    try:
        df = pd.read_csv("data_processed.csv")
        return df
    except FileNotFoundError:
        return None

def train_model_with_split(df: pd.DataFrame):
    """
    Replica el flujo del notebook:
    - Usa solo las FEATURES seleccionadas.
    - Hace train_test_split con test_size=0.20, random_state=42, stratify=y.
    - Entrena GradientBoostingClassifier con los hiperparámetros óptimos.
    """
    missing_cols = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Faltan columnas necesarias en el dataset: " + ", ".join(missing_cols)
        )

    X = df[FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = GradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X_test, y_test

def compute_metrics(y_true, y_proba, threshold: float):
    """Calcula métricas de evaluación dado un vector de probabilidades y un umbral."""
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return metrics, y_pred

def plot_confusion_matrix(y_true, y_pred, title="Matriz de confusión"):
    """Grafica la matriz de confusión como heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No default (predicho)", "Default (predicho)"],
        yticklabels=["No default (real)", "Default (real)"],
        ax=ax
    )
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    st.pyplot(fig)

# ===================== PESTAÑAS PRINCIPALES =====================

tabs = st.tabs([
    "1) Cargar datos",
    "2) Entrenar modelo",
    "3) Hacer predicciones",
    "4) Ver resultados"
])

# --------------------------------------------------------
# TAB 1: CARGAR DATOS Y VER CLASES
# --------------------------------------------------------
with tabs[0]:
    st.header("1) Cargar datos (.csv) e interpretar clases")

    st.markdown(
        f"""
        Sube tu archivo `.csv` ya trabajado (derivado de `df_modelo` en el notebook),
        que contenga las columnas:

        - Variables: {", ".join(FEATURES)}
        - Objetivo: `{TARGET_COL}`

        Si no subes nada, se intentará cargar `data_processed.csv` por defecto.
        """
    )

    uploaded_file = st.file_uploader("Sube tu archivo CSV procesado", type=["csv"])

    # --- Carga del CSV ---
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ CSV cargado correctamente desde el archivo subido.")
    else:
        if st.session_state.df is None:
            df_default = load_default_data()
            if df_default is not None:
                st.session_state.df = df_default
                st.info("📁 Se ha cargado automáticamente `data_processed.csv` del repositorio.")
            else:
                st.warning("⚠️ No se ha subido ningún CSV y no se encontró `data_processed.csv`.")

    df = st.session_state.df

    # --- Mostrar dataset si existe ---
    if df is not None:
        st.subheader("Vista general del dataset")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Número de filas: **{df.shape[0]}**")
            st.write(f"Número de columnas: **{df.shape[1]}**")

        with col2:
            st.write("Columnas del dataset:")
            st.write(list(df.columns))

        st.subheader("Primeras filas del dataset")
        st.dataframe(df.head())

        st.subheader("Resumen estadístico (describe)")
        st.dataframe(df.describe())

        # --- Validación del target ---
        if TARGET_COL in df.columns:

            st.subheader("Distribución de la variable objetivo (tabla)")
            target_counts = df[TARGET_COL].value_counts().sort_index()
            st.write(target_counts)

            unique_vals = sorted(df[TARGET_COL].dropna().unique().tolist())

            if set(unique_vals).issubset({0, 1}):

                st.subheader("Distribución de la variable objetivo (gráfico)")

                dist_df = target_counts.reset_index()
                dist_df.columns = ["Clase", "Conteo"]

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#1f77b4", "#ff7f0e"]  # azul para 0, naranja para 1

                ax.bar(dist_df["Clase"].astype(str), dist_df["Conteo"], color=colors)
                ax.set_xlabel("Clase")
                ax.set_ylabel("Cantidad")
                ax.set_title("Distribución de la variable objetivo")

                for i, v in enumerate(dist_df["Conteo"]):
                    ax.text(i, v + max(dist_df["Conteo"])*0.02, str(v), ha="center")

                st.pyplot(fig)

                st.markdown(
                    """
                    **Interpretación:**

                    - 🟦 **Azul (0)** → Cliente que **NO** cayó en default  
                    - 🟧 **Naranja (1)** → Cliente que **SÍ** cayó en default  

                    Se observa claramente el **desbalance de clases**: predominan los clientes que no entran en default.
                    """
                )

            else:
                st.error(
                    f"❌ La columna objetivo '{TARGET_COL}' no parece ser binaria 0/1.\n\n"
                    f"Valores únicos detectados: {unique_vals}\n\n"
                    "Revisa que estés subiendo el CSV correcto (el ya procesado, con 0 y 1)."
                )

        else:
            st.error(
                f"❌ El dataset no contiene la columna objetivo '{TARGET_COL}'.\n"
                "Asegúrate de que el CSV procesado tenga esa columna con ese nombre exacto."
            )

# --------------------------------------------------------
# TAB 2: ENTRENAR MODELO (REPLICANDO NOTEBOOK)
# --------------------------------------------------------
with tabs[1]:
    st.header("2) Entrenar modelo (Gradient Boosting, como en el notebook)")

    df = st.session_state.df

    if df is None:
        st.warning("Primero debes cargar un dataset en la pestaña 'Cargar datos'.")
    else:
        st.markdown(
            """
            Al hacer clic en **'Entrenar modelo'**, se realizará:

            1. **Selección de variables**: se usan solo las features definidas en el notebook.  
            2. **Train-test split**: 80% train, 20% test, `random_state=42`, `stratify=y`.  
            3. **Entrenamiento** de `GradientBoostingClassifier` con los mejores hiperparámetros.  

            Esto replica el flujo del notebook para que las métricas sean comparables.
            """
        )

        if st.button("Entrenar modelo"):
            try:
                model, X_test, y_test = train_model_with_split(df)
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                st.success("✅ Modelo entrenado y conjunto de prueba almacenado (X_test, y_test).")

                st.info(
                    f"Dimensiones de X_test: {X_test.shape}. "
                    f"Cantidad de ejemplos en el test: {len(y_test)}."
                )

            except Exception as e:
                st.error(f"Error al entrenar el modelo: {e}")

        if st.session_state.model is not None and st.session_state.X_test is not None:
            st.info("Ya hay un modelo entrenado y un conjunto de prueba listo. Puedes re-entrenar si cambias el CSV.")

# --------------------------------------------------------
# TAB 3: HACER PREDICCIONES (SOBRE X_test / y_test)
# --------------------------------------------------------
with tabs[2]:
    st.header("3) Hacer predicciones sobre el conjunto de prueba")

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    if model is None or X_test is None or y_test is None:
        st.warning(
            "Primero debes entrenar el modelo en la pestaña 'Entrenar modelo' "
            "para generar X_test e y_test."
        )
    else:
        st.markdown(
            f"""
            Al hacer clic en **'Hacer predicciones'**, se calcularán:

            - Probabilidades de default sobre **X_test**  
            - Métricas usando el umbral óptimo: **{THRESHOLD:.3f}**  
            - Estas métricas y la matriz de confusión serán visibles en la pestaña **'Ver resultados'**.
            """
        )

        if st.button("Hacer predicciones"):
            proba_test = model.predict_proba(X_test)[:, 1]
            metrics_test, y_pred_test = compute_metrics(y_test, proba_test, THRESHOLD)

            st.session_state.metrics = metrics_test
            st.session_state.y_pred = y_pred_test
            st.session_state.proba = proba_test
            st.session_state.y_test = y_test  # aseguramos que quede almacenado

            st.success("✅ Predicciones realizadas sobre el conjunto de prueba. Ve a 'Ver resultados'.")

# --------------------------------------------------------
# TAB 4: VER RESULTADOS (SOBRE TEST)
# --------------------------------------------------------
with tabs[3]:
    st.header("4) Ver resultados (conjunto de prueba, como en el notebook)")

    metrics_test = st.session_state.metrics
    y_pred_test = st.session_state.y_pred
    proba_test = st.session_state.proba
    y_test = st.session_state.y_test

    if metrics_test is None or y_pred_test is None or proba_test is None or y_test is None:
        st.warning(
            "Primero debes entrenar el modelo (pestaña 2) y luego hacer predicciones "
            "(pestaña 3) para ver los resultados."
        )
    else:
        st.subheader("Métricas en el conjunto de prueba (X_test / y_test)")
        st.dataframe(pd.Series(metrics_test).to_frame("valor").style.format("{:.4f}"))

        st.subheader("Matriz de confusión (test)")
        plot_confusion_matrix(y_test, y_pred_test, title="Matriz de confusión - Conjunto de prueba")

        st.subheader("Distribución de probabilidades de default (test)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(proba_test, bins=30)
        ax.axvline(THRESHOLD, color="red", linestyle="--", label=f"Umbral = {THRESHOLD:.3f}")
        ax.set_xlabel("Probabilidad predicha de default")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Reporte de clasificación (test)")
        report = classification_report(
            y_test,
            y_pred_test,
            target_names=["No default", "Default"],
            zero_division=0
        )
        st.text(report)

        st.markdown(
            """
            **Nota:**  
            Estos resultados están calculados **solo sobre el conjunto de prueba (20% del dataset)**,
            usando el mismo esquema de partición y el mismo tipo de modelo que en el notebook.
            De esa forma, las métricas y la matriz de confusión son directamente comparables.
            """
        )











