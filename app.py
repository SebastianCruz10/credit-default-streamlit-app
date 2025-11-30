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

# Nombre de la columna objetivo en tu CSV trabajado
TARGET_COL = "default.payment.next.month"

# Umbral óptimo que obtuviste en el notebook
THRESHOLD = 0.298

# ===================== ESTADO DE SESIÓN =====================

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "feature_names" not in st.session_state:
    st.session_state.feature_names = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "y_pred" not in st.session_state:
    st.session_state.y_pred = None

if "proba" not in st.session_state:
    st.session_state.proba = None

if "y_true" not in st.session_state:
    st.session_state.y_true = None

# ===================== FUNCIONES AUXILIARES =====================

def load_default_data():
    """Carga un dataset por defecto si existe data_processed.csv."""
    try:
        df = pd.read_csv("data_processed.csv")
        return df
    except FileNotFoundError:
        return None

def train_model(df: pd.DataFrame):
    """Entrena el modelo GradientBoosting con hiperparámetros óptimos."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COL}' no está en el dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    model = GradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    feature_names = list(X.columns)
    return model, feature_names

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
        """
        Sube tu archivo `.csv` ya trabajado (con la columna objetivo `default.payment.next.month`)
        o, si no subes nada, se intentará cargar `data_processed.csv` por defecto.
        """
    )

    uploaded_file = st.file_uploader("Sube tu archivo CSV procesado", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ CSV cargado correctamente desde el archivo subido.")
    else:
        if st.session_state.df is None:
            df_default = load_default_data()
            if df_default is not None:
                st.session_state.df = df_default
                st.info("Se ha cargado automáticamente `data_processed.csv` del repositorio.")
            else:
                st.warning("No se ha subido ningún CSV y no se encontró `data_processed.csv`.")

    df = st.session_state.df

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

        if TARGET_COL in df.columns:
            st.subheader("Revisión rápida de la columna objetivo")
            st.write(df[TARGET_COL].head())

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
                - **Azul (0)**: Cliente que no cayó en default  
                - **Naranja (1)**: Cliente que sí cayó en default  
                """
            )

            else:
                st.error(
                    f"La columna objetivo '{TARGET_COL}' no parece ser binaria 0/1.\n\n"
                    f"Valores únicos detectados: {unique_vals}\n\n"
                    "Revisa que estés subiendo el CSV correcto (el ya trabajado, con la variable objetivo como 0 y 1)."
                )
        else:
            st.error(
                f"El dataset no contiene la columna objetivo '{TARGET_COL}'. "
                "Asegúrate de que el CSV trabajado tenga esa columna con ese nombre exacto."
            )

# --------------------------------------------------------
# TAB 2: ENTRENAR MODELO (BAJO DEMANDA)
# --------------------------------------------------------
with tabs[1]:
    st.header("2) Entrenar modelo (Gradient Boosting)")

    df = st.session_state.df

    if df is None:
        st.warning("Primero debes cargar un dataset en la pestaña 'Cargar datos'.")
    elif TARGET_COL not in df.columns:
        st.error(
            f"El dataset cargado no tiene la columna objetivo '{TARGET_COL}'. "
            "Revisa tu CSV."
        )
    else:
        st.markdown(
            """
            Al hacer clic en **'Entrenar modelo'**, se entrenará el modelo
            **GradientBoostingClassifier** con los hiperparámetros óptimos
            definidos en el trabajo.
            """
        )

        if st.button("Entrenar modelo"):
            try:
                model, feature_names = train_model(df)
                st.session_state.model = model
                st.session_state.feature_names = feature_names
                st.success("✅ Modelo entrenado correctamente.")

                # Métricas simples sobre el mismo dataset (solo para info rápida)
                X_all = df[feature_names]
                y_all = df[TARGET_COL]
                proba_all = model.predict_proba(X_all)[:, 1]
                metrics_all, y_pred_all = compute_metrics(y_all, proba_all, THRESHOLD)

                st.subheader("Resumen rápido de métricas (sobre todo el dataset)")
                st.dataframe(
                    pd.Series(metrics_all).to_frame("valor").style.format("{:.4f}")
                )

            except Exception as e:
                st.error(f"Error al entrenar el modelo: {e}")

        if st.session_state.model is not None:
            st.info("Ya hay un modelo entrenado en memoria. Puedes re-entrenar si cargas otro CSV.")

# --------------------------------------------------------
# TAB 3: HACER PREDICCIONES (BAJO DEMANDA)
# --------------------------------------------------------
with tabs[2]:
    st.header("3) Hacer predicciones con el modelo entrenado")

    df = st.session_state.df
    model = st.session_state.model
    feature_names = st.session_state.feature_names

    if df is None:
        st.warning("Primero debes cargar un dataset en la pestaña 'Cargar datos'.")
    elif model is None or feature_names is None:
        st.warning("Primero debes entrenar el modelo en la pestaña 'Entrenar modelo'.")
    elif TARGET_COL not in df.columns:
        st.error(
            f"El dataset cargado no tiene la columna objetivo '{TARGET_COL}'. "
            "Revisa tu CSV."
        )
    else:
        st.markdown(
            """
            Al hacer clic en **'Hacer predicciones'**, se calcularán las probabilidades
            de default para cada registro del dataset cargado, y se generarán las
            métricas globales que luego se mostrarán en la pestaña **'Ver resultados'**.
            """
        )

        if st.button("Hacer predicciones"):
            X_all = df[feature_names]
            y_all = df[TARGET_COL]

            proba_all = model.predict_proba(X_all)[:, 1]
            metrics_all, y_pred_all = compute_metrics(y_all, proba_all, THRESHOLD)

            st.session_state.metrics = metrics_all
            st.session_state.y_pred = y_pred_all
            st.session_state.proba = proba_all
            st.session_state.y_true = y_all

            st.success("✅ Predicciones realizadas y resultados almacenados. Ve a 'Ver resultados'.")

# --------------------------------------------------------
# TAB 4: VER RESULTADOS
# --------------------------------------------------------
with tabs[3]:
    st.header("4) Ver resultados del modelo")

    metrics_all = st.session_state.metrics
    y_pred_all = st.session_state.y_pred
    proba_all = st.session_state.proba
    y_true_all = st.session_state.y_true

    if metrics_all is None or y_pred_all is None or proba_all is None or y_true_all is None:
        st.warning(
            "Primero debes entrenar el modelo (pestaña 2) y luego hacer predicciones "
            "(pestaña 3) para ver los resultados."
        )
    else:
        st.subheader("Métricas generales")
        st.dataframe(pd.Series(metrics_all).to_frame("valor").style.format("{:.4f}"))

        st.subheader("Matriz de confusión")
        plot_confusion_matrix(y_true_all, y_pred_all, title="Matriz de confusión (dataset completo)")

        st.subheader("Distribución de probabilidades de default")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(proba_all, bins=30)
        ax.axvline(THRESHOLD, color="red", linestyle="--", label=f"Umbral = {THRESHOLD:.3f}")
        ax.set_xlabel("Probabilidad predicha de default")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Reporte de clasificación")
        report = classification_report(
            y_true_all,
            y_pred_all,
            target_names=["No default", "Default"],
            zero_division=0
        )
        st.text(report)

        st.markdown(
            """
            **Interpretación resumida:**

            - El modelo logra un **equilibrio** entre detección de clientes en default
              y control de falsos positivos.  
            - La **recall** de la clase "Default" indica qué proporción de los clientes
              morosos se detecta a tiempo.  
            - La **precision** de la clase "Default" muestra qué tan confiable es la
              señal de riesgo cuando el modelo marca a un cliente como riesgoso.  
            - El **F1-score** combina ambas métricas en un único indicador, mientras que
              el **ROC-AUC** refleja la capacidad global de separación entre buenos y
              malos pagadores.
            """
        )







