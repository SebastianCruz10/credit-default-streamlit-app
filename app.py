import streamlit as st
import pandas as pd
import joblib
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# =============== CONFIGURACIÓN BÁSICA ===============
st.set_page_config(
    page_title="Riesgo de Crédito - Default de Tarjeta",
    layout="wide"
)

st.title("Modelo Predictivo de Riesgo de Crédito (Default de Tarjeta)")
st.markdown("Aplicación en Streamlit basada en el dataset **Default of Credit Card Clients**.")


# =============== FUNCIONES AUXILIARES ===============

@st.cache_data
def load_data():
    df = pd.read_csv("data_processed.csv")
    return df

@st.cache_resource
def load_model_and_config():
    model = joblib.load("best_model.pkl")
    with open("config_streamlit.json", "r") as f:
        config = json.load(f)
    threshold = float(config["threshold"])
    return model, threshold

def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }, y_pred

def plot_confusion_matrix(y_true, y_pred):
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
    ax.set_title("Matriz de confusión")
    st.pyplot(fig)


# Cargar datos y modelo
df = load_data()
model, threshold = load_model_and_config()

# IMPORTANTE: ajusta este nombre al de tu columna objetivo real
TARGET_COL = "default_payment_next_month"  # <-- cambia si es distinto

if TARGET_COL not in df.columns:
    st.error(f"La columna objetivo '{TARGET_COL}' no se encuentra en data_processed.csv. Ajusta TARGET_COL en app.py.")
else:
    X_all = df.drop(columns=[TARGET_COL])
    y_all = df[TARGET_COL]


# =============== PESTAÑAS PRINCIPALES ===============
tabs = st.tabs([
    "1) Datos (.csv)",
    "2) Entrenamiento del modelo",
    "3) Predicciones y métricas",
    "4) Análisis de resultados"
])

# ----------------------------------------------------
# TAB 1: PRESENTACIÓN E INTERPRETACIÓN DE LOS DATOS
# ----------------------------------------------------
with tabs[0]:
    st.header("1) Presentación e interpretación de los datos (.csv)")

    st.subheader("Vista general de los datos")
    st.write(f"Número de filas: **{df.shape[0]}**")
    st.write(f"Número de columnas: **{df.shape[1]}**")
    st.dataframe(df.head())

    st.subheader("Descripción estadística de las variables numéricas")
    st.dataframe(df.describe())

    if TARGET_COL in df.columns:
        st.subheader("Distribución de la variable objetivo")
        st.bar_chart(df[TARGET_COL].value_counts())

        st.markdown("""
        - **0**: Cliente que no cayó en default (buen pagador).  
        - **1**: Cliente que cayó en default (mal pagador).  
        La distribución muestra el desbalance de clases: la mayoría de clientes no entra en default.
        """)


# ----------------------------------------------------
# TAB 2: ENTRENAMIENTO DEL MODELO
# (aquí explicamos cómo se entrenó y mostramos resumen)
# ----------------------------------------------------
with tabs[1]:
    st.header("2) Entrenamiento del modelo")

    st.markdown("""
    En esta sección se describe el proceso de entrenamiento realizado **previamente** en un Jupyter Notebook:
    
    1. **Preparación de datos**:
       - Limpieza del dataset original de clientes de tarjeta de crédito.
       - Selección de variables relevantes.
       - Separación en matriz de características (X) y variable objetivo (y).

    2. **Manejo de desbalance de clases**:
       - Evaluación de diferentes técnicas de *sampling* (por ejemplo, SMOTE, RandomUnderSampler, `passthrough`).
       - En el modelo ganador (Gradient Boosting), la mejor configuración se obtuvo con `samp = "passthrough"`.

    3. **Entrenamiento y selección de modelo**:
       - Se entrenaron y compararon varios modelos (Regresión Logística, Random Forest, XGBoost, Gradient Boosting, etc.).
       - Se utilizó **GridSearchCV** con validación cruzada, optimizando la métrica **ROC-AUC**.
       - El modelo ganador fue **GradientBoosting**, con un ROC-AUC promedio cercano a **0.79–0.80**.

    4. **Ajuste del umbral de decisión**:
       - A partir de las probabilidades del modelo, se analizó la curva precision–recall.
       - Se seleccionó un **umbral óptimo** que maximiza el **F1-score** para la clase de clientes que entran en default.
    """)

    st.info(f"Umbral de decisión utilizado actualmente en la app: **{threshold:.3f}**")

    st.markdown("""
    > Nota: El entrenamiento completo (GridSearch, validación cruzada) se ejecutó fuera de Streamlit para optimizar tiempo de cómputo.  
    > En esta app se carga el **modelo ya entrenado** (`best_model.pkl`) para realizar predicciones y análisis.
    """)


# ----------------------------------------------------
# TAB 3: PREDICCIONES Y MÉTRICAS (SOBRE UN ARCHIVO O SOBRE df)
# ----------------------------------------------------
with tabs[2]:
    st.header("3) Hacer las predicciones de riesgo (Métricas)")

    st.markdown("""
    Puedes:
    - Usar el mismo dataset procesado para calcular métricas globales, o  
    - Subir un `.csv` con la misma estructura para evaluar el modelo sobre nuevos datos.
    """)

    option = st.radio(
        "Selecciona la fuente de datos para evaluar:",
        ["Usar data_processed.csv", "Subir un archivo CSV propio"]
    )

    if option == "Usar data_processed.csv":
        df_eval = df.copy()
    else:
        uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
        if uploaded_file is not None:
            df_eval = pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos cargados:")
            st.dataframe(df_eval.head())
        else:
            df_eval = None

    if df_eval is not None:
        if TARGET_COL not in df_eval.columns:
            st.error(f"El archivo no contiene la columna objetivo '{TARGET_COL}'.")
        else:
            X_eval = df_eval.drop(columns=[TARGET_COL])
            y_eval = df_eval[TARGET_COL]

            # Predicciones
            proba_eval = model.predict_proba(X_eval)[:, 1]
            metrics_eval, y_pred_eval = compute_metrics(y_eval, proba_eval, threshold)

            st.subheader("Métricas de evaluación")
            st.dataframe(pd.Series(metrics_eval).to_frame("valor").style.format("{:.4f}"))

            st.markdown("""
            - **accuracy**: proporción de predicciones correctas.  
            - **precision**: de los clientes que el modelo marcó como "default", cuántos realmente lo fueron.  
            - **recall**: de todos los clientes que entraron en default, cuántos detectó el modelo.  
            - **f1**: balance entre precision y recall.  
            - **roc_auc**: capacidad del modelo para discriminar entre buenos y malos pagadores.
            """)

            st.subheader("Matriz de confusión")
            plot_confusion_matrix(y_eval, y_pred_eval)


# ----------------------------------------------------
# TAB 4: ANÁLISIS DE LOS RESULTADOS
# ----------------------------------------------------
with tabs[3]:
    st.header("4) Análisis de los resultados")

    st.markdown("""
    En esta sección se presenta un análisis cualitativo del desempeño del modelo.
    """)

    # Reutilizamos las métricas calculadas sobre todo el df
    if TARGET_COL in df.columns:
        proba_all = model.predict_proba(X_all)[:, 1]
        metrics_all, y_pred_all = compute_metrics(y_all, proba_all, threshold)

        st.subheader("Métricas generales sobre el dataset completo")
        st.dataframe(pd.Series(metrics_all).to_frame("valor").style.format("{:.4f}"))

        st.subheader("Matriz de confusión global")
        plot_confusion_matrix(y_all, y_pred_all)

        st.subheader("Reporte de clasificación (texto)")
        report = classification_report(y_all, y_pred_all, target_names=["No default", "Default"])
        st.text(report)

        st.markdown("""
        **Interpretación resumida:**

        - El modelo logra un buen **equilibrio** entre la identificación de clientes que entran en default y el control de falsos positivos.
        - La **recall** de la clase "Default" indica qué proporción de los clientes morosos se detecta a tiempo.
        - La **precision** de la clase "Default" indica cuántos de los marcados como riesgosos realmente terminan en default.
        - El **F1-score** resume ese balance, y el **ROC-AUC** refleja la capacidad global de separación entre clases.

        En el contexto del banco, este modelo puede utilizarse para:
        - Apoyar decisiones de otorgamiento o ampliación de líneas de crédito.
        - Priorizar clientes para monitoreo o campañas de prevención.
        - Reducir la probabilidad de pérdida por incumplimiento, manteniendo un nivel razonable de aprobación a buenos clientes.
        """)

    else:
        st.error(f"No se pudo encontrar la columna objetivo '{TARGET_COL}' en el dataset.")


