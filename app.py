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

TARGET_COL = "default_payment_next_month" 

THRESHOLD = 0.298  


# ===================== CARGA DE DATOS =====================

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Carga el dataset procesado desde un CSV.
    Debe incluir la columna objetivo TARGET_COL.
    """
    df = pd.read_csv("data_processed.csv")
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Entrena el modelo final (GradientBoosting) usando los mejores hiperparámetros
    que definiste en el notebook. Esto se cachea para no reentrenar en cada recarga.
    """
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
  
    # Calcula métricas de evaluación dado un vector de probabilidades y un umbral

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
   
   #  Grafica la matriz de confusión como heatmap.
   
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


# ===================== INICIALIZACIÓN =====================

df = load_data()

if TARGET_COL not in df.columns:
    st.error(
        f"El dataset 'data_processed.csv' no contiene la columna objetivo '{TARGET_COL}'. "
        f"Ajústala en la constante TARGET_COL en app.py."
    )
    st.stop()

try:
    model, feature_names = train_model(df)
except Exception as e:
    st.error(f"Error al entrenar el modelo en la nube: {e}")
    st.stop()

X_all = df[feature_names]
y_all = df[TARGET_COL]


# ===================== PESTAÑAS PRINCIPALES =====================

tabs = st.tabs([
    "1) Datos (.csv)",
    "2) Entrenamiento del modelo",
    "3) Predicciones y métricas",
    "4) Análisis de resultados"
])


# --------------------------------------------------------
# TAB 1: PRESENTACIÓN E INTERPRETACIÓN DE LOS DATOS
# --------------------------------------------------------
with tabs[0]:
    st.header("1) Presentación e interpretación de los datos (.csv)")

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

    st.subheader("Descripción estadística de variables numéricas")
    st.dataframe(df.describe())

    st.subheader("Distribución de la variable objetivo")
    target_counts = df[TARGET_COL].value_counts().sort_index()
    st.bar_chart(target_counts)

    st.markdown(
        """
        - **0**: Cliente que no cayó en default (buen pagador).  
        - **1**: Cliente que cayó en default (mal pagador).  

        Esta distribución evidencia el **desbalance de clases**, donde la mayoría
        de clientes no entra en default.
        """
    )


# --------------------------------------------------------
# TAB 2: ENTRENAMIENTO DEL MODELO
# --------------------------------------------------------
with tabs[1]:
    st.header("2) Entrenamiento del modelo")

    st.markdown(
        """
        En esta sección se describe brevemente cómo se entrenó el modelo:

        1. **Preparación de datos**  
           - Se partió del dataset de clientes de tarjeta de crédito.  
           - Se seleccionaron las variables relevantes y se definió la variable objetivo
             (`{TARGET_COL}`).  

        2. **Manejo del desbalance de clases**  
           - Se evaluaron diferentes técnicas de *sampling* como SMOTE, RandomUnderSampler, etc.  
           - Para el modelo de **Gradient Boosting**, la mejor configuración se obtuvo
             sin aplicar sampling adicional (tratando los datos originales tal cual).  

        3. **Comparación de modelos**  
           - Se compararon modelos como Regresión Logística, Random Forest, XGBoost
             y Gradient Boosting, utilizando **GridSearchCV** con validación cruzada.  
           - La métrica objetivo fue **ROC-AUC**, debido al desbalance entre clases.  
           - El modelo ganador fue **GradientBoostingClassifier** con hiperparámetros
             ajustados.

        4. **Ajuste del umbral de decisión**  
           - A partir de las probabilidades de salida del modelo, se analizó la curva
             precision–recall.  
           - Se seleccionó un umbral óptimo que maximiza el **F1-score** para la clase
             de clientes que entran en default.
        """
    )

    st.subheader("Hiperparámetros del modelo en esta app")
    st.code(
        "GradientBoostingClassifier(\n"
        "    learning_rate=0.05,\n"
        "    max_depth=3,\n"
        "    n_estimators=200,\n"
        "    random_state=42\n"
        ")",
        language="python"
    )

    st.info(f"Umbral de decisión utilizado actualmente: **{THRESHOLD:.3f}**")

    st.markdown(
        """
        > Nota: El GridSearch completo se ejecutó previamente en un notebook.
        > En esta app se utiliza el modelo ya definido con los mejores hiperparámetros
        > encontrados, y se entrena de nuevo directamente a partir del dataset cargado.
        """
    )


# --------------------------------------------------------
# TAB 3: PREDICCIONES Y MÉTRICAS
# --------------------------------------------------------
with tabs[2]:
    st.header("3) Hacer las predicciones de riesgo (Métricas)")

    st.markdown(
        """
        En esta pestaña puedes evaluar el modelo:
        - Usando el dataset base (`data_processed.csv`), o  
        - Subiendo un archivo `.csv` con la **misma estructura** (incluyendo la columna objetivo).
        """
    )

    option = st.radio(
        "Selecciona la fuente de datos para evaluar:",
        ["Usar data_processed.csv", "Subir un archivo CSV propio"]
    )

    df_eval = None

    if option == "Usar data_processed.csv":
        df_eval = df.copy()
    else:
        uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
        if uploaded_file is not None:
            df_eval = pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos cargados:")
            st.dataframe(df_eval.head())
        else:
            st.info("Sube un archivo CSV para continuar.")

    if df_eval is not None:
        if TARGET_COL not in df_eval.columns:
            st.error(
                f"El archivo no contiene la columna objetivo '{TARGET_COL}'. "
                "No se pueden calcular métricas sin la variable real."
            )
        else:
            missing_features = [c for c in feature_names if c not in df_eval.columns]
            if missing_features:
                st.error(
                    "El archivo no tiene exactamente las mismas columnas de características "
                    "que el modelo espera. Faltan: " + ", ".join(missing_features)
                )
            else:
                X_eval = df_eval[feature_names]
                y_eval = df_eval[TARGET_COL]

                proba_eval = model.predict_proba(X_eval)[:, 1]
                metrics_eval, y_pred_eval = compute_metrics(y_eval, proba_eval, THRESHOLD)

                st.subheader("Métricas de evaluación")
                st.dataframe(
                    pd.Series(metrics_eval).to_frame("valor").style.format("{:.4f}")
                )

                st.markdown(
                    """
                    - **accuracy**: proporción de predicciones correctas.  
                    - **precision**: de los clientes que el modelo marcó como "default",
                      cuántos realmente lo fueron.  
                    - **recall**: de todos los clientes que entraron en default,
                      cuántos detectó el modelo.  
                    - **f1**: balance entre precision y recall.  
                    - **roc_auc**: capacidad del modelo para discriminar entre buenos
                      y malos pagadores.
                    """
                )

                st.subheader("Matriz de confusión")
                plot_confusion_matrix(y_eval, y_pred_eval, title="Matriz de confusión (datos de evaluación)")

                st.subheader("Distribución de probabilidades de default")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(proba_eval, bins=30)
                ax.axvline(THRESHOLD, color="red", linestyle="--", label=f"Umbral = {THRESHOLD:.3f}")
                ax.set_xlabel("Probabilidad predicha de default")
                ax.set_ylabel("Frecuencia")
                ax.legend()
                st.pyplot(fig)


# --------------------------------------------------------
# TAB 4: ANÁLISIS DE LOS RESULTADOS
# --------------------------------------------------------
with tabs[3]:
    st.header("4) Análisis de los resultados")

    st.markdown(
        """
        Aquí se muestran las métricas y la matriz de confusión calculadas sobre todo
        el dataset base, junto con un breve análisis cualitativo del modelo.
        """
    )

    proba_all = model.predict_proba(X_all)[:, 1]
    metrics_all, y_pred_all = compute_metrics(y_all, proba_all, THRESHOLD)

    st.subheader("Métricas generales sobre el dataset completo")
    st.dataframe(pd.Series(metrics_all).to_frame("valor").style.format("{:.4f}"))

    st.subheader("Matriz de confusión global")
    plot_confusion_matrix(y_all, y_pred_all, title="Matriz de confusión (dataset completo)")

    st.subheader("Reporte de clasificación")
    report = classification_report(
        y_all,
        y_pred_all,
        target_names=["No default", "Default"],
        zero_division=0
    )
    st.text(report)

    st.markdown(
        """
        **Interpretación resumida:**

        - El modelo logra un **buen equilibrio** entre la detección de clientes
          que entran en default y el control de falsos positivos.  
        - La **recall** de la clase "Default" indica qué proporción de los clientes
          morosos se detecta a tiempo.  
        - La **precision** de la clase "Default" muestra qué tan confiable es la
          señal de riesgo cuando el modelo marca a un cliente como riesgoso.  
        - El **F1-score** combina ambas métricas en un único indicador, mientras que
          el **ROC-AUC** refleja la capacidad global de separación entre buenos y
          malos pagadores.

        En el contexto del banco, este modelo puede utilizarse para:
        - Apoyar decisiones de aprobación o ampliación de líneas de crédito.  
        - Priorizar clientes para monitoreo o acciones preventivas.  
        - Reducir la probabilidad de pérdida esperada por incumplimiento,
          manteniendo un nivel razonable de aprobación para buenos clientes.
        """
    )



