# ============================================================
# 🤖 Heart Failure Prediction — Dashboard Streamlit
# Archivo: 03_streamlit_app/app.py
#
# Estructura:
#   Tab 1 — Exploración:  EDA interactivo del dataset
#   Tab 2 — Predicción:   Formulario clínico → predicción del modelo
#   Tab 3 — Modelo:       Métricas, curvas ROC, feature importance
#
# Nota Streamlit: el script se re-ejecuta completo en cada
# interacción del usuario. El flujo es siempre lineal.
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# ── Configuración de página ──────────────────────────────────
# Debe ser la primera instrucción de Streamlit en el script.
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="🤖",
    layout="wide"
)

# ── Rutas ────────────────────────────────────────────────────
# Path(__file__).parent.parent sube un nivel desde 03_streamlit_app/
# hasta la raíz del proyecto, donde están models/ y plots/.
BASE_DIR = Path(__file__).parent.parent

# ── Carga de artefactos ──────────────────────────────────────
# @st.cache_resource: para objetos no serializables (modelos, conexiones).
# @st.cache_data:     para DataFrames y objetos serializables.
# Ambos evitan recargar desde disco en cada re-ejecución.

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / 'models' / 'best_model_gradient_boosting.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / 'heart_processed.csv')
    feature_names = [col for col in df.columns if col != 'HeartDisease']
    return df, feature_names

model = load_model()
df, feature_names = load_data()

# ── Header ───────────────────────────────────────────────────
st.title("🤖 Heart Failure Prediction")
st.markdown("Predicción de enfermedad cardíaca a partir de variables clínicas · Dataset: fedesoriano (Kaggle)")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Exploración", "🔬 Predicción", "📈 Modelo"])


# ════════════════════════════════════════════════════════════
# TAB 1 — EXPLORACIÓN DEL DATASET
# ════════════════════════════════════════════════════════════
with tab1:
    st.header("Exploración del dataset")

    # KPIs resumen
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total pacientes", len(df))
    col2.metric("Con enfermedad", int(df['HeartDisease'].sum()))
    col3.metric("Sin enfermedad", int((df['HeartDisease'] == 0).sum()))
    col4.metric("Prevalencia", f"{df['HeartDisease'].mean() * 100:.1f}%")

    st.divider()

    with st.expander("Ver datos crudos"):
        st.dataframe(df, use_container_width=True)

    st.divider()

    # ── Distribución del target ──────────────────────────────
    st.subheader("Distribución del target")

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.countplot(
        data=df,
        x='HeartDisease',
        hue='HeartDisease',
        palette={0: '#4C72B0', 1: '#C44E52'},
        legend=False,
        ax=ax
    )
    ax.set_title("Distribución de pacientes según diagnóstico")
    ax.set_xlabel("")
    ax.set_ylabel("Número de pacientes")
    ax.set_xticklabels(['Sin enfermedad (0)', 'Con enfermedad (1)'])

    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.pyplot(fig)
    plt.close(fig)

    target_pct = df['HeartDisease'].value_counts(normalize=True) * 100
    st.caption(
        f"El dataset contiene un {target_pct[0]:.1f}% de pacientes sin enfermedad cardíaca "
        f"y un {target_pct[1]:.1f}% con enfermedad cardíaca. "
        "Esta distribución permite evaluar si existe desbalance entre clases, "
        "un aspecto relevante al entrenar modelos clínicos."
    )

    st.divider()

    # ── Variables numéricas ──────────────────────────────────
    st.subheader("Distribución de variables numéricas")

    num_cols = [c for c in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'] if c in df.columns]
    selected_num = st.selectbox("Selecciona una variable numérica", options=num_cols)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(
        data=df,
        x=selected_num,
        hue='HeartDisease',
        palette={0: '#4C72B0', 1: '#C44E52'},
        bins=30,
        multiple='layer',
        alpha=0.6,
        ax=ax
    )
    ax.set_title(f"Distribución de {selected_num} por clase")
    ax.set_xlabel(selected_num)
    ax.set_ylabel("Frecuencia")

    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── Variables categóricas ────────────────────────────────
    st.subheader("Tasa de enfermedad por variable categórica")

    prefixes = ['Sex_', 'ChestPainType_', 'RestingECG_', 'ExerciseAngina_', 'ST_Slope_']
    cat_cols = [col for col in df.columns if any(col.startswith(p) for p in prefixes)]
    selected_cat = st.selectbox("Selecciona una variable categórica", options=cat_cols)

    risk_by_cat = (
        df.groupby(selected_cat)['HeartDisease']
        .mean()
        .mul(100)
        .reset_index()
        .sort_values(selected_cat)
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = sns.barplot(data=risk_by_cat, x=selected_cat, y='HeartDisease', ax=ax)
    for bar, color in zip(ax.patches, ['#4C72B0', '#C44E52']):
        bar.set_facecolor(color)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")
    ax.set_title(f"Tasa de enfermedad cardíaca según {selected_cat}")
    ax.set_xlabel(selected_cat)
    ax.set_ylabel("Pacientes con enfermedad (%)")
    ax.set_xticklabels(["No", "Sí"])

    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Cada barra representa el porcentaje de pacientes diagnosticados "
        "con enfermedad cardíaca dentro de cada categoría."
    )


# ════════════════════════════════════════════════════════════
# TAB 2 — PREDICCIÓN INDIVIDUAL
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("Predicción individual")
    st.markdown("Introduce los valores clínicos del paciente para obtener la predicción del modelo.")

    # ── Inputs numéricos ─────────────────────────────────────
    st.subheader("Variables numéricas")
    col1, col2, col3 = st.columns(3)

    with col1:
        age        = st.slider("Age (años)", min_value=20, max_value=90, value=50)
        restingbp  = st.slider("RestingBP (mmHg)", min_value=80, max_value=200, value=120)

    with col2:
        cholesterol = st.slider("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        maxhr       = st.slider("MaxHR (lpm)", min_value=60, max_value=220, value=150)

    with col3:
        oldpeak   = st.slider("Oldpeak (mm)", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
        fastingbs = st.selectbox("FastingBS (glucemia ayunas >120 mg/dL)", options=[0, 1])

    # ── Inputs categóricos ───────────────────────────────────
    st.subheader("Variables categóricas")
    col4, col5 = st.columns(2)

    with col4:
        sex            = st.selectbox("Sex", options=["M", "F"])
        exerciseangina = st.selectbox("ExerciseAngina (angina inducida por ejercicio)", options=["Y", "N"])

    with col5:
        chestpaintype = st.selectbox("ChestPainType", options=["ATA", "NAP", "ASY", "TA"])
        restingecg    = st.selectbox("RestingECG", options=["Normal", "ST", "LVH"])
        st_slope      = st.selectbox("ST_Slope (pendiente segmento ST)", options=["Up", "Flat", "Down"])

    st.divider()

    # ── Construcción del vector de entrada ───────────────────
    # El modelo fue entrenado con columnas en un orden específico (feature_names).
    # Se construye el vector manualmente en vez de usar get_dummies para evitar
    # inconsistencias cuando una sola fila no genera todas las categorías esperadas.
    #
    # Categorías eliminadas por drop_first=True durante el entrenamiento:
    #   Sex            → eliminó F  → existe Sex_M
    #   ExerciseAngina → eliminó N  → existe ExerciseAngina_Y
    #   ChestPainType  → eliminó ASY → existen ATA, NAP, TA
    #   RestingECG     → eliminó LVH → existen Normal, ST
    #   ST_Slope       → eliminó Down → existen Flat, Up

    input_encoded = {col: 0 for col in feature_names}

    # Numéricos
    input_encoded['Age']         = age
    input_encoded['RestingBP']   = restingbp
    input_encoded['Cholesterol'] = cholesterol
    input_encoded['FastingBS']   = fastingbs
    input_encoded['MaxHR']       = maxhr
    input_encoded['Oldpeak']     = oldpeak

    # Categóricos — activa la columna dummy correspondiente
    if sex == 'M':
        input_encoded['Sex_M'] = 1
    if exerciseangina == 'Y':
        input_encoded['ExerciseAngina_Y'] = 1
    if chestpaintype in ['ATA', 'NAP', 'TA']:
        input_encoded[f'ChestPainType_{chestpaintype}'] = 1
    if restingecg in ['Normal', 'ST']:
        input_encoded[f'RestingECG_{restingecg}'] = 1
    if st_slope in ['Flat', 'Up']:
        input_encoded[f'ST_Slope_{st_slope}'] = 1

    input_df_encoded = pd.DataFrame([input_encoded]).reindex(columns=feature_names, fill_value=0)

    # ── Predicción ───────────────────────────────────────────
    if st.button("🔬 Predecir", type="primary", use_container_width=True):

        pred         = model.predict(input_df_encoded)[0]
        pred_proba   = model.predict_proba(input_df_encoded)[0]
        prob_risk    = pred_proba[1] * 100
        prob_no_risk = pred_proba[0] * 100

        st.divider()

        if pred == 1:
            st.error("⚠️ Riesgo de enfermedad cardíaca detectado")
        else:
            st.success("✅ Sin enfermedad cardíaca detectada")

        # Métricas de probabilidad
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Probabilidad de riesgo",    f"{prob_risk:.1f}%")
        col_r2.metric("Probabilidad sin riesgo",   f"{prob_no_risk:.1f}%")
        col_r3.metric("Predicción", "Enfermedad" if pred == 1 else "Sano")

        st.progress(int(prob_risk))

        # ── Aviso clínico ─────────────────────────────────────
        # Contextualiza las limitaciones del modelo para un uso responsable.
        st.divider()
        st.warning(
            "**⚕️ Nota clínica — limitaciones del modelo**\n\n"
            "Este modelo alcanza un AUC-ROC de 0.95 y un Recall de 0.90 en la clase positiva, "
            "lo que significa que identifica correctamente el 90% de los pacientes con enfermedad cardíaca. "
            "Sin embargo, el 10% restante genera **Falsos Negativos**: pacientes enfermos clasificados como sanos.\n\n"
            "Esto ocurre especialmente en **perfiles límite**: mujeres de mediana edad con ST_Slope Flat "
            "pero sin angina ni MaxHR bajo, donde las variables combinadas no superan el umbral de riesgo aprendido. "
            "En estos casos, el modelo puede subestimar el riesgo real.\n\n"
            "**Esta herramienta es un apoyo al juicio clínico, no un sustituto. "
            "Cualquier resultado debe interpretarse junto con la historia clínica completa del paciente.**"
        )


# ════════════════════════════════════════════════════════════
# TAB 3 — MÉTRICAS DEL MODELO
# ════════════════════════════════════════════════════════════
with tab3:
    st.header("Rendimiento del modelo")
    st.markdown("Gradient Boosting Classifier · Evaluado sobre test set (20% del dataset, 184 pacientes)")

    # KPIs del modelo
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", "0.9479", help="Capacidad de discriminación entre clases. 1.0 = perfecto.")
    col2.metric("F1-Score", "0.9020", help="Media armónica de precisión y recall.")
    col3.metric("Accuracy", "0.8913", help="Porcentaje de predicciones correctas.")

    st.divider()

    st.subheader("Curvas ROC — comparativa de modelos")
    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.image(str(BASE_DIR / 'plots' / '10_roc_curves.png'))
    st.caption("Cada curva representa un modelo. Cuanto mayor el área bajo la curva (AUC), mejor la capacidad discriminativa.")

    st.divider()

    st.subheader("Importancia de features")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.image(str(BASE_DIR / 'plots' / '12_feature_importance.png'))
    st.caption(
        "ST_Slope_Up domina con ~50% de importancia: su ausencia es un marcador clásico de isquemia miocárdica. "
        "MaxHR y Oldpeak reflejan la respuesta al esfuerzo. ChestPainType y Sex completan el perfil de riesgo cardiovascular."
    )

    st.divider()

    st.subheader("Matrices de confusión")
    col_left, col_center, col_right = st.columns([0.5, 4, 0.5])
    with col_center:
        st.image(str(BASE_DIR / 'plots' / '11_confusion_matrices.png'))
    st.caption(
        "Los Falsos Negativos (FN) — pacientes enfermos clasificados como sanos — son el error más costoso "
        "en contexto clínico. El modelo prioriza minimizarlos mediante un Recall alto en la clase positiva."
    )


# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("Proyecto de portafolio · Heart Failure Prediction · Sergio Falcó · 2026")
