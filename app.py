import streamlit as st

st.title("Mi primer dashboard en Streamlit")
st.write("¡Hola, esto es un ejemplo!")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# --------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------
st.set_page_config(
    page_title="Ejemplo Dashboard Portafolio",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard de Ejemplo con Streamlit")
st.markdown("Este es un **ejemplo interactivo** para tu portafolio.")

# --------------------------
# DATA SIMULADA
# --------------------------
np.random.seed(42)
fechas = pd.date_range("2024-01-01", periods=100)
df = pd.DataFrame({
    "fecha": fechas,
    "ventas": np.random.randint(100, 500, size=100),
    "clientes": np.random.randint(20, 100, size=100),
    "categoria": np.random.choice(["A", "B", "C"], size=100)
})

# --------------------------
# SIDEBAR (Filtros)
# --------------------------
st.sidebar.header("Filtros")
fecha_min = st.sidebar.date_input("Fecha mínima", df["fecha"].min())
fecha_max = st.sidebar.date_input("Fecha máxima", df["fecha"].max())
categorias = st.sidebar.multiselect("Categorías", df["categoria"].unique(), default=df["categoria"].unique())

# Aplicar filtros
df_filtered = df[(df["fecha"] >= pd.to_datetime(fecha_min)) &
                 (df["fecha"] <= pd.to_datetime(fecha_max)) &
                 (df["categoria"].isin(categorias))]

# --------------------------
# KPIs (Métricas principales)
# --------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Ventas", f"${df_filtered['ventas'].sum():,}")
col2.metric("Clientes", f"{df_filtered['clientes'].sum():,}")
col3.metric("Categorías activas", len(df_filtered["categoria"].unique()))

# --------------------------
# TABS DE ANÁLISIS
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Series temporales", "📊 Comparaciones", "🌍 Geográfico", "📥 Exportar datos"])

with tab1:
    st.subheader("Evolución de Ventas")
    fig = px.line(df_filtered, x="fecha", y="ventas", title="Ventas en el tiempo")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Clientes en el tiempo")
    fig2 = px.bar(df_filtered, x="fecha", y="clientes", title="Clientes en el tiempo")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Distribución de ventas por categoría")
    fig3 = px.pie(df_filtered, names="categoria", values="ventas", hole=0.3)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Heatmap de correlaciones")
    corr = df_filtered[["ventas", "clientes"]].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Mapa de clientes (simulado)")
    df_map = pd.DataFrame({
        "lat": -33.45 + np.random.randn(50) * 0.1,
        "lon": -70.66 + np.random.randn(50) * 0.1,
        "clientes": np.random.randint(10, 100, size=50)
    })
    st.map(df_map)

with tab4:
    st.subheader("Descargar datos filtrados")
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV",
        data=csv,
        file_name="datos_filtrados.csv",
        mime="text/csv"
    )
    st.write("Vista previa de los datos:")
    st.dataframe(df_filtered.head())

