import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from data import get_data

# Generar datos simulados
df = get_data()
regiones = df["region"].unique()
años = df["Año"].unique()
delitos = df["Delito"].unique()

# Calcular IDI total por región y año
idi_total = df.groupby(['Región', 'Año'])['Aporte Delito al IDI'].sum().reset_index()
idi_total.rename(columns={'Aporte Delito al IDI': 'IDI Total'}, inplace=True)
# Unir IDI total al DataFrame original
df = pd.merge(df, idi_total, on=['Región', 'Año'])

# AGREGAR COORDENADAS DE REGIONES
coordenadas_regiones = {
    'Arica y Parinacota': {'lat': -18.4788, 'lon': -70.3126},
    'Tarapacá': {'lat': -20.2132, 'lon': -69.5714},
    'Antofagasta': {'lat': -23.6500, 'lon': -70.4000},
    'Atacama': {'lat': -27.5500, 'lon': -70.0000},
    'Coquimbo': {'lat': -30.0000, 'lon': -71.0000},
    'Valparaíso': {'lat': -33.0000, 'lon': -71.5000},
    'Metropolitana': {'lat': -33.4500, 'lon': -70.6667},
    "O'Higgins": {'lat': -34.5000, 'lon': -71.0000},
    'Maule': {'lat': -35.5000, 'lon': -71.5000},
    'Ñuble': {'lat': -36.6000, 'lon': -72.0000},
    'Biobío': {'lat': -37.5000, 'lon': -72.0000},
    'La Araucanía': {'lat': -38.7000, 'lon': -72.5000},
    'Los Ríos': {'lat': -40.5000, 'lon': -73.0000},
    'Los Lagos': {'lat': -42.0000, 'lon': -73.5000},
    'Aysén': {'lat': -46.5000, 'lon': -73.0000},
    'Magallanes': {'lat': -53.0000, 'lon': -70.0000}
}

# Crear DataFrame con coordenadas
df_coordenadas = pd.DataFrame([
    {'Región': region, 'lat': data['lat'], 'lon': data['lon']}
    for region, data in coordenadas_regiones.items()
])

# Configuración de la página de Streamlit
st.set_page_config(layout="wide")
st.title("Análisis de Delincuencia por Región")

# Filtro de región en la barra lateral
st.sidebar.header("Filtros")
region_seleccionada = st.sidebar.selectbox(
    "Selecciona una región:",
    options=regiones,
    index=0
)

# Filtrar datos por región seleccionada
df_region = df[df['Región'] == region_seleccionada]

# Gráfico 1: Frecuencia de Delitos (último año disponible)
ultimo_año = df_region['Año'].max()
df_ultimo_año = df_region[df_region['Año'] == ultimo_año]
st.header(f"Frecuencia de Delitos - {region_seleccionada} ({ultimo_año})")
fig1 = px.bar(
    df_ultimo_año,
    x="Delito",
    y="Frecuencia",
    color="Delito",
    text="Frecuencia",
    labels={"Frecuencia": "Número de Casos", "Delito": "Tipo de Delito"},
    title=f"Distribución de Frecuencia de Delitos ({ultimo_año})"
)
fig1.update_traces(texttemplate='%{text}', textposition='outside')
fig1.update_layout(
    xaxis_title="Tipo de Delito",
    yaxis_title="Número de Casos",
    showlegend=False,
    height=500
)
st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2: Contribución al IDI (último año disponible)
st.header(f"Contribución al IDI - {region_seleccionada} ({ultimo_año})")
fig2 = px.bar(
    df_ultimo_año,
    x="Delito",
    y="Aporte Delito al IDI",
    color="Delito",
    text="Aporte Delito al IDI",
    labels={"Aporte Delito al IDI": "Aporte al IDI", "Delito": "Tipo de Delito"},
    title=f"Contribución de Cada Delito al IDI ({ultimo_año})"
)
fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig2.update_layout(
    xaxis_title="Tipo de Delito",
    yaxis_title="Aporte al IDI",
    showlegend=False,
    height=500
)
st.plotly_chart(fig2, use_container_width=True)

# Gráfico 3: Evolución del IDI (Componente Principal 1)
st.header(f"Evolución del Índice de Delincuencia Integrado - {region_seleccionada}")
df_idi = df_region[['Año', 'IDI Total']].drop_duplicates().sort_values('Año')
fig3 = px.line(
    df_idi,
    x="Año",
    y="IDI Total",
    markers=True,
    title="Evolución Temporal del IDI",
    labels={"IDI Total": "Valor del IDI", "Año": "Año"}
)
fig3.update_traces(line=dict(width=3), marker=dict(size=8))
fig3.update_layout(
    xaxis_title="Año",
    yaxis_title="Valor del IDI",
    height=500
)
st.plotly_chart(fig3, use_container_width=True)

# Gráfico 4: Análisis de Componentes Principales (Componente Principal 2)
st.header(f"Análisis de Componentes Principales - {region_seleccionada} ({ultimo_año})")
# Preparar datos para PCA
df_pca = df_ultimo_año[['Delito', 'Frecuencia', 'Tasa cada 100 mil hab', 'Aporte Delito al IDI']].copy()
# Normalizar datos
df_pca['Frecuencia_norm'] = (df_pca['Frecuencia'] - df_pca['Frecuencia'].mean()) / df_pca['Frecuencia'].std()
df_pca['Tasa_norm'] = (df_pca['Tasa cada 100 mil hab'] - df_pca['Tasa cada 100 mil hab'].mean()) / df_pca['Tasa cada 100 mil hab'].std()
df_pca['Aporte_norm'] = (df_pca['Aporte Delito al IDI'] - df_pca['Aporte Delito al IDI'].mean()) / df_pca['Aporte Delito al IDI'].std()
# Calcular componentes principales (simplificado)
df_pca['Componente 1'] = (df_pca['Frecuencia_norm'] + df_pca['Tasa_norm'] + df_pca['Aporte_norm']) / 3
df_pca['Componente 2'] = (df_pca['Frecuencia_norm'] - df_pca['Tasa_norm'] + df_pca['Aporte_norm']) / 3
# Crear gráfico de dispersión
fig4 = px.scatter(
    df_pca,
    x="Componente 1",
    y="Componente 2",
    color="Delito",
    size="Frecuencia",
    hover_data=['Frecuencia', 'Tasa cada 100 mil hab', 'Aporte Delito al IDI'],
    title="Análisis de Componentes Principales",
    labels={
        "Componente 1": "Componente Principal 1 (Frecuencia + Tasa + Aporte)",
        "Componente 2": "Componente Principal 2 (Frecuencia - Tasa + Aporte)"
    }
)
fig4.update_traces(marker=dict(sizemode='diameter', sizeref=0.5))
fig4.update_layout(
    height=600,
    legend_title_text="Tipo de Delito"
)
st.plotly_chart(fig4, use_container_width=True)

# SECCIÓN DE MAPAS
st.header("Análisis Geográfico de Delincuencia")

# Obtener último año disponible en todo el dataset
ultimo_año_global = df['Año'].max()

# Preparar datos para el mapa
df_mapa = df[df['Año'] == ultimo_año_global].groupby('Región').agg({
    'IDI Total': 'first',
    'Frecuencia': 'sum',
    'Población Estimada': 'first'
}).reset_index()

# Unir con coordenadas
df_mapa = pd.merge(df_mapa, df_coordenadas, on='Región', how='left')

# Crear mapa de burbujas
fig_mapa = px.scatter_mapbox(
    df_mapa,
    lat="lat",
    lon="lon",
    size="IDI Total",
    color="IDI Total",
    hover_name="Región",
    hover_data={
        "IDI Total": ":.2f",
        "Frecuencia": ":,.0f",
        "Población Estimada": ":,.0f",
        "lat": False,
        "lon": False
    },
    color_continuous_scale=px.colors.sequential.Viridis,
    size_max=50,
    zoom=4,
    title=f"Índice de Delincuencia por Región ({ultimo_año_global})"
)

fig_mapa.update_layout(
    mapbox_style="open-street-map",
    height=600,
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)

st.plotly_chart(fig_mapa, use_container_width=True)

# Métricas clave
st.header("Métricas Clave")
col1, col2, col3 = st.columns(3)
col1.metric("IDI Total", f"{df_ultimo_año['IDI Total'].iloc[0]:.1f}")
col2.metric("Población", f"{df_ultimo_año['Población Estimada'].iloc[0]:,.0f}")
col3.metric("Total Delitos", f"{df_ultimo_año['Frecuencia'].sum():,.0f}")

# Explicación del análisis
with st.expander("Explicación del Análisis de Componentes Principales"):
    st.markdown("""
    **¿Qué es el Análisis de Componentes Principales (PCA)?**
    
    El PCA es una técnica estadística que reduce la dimensionalidad de los datos, transformando múltiples variables en un conjunto más pequeño de componentes principales que conservan la mayor parte de la información original.
    
    **Interpretación del Gráfico:**
    
    - **Eje X (Componente 1):** Representa una combinación lineal de la frecuencia, tasa y aporte al IDI. Valores más altos indican mayor incidencia delictiva en general.
    
    - **Eje Y (Componente 2):** Representa una combinación que contrasta la frecuencia con la tasa. Valores altos pueden indicar delitos con alta frecuencia pero relativamente baja tasa per cápita.
    
    - **Tamaño de los puntos:** Representa la frecuencia absoluta de cada delito.
    
    - **Colores:** Diferencian los tipos de delitos.
    
    Este análisis permite identificar patrones y relaciones complejas entre los diferentes tipos de delitos que no son evidentes en los análisis univariados.
    """)

with st.expander("Explicación del Mapa de Delincuencia"):
    st.markdown("""
    **Mapa de Delincuencia por Región**
    
    Este mapa muestra la distribución geográfica del Índice de Delincuencia Integrado (IDI) a nivel nacional:
    
    - **Tamaño de las burbujas:** Representa la magnitud del IDI total de cada región (a mayor tamaño, mayor índice de delincuencia).
    
    - **Color de las burbujas:** Indica la intensidad del IDI (escala de verde a amarillo, donde amarillo representa valores más altos).
    
    - **Información al pasar el cursor:** Muestra detalles específicos de cada región:
        - IDI Total
        - Total de delitos
        - Población estimada
    
    **Interpretación:**
    
    - Las regiones con burbujas más grandes y de color más intenso tienen mayor incidencia delictiva.
    - Permite comparar rápidamente la situación de delincuencia entre diferentes regiones.
    - Facilita la identificación de patrones geográficos en la distribución de la delincuencia.
    """)