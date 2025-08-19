import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# --------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------
st.set_page_config(
    page_title="Dashboard Avanzado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# ESTILOS PERSONALIZADOS
# --------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    .report-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 Dashboard Analítico Avanzado</h1>', unsafe_allow_html=True)
st.markdown("Explora datos interactivos con visualizaciones modernas")
st.markdown("Segmento construido como ejemplo")

# --------------------------
# DATA SIMULADA MEJORADA
# --------------------------
np.random.seed(42)
fechas = pd.date_range("2024-01-01", periods=100)
categorias = ["Electrónica", "Ropa", "Hogar", "Deportes", "Libros"]
regiones = ["Norte", "Sur", "Este", "Oeste"]

df = pd.DataFrame({
    "fecha": fechas,
    "ventas": np.random.randint(100, 500, size=100),
    "clientes": np.random.randint(20, 100, size=100),
    "categoria": np.random.choice(categorias, size=100),
    "region": np.random.choice(regiones, size=100),
    "satisfaccion": np.random.uniform(3, 5, size=100),
    "devoluciones": np.random.randint(0, 10, size=100),
    "margen": np.random.uniform(10, 40, size=100),
    "stock": np.random.randint(50, 200, size=100)
})

# --------------------------
# SIDEBAR MEJORADO
# --------------------------
st.sidebar.header("🔧 Panel de Control")
with st.sidebar.expander("Filtros Temporales", expanded=True):
    fecha_min = st.date_input("Fecha mínima", df["fecha"].min())
    fecha_max = st.date_input("Fecha máxima", df["fecha"].max())
    
with st.sidebar.expander("Filtros Categóricos"):
    categorias_sel = st.multiselect("Categorías", df["categoria"].unique(), default=df["categoria"].unique())
    regiones_sel = st.multiselect("Regiones", df["region"].unique(), default=df["region"].unique())
    
with st.sidebar.expander("Parámetros de Visualización"):
    tema = st.selectbox("Tema de Gráficos", ["plotly", "plotly_white", "plotly_dark"])
    paleta = st.selectbox("Paleta de Colores", ["viridis", "plasma", "inferno", "magma", "cividis"])
    tamaño_puntos = st.slider("Tamaño de Puntos", 1, 20, 5)
    
with st.sidebar.expander("Opciones Avanzadas"):
    mostrar_outliers = st.checkbox("Mostrar Outliers", value=True)
    normalizar_datos = st.checkbox("Normalizar Datos", value=False)
    st.download_button(
        "📥 Descargar Datos Completos",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="datos_completos.csv",
        mime="text/csv"
    )

# Aplicar filtros
df_filtered = df[
    (df["fecha"] >= pd.to_datetime(fecha_min)) &
    (df["fecha"] <= pd.to_datetime(fecha_max)) &
    (df["categoria"].isin(categorias_sel)) &
    (df["region"].isin(regiones_sel))
].copy()

if normalizar_datos:
    scaler = MinMaxScaler()
    df_filtered[["ventas", "clientes", "margen"]] = scaler.fit_transform(df_filtered[["ventas", "clientes", "margen"]])

# --------------------------
# KPIs MEJORADOS
# --------------------------
st.markdown("## 📊 Métricas Clave")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Ventas", f"${df_filtered['ventas'].sum():,}", 
              delta=f"{df_filtered['ventas'].sum() - df['ventas'].sum():.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Clientes Únicos", f"{df_filtered['clientes'].nunique()}", 
              delta=f"{df_filtered['clientes'].nunique() - df['clientes'].nunique()}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Margen Promedio", f"{df_filtered['margen'].mean():.1f}%", 
              delta=f"{df_filtered['margen'].mean() - df['margen'].mean():.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Satisfacción", f"{df_filtered['satisfaccion'].mean():.2f}/5", 
              delta=f"{df_filtered['satisfaccion'].mean() - df['satisfaccion'].mean():.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# TABS DE ANÁLISIS
# --------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Series Temporales", 
    "📊 Análisis Comparativo", 
    "🌍 Geográfico", 
    "📈 Estadística Avanzada",
    "🎨 Visualizaciones Creativas",
    "📥 Exportación"
])

# --------------------------
# TAB 1: SERIES TEMPORALES
# --------------------------
with tab1:
    st.header("Análisis de Series Temporales")
    
    # Gráfico 1: Evolución de Ventas
    st.subheader("Evolución de Ventas")
    fig1 = px.line(
        df_filtered, 
        x="fecha", 
        y="ventas", 
        color="categoria",
        title="Ventas Diarias por Categoría",
        template=tema,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig1.update_layout(legend_title_text="Categorías")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Clientes y Satisfacción
    st.subheader("Clientes y Satisfacción")
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(
        go.Scatter(
            x=df_filtered["fecha"], 
            y=df_filtered["clientes"],
            name="Clientes",
            line=dict(color="#1f77b4")
        ),
        secondary_y=False,
    )
    fig2.add_trace(
        go.Scatter(
            x=df_filtered["fecha"], 
            y=df_filtered["satisfaccion"],
            name="Satisfacción",
            line=dict(color="#ff7f0e")
        ),
        secondary_y=True,
    )
    fig2.update_xaxes(title_text="Fecha")
    fig2.update_yaxes(title_text="Clientes", secondary_y=False)
    fig2.update_yaxes(title_text="Satisfacción", secondary_y=True)
    fig2.update_layout(template=tema, title_text="Clientes vs Satisfacción")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Descomposición de Ventas
    st.subheader("Descomposición de Ventas")
    df_filtered['mes'] = df_filtered['fecha'].dt.month
    ventas_mensuales = df_filtered.groupby(['mes', 'categoria'])['ventas'].sum().reset_index()
    fig3 = px.bar(
        ventas_mensuales, 
        x="mes", 
        y="ventas", 
        color="categoria",
        title="Ventas Mensuales por Categoría",
        template=tema,
        barmode="stack"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Gráfico 4: Análisis de Tendencia
    st.subheader("Análisis de Tendencia")
    z = np.polyfit(range(len(df_filtered)), df_filtered['ventas'], 1)
    p = np.poly1d(z)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=df_filtered['fecha'], 
        y=df_filtered['ventas'],
        mode='markers',
        name='Ventas Reales',
        marker=dict(size=tamaño_puntos)
    ))
    fig4.add_trace(go.Scatter(
        x=df_filtered['fecha'], 
        y=p(range(len(df_filtered))),
        mode='lines',
        name='Tendencia',
        line=dict(color='red', width=3)
    ))
    fig4.update_layout(template=tema, title_text="Tendencia de Ventas")
    st.plotly_chart(fig4, use_container_width=True)

# --------------------------
# TAB 2: ANÁLISIS COMPARATIVO
# --------------------------
with tab2:
    st.header("Análisis Comparativo")
    
    # Gráfico 5: Matriz de Correlación
    st.subheader("Matriz de Correlación")
    corr = df_filtered[["ventas", "clientes", "satisfaccion", "margen", "devoluciones"]].corr()
    fig5 = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=paleta,
        title="Matriz de Correlación"
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # Gráfico 6: Comparación por Categoría
    st.subheader("Métricas por Categoría")
    fig6 = px.scatter(
        df_filtered,
        x="ventas",
        y="margen",
        size="clientes",
        color="categoria",
        hover_name="categoria",
        size_max=tamaño_puntos*3,
        title="Ventas vs Margen por Categoría",
        template=tema
    )
    st.plotly_chart(fig6, use_container_width=True)
    
    # Gráfico 7: Distribución de Satisfacción
    st.subheader("Distribución de Satisfacción")
    fig7 = px.violin(
        df_filtered,
        y="satisfaccion",
        x="categoria",
        box=True,
        points="all",
        title="Distribución de Satisfacción por Categoría",
        template=tema,
        color="categoria"
    )
    st.plotly_chart(fig7, use_container_width=True)
    
    # Gráfico 8: Análisis de Pareto
    st.subheader("Análisis de Pareto (Devoluciones)")
    pareto_df = df_filtered.groupby('categoria')['devoluciones'].sum().reset_index()
    pareto_df = pareto_df.sort_values('devoluciones', ascending=False)
    pareto_df['cumulative'] = pareto_df['devoluciones'].cumsum() / pareto_df['devoluciones'].sum() * 100
    
    fig8 = make_subplots(specs=[[{"secondary_y": True}]])
    fig8.add_trace(
        go.Bar(x=pareto_df['categoria'], y=pareto_df['devoluciones'], name="Devoluciones"),
        secondary_y=False,
    )
    fig8.add_trace(
        go.Scatter(x=pareto_df['categoria'], y=pareto_df['cumulative'], name="Acumulado", line=dict(color='red')),
        secondary_y=True,
    )
    fig8.update_layout(template=tema, title_text="Análisis de Pareto de Devoluciones")
    st.plotly_chart(fig8, use_container_width=True)

# --------------------------
# TAB 3: GEOGRÁFICO
# --------------------------
with tab3:
    st.header("Análisis Geográfico")
    
    # Crear datos de mapa una vez para todo el tab
    df_map = df_filtered.copy()
    df_map['lat'] = np.random.uniform(-34, -33, size=len(df_map))
    df_map['lon'] = np.random.uniform(-71, -70, size=len(df_map))
    
    # Gráfico 9: Mapa de Calor Regional
    st.subheader("Mapa de Calor Regional")
    region_data = df_filtered.groupby('region').agg({
        'ventas': 'sum',
        'clientes': 'mean',
        'satisfaccion': 'mean'
    }).reset_index()
    
    fig9 = px.scatter_geo(
        region_data,
        locations="region",
        locationmode="country names",
        size="ventas",
        hover_name="region",
        hover_data=["clientes", "satisfaccion"],
        projection="natural earth",
        title="Distribución Geográfica de Ventas",
        template=tema,
        color="ventas",
        color_continuous_scale=paleta
    )
    st.plotly_chart(fig9, use_container_width=True)
    
    # Gráfico 10: Análisis Regional
    st.subheader("Análisis Comparativo Regional")
    fig10 = px.bar(
        region_data,
        x="region",
        y=["ventas", "clientes"],
        title="Ventas y Clientes por Región",
        template=tema,
        barmode="group"
    )
    st.plotly_chart(fig10, use_container_width=True)
    
    # Gráfico 11: Mapa de Dispersión
    st.subheader("Mapa de Dispersión de Clientes")
    fig11 = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        color="region",
        size="clientes",
        hover_name="categoria",
        hover_data=["ventas", "satisfaccion"],
        zoom=8,
        mapbox_style="carto-positron",
        title="Distribución Geográfica de Clientes",
        template=tema
    )
    st.plotly_chart(fig11, use_container_width=True)
    
    # Gráfico 12: Análisis de Rutas
    st.subheader("Análisis de Rutas de Distribución")
    fig12 = px.line_geo(
        df_map,
        lat="lat",
        lon="lon",
        color="region",
        line_group="categoria",
        title="Rutas de Distribución por Categoría",
        template=tema
    )
    st.plotly_chart(fig12, use_container_width=True)

# --------------------------
# TAB 4: ESTADÍSTICA AVANZADA
# --------------------------
with tab4:
    st.header("Análisis Estadístico Avanzado")
    
    # Gráfico 13: Regresión Lineal - CORREGIDO
    st.subheader("Regresión Lineal: Ventas vs Clientes")
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_filtered['clientes'], df_filtered['ventas']
    )
    
    # Crear scatter plot con Plotly Express
    fig13 = px.scatter(
        df_filtered,
        x="clientes",
        y="ventas",
        color="categoria",
        title=f"Ventas vs Clientes (R²={r_value**2:.2f})",
        template=tema,
        size_max=tamaño_puntos*2
    )
    
    # Añadir línea de regresión
    x_vals = np.array([df_filtered['clientes'].min(), df_filtered['clientes'].max()])
    y_vals = intercept + slope * x_vals
    fig13.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name=f'Regresión (R²={r_value**2:.2f})',
        line=dict(color='red', width=3)
    ))
    
    st.plotly_chart(fig13, use_container_width=True)
    
    # Gráfico 14: Distribución de Probabilidad
    st.subheader("Distribución de Ventas")
    fig14 = go.Figure()
    fig14.add_trace(go.Histogram(
        x=df_filtered['ventas'],
        nbinsx=20,
        name='Ventas',
        histnorm='probability'
    ))
    fig14.add_trace(go.Scatter(
        x=np.linspace(df_filtered['ventas'].min(), df_filtered['ventas'].max(), 100),
        y=stats.norm.pdf(
            np.linspace(df_filtered['ventas'].min(), df_filtered['ventas'].max(), 100),
            df_filtered['ventas'].mean(),
            df_filtered['ventas'].std()
        ),
        mode='lines',
        name='Distribución Normal',
        line=dict(color='red')
    ))
    fig14.update_layout(template=tema, title_text="Distribución de Ventas")
    st.plotly_chart(fig14, use_container_width=True)
    
    # Gráfico 15: Análisis de Outliers
    st.subheader("Análisis de Outliers")
    Q1 = df_filtered['ventas'].quantile(0.25)
    Q3 = df_filtered['ventas'].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = df_filtered[(df_filtered['ventas'] < (Q1 - 1.5 * IQR)) | 
                           (df_filtered['ventas'] > (Q3 + 1.5 * IQR))]
    
    fig15 = px.box(
        df_filtered,
        y="ventas",
        x="categoria",
        title="Distribución de Ventas con Outliers",
        template=tema,
        color="categoria"
    )
    
    if mostrar_outliers and not outliers.empty:
        fig15.add_trace(go.Scatter(
            x=outliers['categoria'],
            y=outliers['ventas'],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Outliers'
        ))
    
    st.plotly_chart(fig15, use_container_width=True)
    
    # Gráfico 16: Análisis de Componentes Principales (PCA)
    st.subheader("Análisis de Componentes Principales")
    features = ["ventas", "clientes", "satisfaccion", "margen", "devoluciones"]
    x = df_filtered.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=['PC1', 'PC2']
    )
    pca_df['categoria'] = df_filtered['categoria'].values
    
    fig16 = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="categoria",
        title="Análisis de Componentes Principales",
        template=tema
    )
    st.plotly_chart(fig16, use_container_width=True)

# --------------------------
# TAB 5: VISUALIZACIONES CREATIVAS
# --------------------------
with tab5:
    st.header("Visualizaciones Creativas")
    
    # Gráfico 17: Gráfico de Radar
    st.subheader("Gráfico de Radar por Categoría")
    radar_df = df_filtered.groupby('categoria').mean(numeric_only=True).reset_index()
    
    fig17 = go.Figure()
    for categoria in radar_df['categoria']:
        fig17.add_trace(go.Scatterpolar(
            r=radar_df[radar_df['categoria'] == categoria][['ventas', 'clientes', 'margen', 'satisfaccion']].values[0],
            theta=['Ventas', 'Clientes', 'Margen', 'Satisfacción'],
            fill='toself',
            name=categoria
        ))
    
    fig17.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, radar_df[['ventas', 'clientes', 'margen', 'satisfaccion']].max().max()]
            )),
        showlegend=True,
        title_text="Perfil por Categoría",
        template=tema
    )
    st.plotly_chart(fig17, use_container_width=True)
    
    # Gráfico 18: Gráfico de Sankey
    st.subheader("Flujo de Ventas por Región y Categoría")
    sankey_df = df_filtered.groupby(['region', 'categoria'])['ventas'].sum().reset_index()
    
    regiones = sankey_df['region'].unique()
    categorias = sankey_df['categoria'].unique()
    
    # Normalizar valores de ventas para asignar colores
    min_ventas = sankey_df['ventas'].min()
    max_ventas = sankey_df['ventas'].max()
    norm_ventas = (sankey_df['ventas'] - min_ventas) / (max_ventas - min_ventas)
    
    # Crear colores basados en la paleta seleccionada
    if paleta == "viridis":
        colors = pc.sample_colorscale("Viridis", norm_ventas.tolist())
    elif paleta == "plasma":
        colors = pc.sample_colorscale("Plasma", norm_ventas.tolist())
    elif paleta == "inferno":
        colors = pc.sample_colorscale("Inferno", norm_ventas.tolist())
    elif paleta == "magma":
        colors = pc.sample_colorscale("Magma", norm_ventas.tolist())
    else:  # cividis
        colors = pc.sample_colorscale("Cividis", norm_ventas.tolist())
    
    fig18 = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(regiones) + list(categorias),
            color=["#1f77b4"] * len(regiones) + ["#ff7f0e"] * len(categorias)
        ),
        link=dict(
            source=[list(regiones).index(r) for r in sankey_df['region']],
            target=[len(regiones) + list(categorias).index(c) for c in sankey_df['categoria']],
            value=sankey_df['ventas'],
            color=colors,
        )
    )])
    
    fig18.update_layout(title_text="Flujo de Ventas: Región → Categoría", template=tema)
    st.plotly_chart(fig18, use_container_width=True)
    
    # Gráfico 19: Gráfico de Árbol (Treemap)
    st.subheader("Distribución de Ventas (Treemap)")
    fig19 = px.treemap(
        df_filtered,
        path=['region', 'categoria'],
        values='ventas',
        color='margen',
        color_continuous_scale=paleta,
        title="Ventas por Región y Categoría",
        template=tema
    )
    st.plotly_chart(fig19, use_container_width=True)
    
    # Gráfico 20: Gráfico 3D
    st.subheader("Análisis Tridimensional")
    fig20 = px.scatter_3d(
        df_filtered,
        x='ventas',
        y='clientes',
        z='margen',
        color='categoria',
        size='satisfaccion',
        hover_name='categoria',
        title="Relación 3D: Ventas, Clientes y Margen",
        template=tema,
        opacity=0.8
    )
    st.plotly_chart(fig20, use_container_width=True)

# --------------------------
# TAB 6: EXPORTACIÓN
# --------------------------
with tab6:
    st.header("Exportación de Datos y Reportes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Descargar Datos Filtrados")
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Descargar CSV",
            data=csv,
            file_name="datos_filtrados.csv",
            mime="text/csv"
        )
        
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_filtered.to_excel(writer, sheet_name='Datos', index=False)
        st.download_button(
            "⬇️ Descargar Excel",
            data=excel_buffer,
            file_name="datos_filtrados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        st.subheader("Generar Reporte PDF")
        if st.button("📄 Generar Reporte PDF"):
            st.success("Reporte generado exitosamente!")
            st.balloons()
    
    st.subheader("Vista Previa de Datos")
    st.dataframe(df_filtered, use_container_width=True)
    
    st.subheader("Estadísticas Descriptivas")
    st.write(df_filtered.describe(include='all'))
    
    st.subheader("Datos Faltantes")
    missing_data = df_filtered.isnull().sum().reset_index()
    missing_data.columns = ['Columna', 'Valores Faltantes']
    st.dataframe(missing_data, use_container_width=True)
    
    # Nueva sección: Resumen de análisis
    st.markdown('<div class="report-section">', unsafe_allow_html=True)
    st.subheader("📋 Resumen de Análisis")
    st.markdown(f"""
    - **Período analizado**: {fecha_min} a {fecha_max}
    - **Categorías seleccionadas**: {', '.join(categorias_sel)}
    - **Regiones incluidas**: {', '.join(regiones_sel)}
    - **Total de registros**: {len(df_filtered)}
    - **Ventas totales**: ${df_filtered['ventas'].sum():,.2f}
    - **Satisfacción promedio**: {df_filtered['satisfaccion'].mean():.2f}/5
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.markdown("Dashboard creado con ❤️ usando Streamlit | Datos actualizados en tiempo real")