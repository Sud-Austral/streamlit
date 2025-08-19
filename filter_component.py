# --------------------------
# PANDAS FILTER TEMPLATE - VERSION SIN dash_design_kit
# --------------------------
import os
import sys
from datetime import datetime, time
from typing import TypedDict, Any

from dash import callback, html, dcc, Output, Input
import numpy as np
import pandas as pd

from data import get_data

class FILTER_COMPONENT_IDS:
    region = "region"
    year_min = "year_min"
    year_max = "year_max"
    crime_type = "crime_type"
    municipality = "municipality"
    frequency_min = "frequency_min"
    frequency_max = "frequency_max"
    idi_min = "idi_min"
    idi_max = "idi_max"
    population_min = "population_min"
    population_max = "population_max"
    crime_rate_min = "crime_rate_min"
    crime_rate_max = "crime_rate_max"

# Callback inputs
FILTER_CALLBACK_INPUTS = {
    "region": Input(FILTER_COMPONENT_IDS.region, "value"),
    "year_min": Input(FILTER_COMPONENT_IDS.year_min, "value"),
    "year_max": Input(FILTER_COMPONENT_IDS.year_max, "value"),
    "crime_type": Input(FILTER_COMPONENT_IDS.crime_type, "value"),
    "municipality": Input(FILTER_COMPONENT_IDS.municipality, "value"),
    "frequency_min": Input(FILTER_COMPONENT_IDS.frequency_min, "value"),
    "frequency_max": Input(FILTER_COMPONENT_IDS.frequency_max, "value"),
    "idi_min": Input(FILTER_COMPONENT_IDS.idi_min, "value"),
    "idi_max": Input(FILTER_COMPONENT_IDS.idi_max, "value"),
    "population_min": Input(FILTER_COMPONENT_IDS.population_min, "value"),
    "population_max": Input(FILTER_COMPONENT_IDS.population_max, "value"),
    "crime_rate_min": Input(FILTER_COMPONENT_IDS.crime_rate_min, "value"),
    "crime_rate_max": Input(FILTER_COMPONENT_IDS.crime_rate_max, "value"),
}

class TestInput(TypedDict):
    options: list[Any]
    default: Any

class ComponentResponse(TypedDict):
    layout: html.Div
    test_inputs: dict[str, TestInput]

def component() -> ComponentResponse:
    df = get_data()

    

    # Unique values
    unique_regions = sorted(df["Región"].dropna().replace('', np.nan).dropna().unique().tolist())
    unique_crime_types = sorted(df["Delito"].dropna().replace('', np.nan).dropna().unique().tolist())
    unique_municipalities = sorted(df["Comuna"].dropna().replace('', np.nan).dropna().unique().tolist())

    # Min/max
    year_min = int(df["Año"].min())
    year_max = int(df["Año"].max())
    frequency_min = int(df["Frecuencia"].min())
    frequency_max = int(df["Frecuencia"].max())
    idi_min = float(df["Índice Delincuencia Integrado (IDI)"].min())
    idi_max = float(df["Índice Delincuencia Integrado (IDI)"].max())
    population_min = int(df["Población Estimada_numeric"].dropna().min())
    population_max = int(df["Población Estimada_numeric"].dropna().max())
    crime_rate_min = float(df["Tasa cada 100 mil hab"].min())
    crime_rate_max = float(df["Tasa cada 100 mil hab"].max())

    # Layout usando solo html y dcc
    layout = html.Div([
        html.H4("Filters"),
        html.Div([
            html.Label("Región"),
            dcc.Dropdown(
                id=FILTER_COMPONENT_IDS.region,
                options=[{"label": "All", "value": "all"}] + [{"label": r, "value": r} for r in unique_regions],
                multi=True,
                value=["all"]
            )
        ], style={"minWidth": "200px", "marginBottom": "10px"}),

        html.Div([
            html.Label("Year Range"),
            dcc.Input(id=FILTER_COMPONENT_IDS.year_min, value=year_min, type="number", style={"width": 80}),
            html.Span(" - "),
            dcc.Input(id=FILTER_COMPONENT_IDS.year_max, value=year_max, type="number", style={"width": 80})
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Crime Type"),
            dcc.Dropdown(
                id=FILTER_COMPONENT_IDS.crime_type,
                options=[{"label": "All", "value": "all"}] + [{"label": c, "value": c} for c in unique_crime_types],
                multi=True,
                value=["all"]
            )
        ], style={"minWidth": "200px", "marginBottom": "10px"}),

        html.Div([
            html.Label("Municipality"),
            dcc.Dropdown(
                id=FILTER_COMPONENT_IDS.municipality,
                options=[{"label": "All", "value": "all"}] + [{"label": m, "value": m} for m in unique_municipalities],
                multi=True,
                value=["all"]
            )
        ], style={"minWidth": "200px", "marginBottom": "10px"}),

        html.Div([
            html.Label("Crime Frequency Range"),
            dcc.Input(id=FILTER_COMPONENT_IDS.frequency_min, value=frequency_min, type="number", style={"width": 80}),
            html.Span(" - "),
            dcc.Input(id=FILTER_COMPONENT_IDS.frequency_max, value=frequency_max, type="number", style={"width": 80})
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("IDI Range"),
            dcc.Input(id=FILTER_COMPONENT_IDS.idi_min, value=idi_min, type="number", style={"width": 80}),
            html.Span(" - "),
            dcc.Input(id=FILTER_COMPONENT_IDS.idi_max, value=idi_max, type="number", style={"width": 80})
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Population Range"),
            dcc.Input(id=FILTER_COMPONENT_IDS.population_min, value=population_min, type="number", style={"width": 80}),
            html.Span(" - "),
            dcc.Input(id=FILTER_COMPONENT_IDS.population_max, value=population_max, type="number", style={"width": 80})
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Crime Rate Range"),
            dcc.Input(id=FILTER_COMPONENT_IDS.crime_rate_min, value=crime_rate_min, type="number", style={"width": 80}),
            html.Span(" - "),
            dcc.Input(id=FILTER_COMPONENT_IDS.crime_rate_max, value=crime_rate_max, type="number", style={"width": 80})
        ], style={"marginBottom": "10px"}),

        html.Div(id='total_results', style={'paddingTop': 20, 'fontStyle': 'italic', 'minHeight': 45})
    ])

    # Test inputs
    test_inputs = {k: {"options": [0], "default": 0} for k in FILTER_CALLBACK_INPUTS}  # puedes personalizar

    return {"layout": layout, "test_inputs": test_inputs}


# --------------------------
# Función de filtrado
# --------------------------
def filter_data(df, **filters):
    df = df.copy()
    if len(filters["region"]) > 0 and "all" not in filters["region"]:
        df = df[df["Región"].isin(filters["region"])]
    if len(filters["crime_type"]) > 0 and "all" not in filters["crime_type"]:
        df = df[df["Delito"].isin(filters["crime_type"])]
    if len(filters["municipality"]) > 0 and "all" not in filters["municipality"]:
        df = df[df["Comuna"].isin(filters["municipality"])]
    if "Año" in df.columns:
        df = df[(df["Año"] >= int(filters["year_min"])) & (df["Año"] <= int(filters["year_max"]))]
    if "Frecuencia" in df.columns:
        df = df[(df["Frecuencia"] >= int(filters["frequency_min"])) & (df["Frecuencia"] <= int(filters["frequency_max"]))]
    if "Índice Delincuencia Integrado (IDI)" in df.columns:
        df = df[(df["Índice Delincuencia Integrado (IDI)"] >= float(filters["idi_min"])) & 
                (df["Índice Delincuencia Integrado (IDI)"] <= float(filters["idi_max"]))]
    if "Población Estimada_numeric" in df.columns:
        df = df[df["Población Estimada_numeric"].notna()]
        df = df[(df["Población Estimada_numeric"] >= float(filters["population_min"])) & 
                (df["Población Estimada_numeric"] <= float(filters["population_max"]))]
    if "Tasa cada 100 mil hab" in df.columns:
        df = df[(df["Tasa cada 100 mil hab"] >= float(filters["crime_rate_min"])) & 
                (df["Tasa cada 100 mil hab"] <= float(filters["crime_rate_max"]))]
    return df

# --------------------------
# Callback para contar filas
# --------------------------
@callback(Output("total_results", "children"), inputs=FILTER_CALLBACK_INPUTS)
def display_count(**kwargs):
    df = get_data()
    count = len(df)
    filtered_count = len(filter_data(df, **kwargs))
    return f"{filtered_count:,} / {count:,} rows"
