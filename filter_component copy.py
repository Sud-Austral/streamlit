# PANDAS FILTER TEMPLATE - DO NOT CONVERT TO POLARS
# IMPORTANT: Always include the following imports as shown
import os
import sys
from datetime import datetime, time
from typing import TypedDict, Any

from dash import callback, html, dcc, Output, Input
import dash_design_kit as ddk
import numpy as np
import pandas as pd

from data import get_data
from logger import logger

class FILTER_COMPONENT_IDS:
    '''
    A map of all component IDs used in the filter.
    These should all be column names of columns that will be filtered.
    IMPORTANT - Use underscores not hyphens
    '''
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


# Map of callback inputs to their corresponding Input objects
# IMPORTANT - This must always be defined
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
    df = get_data()  # pandas dataframe

    # DEBUGGING: Log data loading information
    logger.debug("Filter component data loaded. Shape: %s", df.shape)
    logger.debug("Filter component sample data:\n%s", df.head())

    # Get unique regions
    unique_regions = df["Región"].dropna().replace('', np.nan).dropna().unique().tolist()
    unique_regions.sort()

    # Get unique crime types
    unique_crime_types = df["Delito"].dropna().replace('', np.nan).dropna().unique().tolist()
    unique_crime_types.sort()

    # Get unique municipalities
    unique_municipalities = df["Comuna"].dropna().replace('', np.nan).dropna().unique().tolist()
    unique_municipalities.sort()

    # Compute min/max for year
    year_min = int(df["Año"].min())
    year_max = int(df["Año"].max())

    # Compute min/max for frequency
    frequency_min = int(df["Frecuencia"].min())
    frequency_max = int(df["Frecuencia"].max())

    # Compute min/max for IDI
    idi_min = float(df["Índice Delincuencia Integrado (IDI)"].min())
    idi_max = float(df["Índice Delincuencia Integrado (IDI)"].max())

    # Compute min/max for population (using numeric column)
    population_min = int(df["Población Estimada_numeric"].dropna().min())
    population_max = int(df["Población Estimada_numeric"].dropna().max())

    # Compute min/max for crime rate
    crime_rate_min = float(df["Tasa cada 100 mil hab"].min())
    crime_rate_max = float(df["Tasa cada 100 mil hab"].max())

    # Create the filter panel layout
    layout = html.Div([ddk._ControlPanel(
        position="top",
        default_open=True,
        control_groups=[
            {
                "title": "Filters",
                "id": "filter_control_group",
                "description": "",
                "children": [
                    # Region filter
                    html.Div(
                        children=dcc.Dropdown(
                            id=FILTER_COMPONENT_IDS.region,
                            options=[{"label": "All", "value": "all"}] + [{"label": r, "value": r} for r in unique_regions],
                            multi=True,
                            value=["all"]
                        ),
                        id=FILTER_COMPONENT_IDS.region + "_parent",
                        title="Región",
                        style={"minWidth": "200px"}
                    ),

                    # Year range filter
                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.year_min, value=year_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.year_max, value=year_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Year Range"
                    ),

                    # Crime type filter
                    html.Div(
                        children=dcc.Dropdown(
                            id=FILTER_COMPONENT_IDS.crime_type,
                            options=[{"label": "All", "value": "all"}] + [{"label": c, "value": c} for c in unique_crime_types],
                            multi=True,
                            value=["all"]
                        ),
                        id=FILTER_COMPONENT_IDS.crime_type + "_parent",
                        title="Crime Type",
                        style={"minWidth": "200px"}
                    ),

                    # Municipality filter
                    html.Div(
                        children=dcc.Dropdown(
                            id=FILTER_COMPONENT_IDS.municipality,
                            options=[{"label": "All", "value": "all"}] + [{"label": m, "value": m} for m in unique_municipalities],
                            multi=True,
                            value=["all"]
                        ),
                        id=FILTER_COMPONENT_IDS.municipality + "_parent",
                        title="Municipality",
                        style={"minWidth": "200px"}
                    ),

                    # Crime frequency range filter
                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.frequency_min, value=frequency_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.frequency_max, value=frequency_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Crime Frequency Range"
                    ),

                    # IDI range filter
                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.idi_min, value=idi_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.idi_max, value=idi_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Integrated Delinquency Index Range"
                    ),

                    # Population range filter
                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.population_min, value=population_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.population_max, value=population_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Population Range"
                    ),

                    # Crime rate range filter
                    html.Div(
                        children=html.Div([
                            dcc.Input(id=FILTER_COMPONENT_IDS.crime_rate_min, value=crime_rate_min, debounce=True, style={"width": 100}),
                            html.Span(" - ", style={"margin": "0 8px", "alignSelf": "center"}),
                            dcc.Input(id=FILTER_COMPONENT_IDS.crime_rate_max, value=crime_rate_max, debounce=True, style={"width": 100})
                        ], style={
                            "display": "flex",
                            "alignItems": "center",
                            "flexWrap": "wrap"
                        }),
                        title="Crime Rate per 100k Range"
                    )
                ],
            },
        ],
    ), html.Div(id='total_results', style={ 'paddingTop': 20, 'marginLeft': 50, 'fontStyle': 'italic', 'minHeight': 45 })])

    # Create test inputs dictionary for component testing
    # All values from FILTER_CALLBACK_INPUTS must be present here
    test_inputs: dict[str, TestInput] = {
        "region": {
            "options": ["all"] + unique_regions[:3],
            "default": ["all"]
        },
        "year_min": {
            "options": [year_min, (year_min + year_max) // 2, year_max],
            "default": year_min
        },
        "year_max": {
            "options": [year_max, (year_min + year_max) // 2, year_min],
            "default": year_max
        },
        "crime_type": {
            "options": ["all"] + unique_crime_types[:3],
            "default": ["all"]
        },
        "municipality": {
            "options": ["all"] + unique_municipalities[:3],
            "default": ["all"]
        },
        "frequency_min": {
            "options": [frequency_min, (frequency_min + frequency_max) // 2, frequency_max],
            "default": frequency_min
        },
        "frequency_max": {
            "options": [frequency_max, (frequency_min + frequency_max) // 2, frequency_min],
            "default": frequency_max
        },
        "idi_min": {
            "options": [idi_min, (idi_min + idi_max) / 2, idi_max],
            "default": idi_min
        },
        "idi_max": {
            "options": [idi_max, (idi_min + idi_max) / 2, idi_min],
            "default": idi_max
        },
        "population_min": {
            "options": [population_min, (population_min + population_max) // 2, population_max],
            "default": population_min
        },
        "population_max": {
            "options": [population_max, (population_min + population_max) // 2, population_min],
            "default": population_max
        },
        "crime_rate_min": {
            "options": [crime_rate_min, (crime_rate_min + crime_rate_max) / 2, crime_rate_max],
            "default": crime_rate_min
        },
        "crime_rate_max": {
            "options": [crime_rate_max, (crime_rate_min + crime_rate_max) / 2, crime_rate_min],
            "default": crime_rate_max
        }
    }

    # Return both layout and test inputs
    return {
        "layout": layout,
        "test_inputs": test_inputs
    }

def filter_data(df, **filters):  # IMPORTANT - Keep this as **filters
    # Apply the filters from a callback to the pandas DataFrame and return a filtered pandas DataFrame
    # filters is a dictionary with columns and values where the keys are
    # the same keys as FILTER_INPUTS (e.g. "region", "year_min", "year_max", etc.)

    # DEBUGGING: Log filtering start and applied filters for debugging
    logger.debug("Starting data filtering. Original shape: %s", df.shape)
    logger.debug("Applied filters: %s", filters)

    df = df.copy()

    # IMPORTANT - All keys in FILTER_CALLBACK_INPUTS will always be present in the filters dictionary
    # No need to check if keys exist - just check if values meet criteria for applying filters

    # For dropdown/multi-select filters, check if "all" is not selected and list is not empty
    if len(filters["region"]) > 0 and "all" not in filters["region"]:
        df = df[df["Región"].isin(filters["region"])]

    if len(filters["crime_type"]) > 0 and "all" not in filters["crime_type"]:
        df = df[df["Delito"].isin(filters["crime_type"])]

    if len(filters["municipality"]) > 0 and "all" not in filters["municipality"]:
        df = df[df["Comuna"].isin(filters["municipality"])]

    # For range inputs, apply filters directly - filter values will always exist
    if "Año" in df.columns:
        df = df[df["Año"] >= int(filters["year_min"])]
        df = df[df["Año"] <= int(filters["year_max"])]

    if "Frecuencia" in df.columns:
        df = df[df["Frecuencia"] >= int(filters["frequency_min"])]
        df = df[df["Frecuencia"] <= int(filters["frequency_max"])]

    if "Índice Delincuencia Integrado (IDI)" in df.columns:
        df = df[df["Índice Delincuencia Integrado (IDI)"] >= float(filters["idi_min"])]
        df = df[df["Índice Delincuencia Integrado (IDI)"] <= float(filters["idi_max"])]

    if "Población Estimada_numeric" in df.columns:
        # Filter out null values first
        df = df[df["Población Estimada_numeric"].notna()]
        df = df[df["Población Estimada_numeric"] >= float(filters["population_min"])]
        df = df[df["Población Estimada_numeric"] <= float(filters["population_max"])]

    if "Tasa cada 100 mil hab" in df.columns:
        df = df[df["Tasa cada 100 mil hab"] >= float(filters["crime_rate_min"])]
        df = df[df["Tasa cada 100 mil hab"] <= float(filters["crime_rate_max"])]

    # DEBUGGING: Log filtering completion with final shape and sample data
    logger.debug("Filtering complete. Final shape: %s", df.shape)
    logger.debug("Filtered data sample:\n%s", df.head())

    return df

@callback(Output("total_results", "children"), inputs=FILTER_CALLBACK_INPUTS)
def display_count(**kwargs):
    df = get_data()
    # Get total count
    count = len(df)

    filtered_df = filter_data(df, **kwargs)
    # Get filtered count
    filtered_count = len(filtered_df)

    return f"{filtered_count:,} / {count:,} rows"