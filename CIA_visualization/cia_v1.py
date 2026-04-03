# =============================================================================
# CIA Energy Data Visualization Tool
# =============================================================================
"""
Energy Data Visualization Dashboard for Energy Analysts

This interactive visualization tool enables energy analysts to explore the CIA
Energy Dataset across 260 countries using coordinated multiple views.

FEATURES:
- Choropleth Map: Geographic distribution of energy indicators
- Scatter Plot: Bivariate correlation analysis
- Parallel Coordinates: Multivariate country profiling with brushing
- PCA + K-Means: Dimensionality reduction and automated clustering
- Correlation Matrix: Pairwise indicator relationships
- Bar Chart: Top-N country ranking
- Bubble Chart: 3-variable comparison
- Comparison Table: Side-by-side country statistics

LINKED VIEW INTERACTIONS:
- Click on Map/Scatter → Highlight in all views
- Brush Parallel Coordinates → Filter displayed countries
- Region dropdown → Filter all views by continent

USAGE:
    python cia_v1.py
    Open browser to http://127.0.0.1:8050/

DEPENDENCIES:
    - dash, plotly, pandas, numpy
    - pycountry, pycountry_convert (optional, for ISO lookup)
    - sklearn (optional, for PCA/clustering)

AUTHORS: Harshavardhan Dharman, Surya Kannan, Shashank Venkatesha, Leela Karthikeyan
DATE: December 2025
"""
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# -----------------------------
# THEME CONSTANTS - Professional Energy Dashboard
# -----------------------------
BG          = "#0a0b0d"          # Deep dark background
PANEL       = "#12141a"          # Slightly lighter panels
PANEL_DARK  = "#0d0e12"          # Darker accent panels
BORDER      = "rgba(255,255,255,0.06)"
TEXT        = "#f4f4f5"          # Crisp white text
TEXT_DIM    = "#71717a"          # Muted gray
TEXT_BRIGHT = "#fafafa"          # Bright accent text

# Energy-focused accent colors
ACCENT      = "#10b981"          # Emerald green (energy/sustainability)
ACCENT_ALT  = "#3b82f6"          # Electric blue (electricity)
ACCENT_WARM = "#f59e0b"          # Amber (oil/gas)
ACCENT_SOFT = "rgba(16,185,129,0.15)"
HILITE      = "#ef4444"          # Vibrant Red for highlights (Map, Rank)
HILITE_BRIGHT = "#ff0000"        # Pure Red for brighter highlights (Scatter)
HILITE_YELLOW = "#FFFF00"        # Pure Yellow for PCA highlights
DANGER      = "#ef4444"          # Red for warnings/CO2

# UI elements
FADE_LINE   = "rgba(255,255,255,0.08)"
GRADIENT_BG = "linear-gradient(135deg, #0a0b0d 0%, #12141a 100%)"
CARD_SHADOW = "0 4px 24px rgba(0,0,0,0.4)"
FONT_FAMILY = "'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"

pio.templates.default = "plotly_dark"

# -----------------------------
# DATASET CONFIG
# -----------------------------
DATASETS = {
    "Energy Analyst": "Datasets/energyAnalyst.csv",
}
GEOGRAPHY_DATASET = "Datasets/geography_data.csv"
COUNTRY_COL = "Country"
BASE_DIR = Path(__file__).resolve().parent

# Define ALL relevant columns by category for Energy Analyst
# Energy columns - core energy data
ENERGY_COLUMNS = [
    "electricity_access_percent",
    "electricity_generating_capacity_kW",
    "coal_metric_tons",
    "petroleum_bbl_per_day",
    "refined_petroleum_products_bbl_per_day",
    "refined_petroleum_exports_bbl_per_day",
    "refined_petroleum_imports_bbl_per_day",
    "natural_gas_cubic_meters",
    "carbon_dioxide_emissions_Mt",
]

# Economy columns - economic context
ECONOMY_COLUMNS = [
    "Real_GDP_PPP_billion_USD",
    "Real_GDP_per_Capita_USD",
    "Exports_billion_USD",
    "Imports_billion_USD",
]

# Demographics columns - population context
DEMOGRAPHICS_COLUMNS = [
    "Total_Population",
]

# Transport columns - infrastructure
TRANSPORT_COLUMNS = [
    "gas_pipelines_km",
    "oil_pipelines_km",
]

# Geography columns - geographical context
GEOGRAPHY_COLUMNS = [
    "Area_Total",
    "Land_Area",
    "Coastline",
    "Forest_Land",
    "Agricultural_Land",
]

# Combined columns for analysis
ALL_ANALYST_COLUMNS = ENERGY_COLUMNS + ECONOMY_COLUMNS + DEMOGRAPHICS_COLUMNS + TRANSPORT_COLUMNS

# Base ISO & continent from gapminder
GAP = px.data.gapminder()[["country", "iso_alpha", "continent"]].drop_duplicates()

# CIA → gapminder name fixes
CIA_NAME_FIXES = {
    "BAHAMAS, THE": "Bahamas",
    "BOLIVIA": "Bolivia",
    "BOSNIA AND HERZEGOVINA": "Bosnia and Herzegovina",
    "BRUNEI": "Brunei",
    "BURMA": "Myanmar",
    "CABO VERDE": "Cape Verde",
    "CONGO, DEMOCRATIC REPUBLIC OF THE": "Congo, Dem. Rep.",
    "CONGO, REPUBLIC OF THE": "Congo, Rep.",
    "COTE D'IVOIRE": "Cote d'Ivoire",
    "CZECHIA": "Czech Republic",
    "GAMBIA, THE": "Gambia",
    "IRAN": "Iran",
    "KOREA, NORTH": "Korea, Dem. Rep.",
    "KOREA, SOUTH": "Korea, Rep.",
    "LAOS": "Laos",
    "MACEDONIA": "North Macedonia",
    "MICRONESIA, FEDERATED STATES OF": "Micronesia",
    "NETHERLANDS, THE": "Netherlands",
    "RUSSIA": "Russia",
    "SYRIA": "Syrian Arab Republic",
    "TANZANIA": "Tanzania",
    "TURKEY (TURKIYE)": "Turkey",
    "UNITED STATES": "United States",
    "VENEZUELA": "Venezuela",
    "VIETNAM": "Vietnam",
}

# -----------------------------
# HELPERS
# -----------------------------
def clean_numeric_column(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False)
    for u in [" sq km", " km2", " km", " m ", "%", " million", " billion", " trillion", "$", " USD", " usd"]:
        s = s.str.replace(u, "", case=False, regex=False)
    s = s.str.strip().replace({"NA": np.nan, "": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")

def add_iso_and_continent(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COUNTRY_COL] = out[COUNTRY_COL].astype(str).str.strip()
    raw_upper = out[COUNTRY_COL].str.upper()
    out["_country_fixed"] = raw_upper.map(lambda x: CIA_NAME_FIXES.get(x, x.title()))
    out = out.merge(GAP, left_on="_country_fixed", right_on="country", how="left").drop(columns=["country"])
    # pycountry fallback
    try:
        import pycountry
        def lookup_iso(name):
            try:
                return pycountry.countries.lookup(name).alpha_3
            except Exception:
                try:
                    # Robust fallback: fuzzy search handles "Russia", "Turkey", etc.
                    results = pycountry.countries.search_fuzzy(name)
                    if results:
                        return results[0].alpha_3
                except Exception:
                    pass
                return np.nan
        missing_iso = out["iso_alpha"].isna()
        if missing_iso.any():
            out.loc[missing_iso, "iso_alpha"] = out.loc[missing_iso, "_country_fixed"].apply(lookup_iso)
    except Exception:
        pass

    # --- CRITICAL FIXES: Manual ISO Overrides for accurate mapping ---
    # Fixes Gapminder collisions and pycountry fuzzy-lookup territory errors
    iso_overrides = {
        "Korea, Dem. Rep.": "PRK",  # Gapminder often mis-maps this to KOR
        "Korea, Rep.": "KOR",
        "Jan Mayen": "SJM",        # pycountry fuzzy sometimes maps this to NOR
        "Curacao": "CUW",          # Maps to NLD in some datasets
        "Sint Maarten": "SXM",     # Maps to NLD in some datasets
        "Kosovo": "XKX",           # Not in all ISO standards but handles map displays
    }
    for name, iso in iso_overrides.items():
        mask = out["_country_fixed"] == name
        if mask.any():
            out.loc[mask, "iso_alpha"] = iso

    # pycountry_convert for continent (unify North/South America)
    try:
        import pycountry_convert as pcc
        # Force re-mapping continent for ALL countries with ISO to ensure consistency (e.g. split Americas)
        has_iso = out["iso_alpha"].notna()
        if has_iso.any():
            def iso3_to_cont(i3):
                try:
                    a2 = pcc.country_alpha3_to_country_alpha2(i3)
                    code = pcc.country_alpha2_to_continent_code(a2)
                    return dict(AF="Africa", AS="Asia", EU="Europe", NA="North America",
                                SA="South America", OC="Oceania", AN="Antarctica").get(code)
                except Exception:
                    return np.nan
            # Only update if we can't find it or if it's "Americas"
            # Actually, to be safe and consistent with filter options, let's just map all matched ISOs
            out.loc[has_iso, "continent"] = out.loc[has_iso, "iso_alpha"].apply(iso3_to_cont)
    except Exception:
        pass
    return out

# -----------------------------
# LOAD & CLEAN DATASETS
# -----------------------------
loaded = {}
for name, rel in DATASETS.items():
    path = BASE_DIR / rel
    try:
        df = pd.read_csv(path)
        df = add_iso_and_continent(df)
        skip = {COUNTRY_COL, "_country_fixed", "iso_alpha", "continent", "Geographic_Coordinates"}
        
        # Clean numeric columns
        for c in df.columns:
            if c not in skip:
                col = clean_numeric_column(df[c])
                if col.notna().any():
                    df[c] = col
        
        # Check for missing ISO codes before dropping
        missing_iso = df[df["iso_alpha"].isna()][COUNTRY_COL].unique()
        if len(missing_iso) > 0:
            print(f"  [WARN] Dropping {len(missing_iso)} countries due to missing ISO/Gapminder match: {missing_iso[:20]}")

        # Drop missing ISOs and handle duplicates by prioritizing entries with more data
        df = df.dropna(subset=["iso_alpha"])
        df["_data_count"] = df.notna().sum(axis=1)
        df = df.sort_values("_data_count", ascending=False).drop_duplicates(subset=["iso_alpha"]).drop(columns=["_data_count"])
        
        # Keep ALL analyst-relevant columns (Energy + Economy + Demographics + Transport)
        analyst_cols = [c for c in ALL_ANALYST_COLUMNS if c in df.columns]
        meta_cols = [COUNTRY_COL, "_country_fixed", "iso_alpha", "continent"]
        df = df[meta_cols + analyst_cols]
        
        # Merge Geography data
        geo_path = BASE_DIR / GEOGRAPHY_DATASET
        if geo_path.exists():
            geo_df = pd.read_csv(geo_path)
            geo_df[COUNTRY_COL] = geo_df[COUNTRY_COL].astype(str).str.strip()
            # Clean geography columns
            for c in GEOGRAPHY_COLUMNS:
                if c in geo_df.columns:
                    geo_df[c] = clean_numeric_column(geo_df[c])
            # Merge on country name (uppercase match)
            geo_df["_merge_key"] = geo_df[COUNTRY_COL].str.upper()
            df["_merge_key"] = df[COUNTRY_COL].str.upper()
            geo_cols_available = [c for c in GEOGRAPHY_COLUMNS if c in geo_df.columns]
            df = df.merge(geo_df[["_merge_key"] + geo_cols_available], on="_merge_key", how="left")
            df = df.drop(columns=["_merge_key"])
            analyst_cols = analyst_cols + geo_cols_available
            print(f"  Merged Geography: {len(geo_cols_available)} columns")
        
        # Merge Government Type data
        gov_path = BASE_DIR / "Datasets/government_and_civics_data.csv"
        if gov_path.exists():
            gov_df = pd.read_csv(gov_path)
            gov_df[COUNTRY_COL] = gov_df[COUNTRY_COL].astype(str).str.strip()
            # Merge on country name (uppercase match)
            gov_df["_merge_key"] = gov_df[COUNTRY_COL].str.upper()
            df["_merge_key"] = df[COUNTRY_COL].str.upper()
            if "Government_Type" in gov_df.columns:
                df = df.merge(gov_df[["_merge_key", "Government_Type"]], on="_merge_key", how="left")
                print(f"  Merged Government Type: {df['Government_Type'].notna().sum()} countries with data")
            df = df.drop(columns=["_merge_key"], errors="ignore")
        
        num_cols = analyst_cols
        num_cols.sort()
        loaded[name] = {"df": df, "num_cols": num_cols}
        print(f"Loaded {name}: {len(df)} rows, {len(num_cols)} total indicators")
        print(f"  Energy: {len([c for c in ENERGY_COLUMNS if c in df.columns])}")
        print(f"  Economy: {len([c for c in ECONOMY_COLUMNS if c in df.columns])}")
        print(f"  Demographics: {len([c for c in DEMOGRAPHICS_COLUMNS if c in df.columns])}")
        print(f"  Transport: {len([c for c in TRANSPORT_COLUMNS if c in df.columns])}")
        print(f"  Geography: {len([c for c in GEOGRAPHY_COLUMNS if c in df.columns])}")
    except FileNotFoundError:
        print(f"[WARN] Missing: {rel} — skipping.")

if not loaded:
    loaded["Demo"] = {"df": pd.DataFrame({
        COUNTRY_COL:["United States"], "_country_fixed":["United States"],
        "iso_alpha":["USA"], "continent":["North America"], "Value":[1.0]}),
        "num_cols": ["Value"]
    }

DATASET_OPTIONS = [{"label": k, "value": k} for k in loaded.keys()]
FIRST_DS = DATASET_OPTIONS[0]["value"]
COUNTRY_OPTIONS = [{"label": c, "value": c}
                   for c in sorted(set(np.concatenate([d["df"]["_country_fixed"].dropna().unique() for d in loaded.values()])))]

# Global country & ISO reference for bidirectional mapping
REF = pd.concat([d["df"][["_country_fixed", "iso_alpha", "continent"]] for d in loaded.values()], ignore_index=True)\
        .dropna(subset=["iso_alpha"]).drop_duplicates(subset=["iso_alpha"])
NAME_TO_ISO = dict(zip(REF["_country_fixed"], REF["iso_alpha"]))
ISO_TO_NAME = dict(zip(REF["iso_alpha"], REF["_country_fixed"]))

CONTINENT_OPTIONS = [{"label": "All", "value": "All"}] + [
    {"label": c, "value": c} for c in sorted(set(np.concatenate([d["df"]["continent"].dropna().unique() for d in loaded.values()])))
    if c != "Antarctica"
]

# -----------------------------
# APP + PROFESSIONAL DASHBOARD CSS
# -----------------------------
external_css = ['https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap']
app = Dash(__name__, external_stylesheets=external_css)
app.title = "Energy Analytics Dashboard"

app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Energy Analytics Dashboard</title>
    {%css%}
    <style>
    * { box-sizing: border-box; }
    body { margin: 0; padding: 0; background: #0a0b0d; }
    
    /* Professional Dropdown theming */
    .Select-control {
        background: linear-gradient(180deg, #1a1c22 0%, #15171c 100%) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #f4f4f5 !important;
        border-radius: 10px !important;
        min-height: 40px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }
    .Select-control:hover { border-color: rgba(16,185,129,0.4) !important; }
    .Select--single > .Select-control .Select-value, .Select-placeholder { color: #a1a1aa !important; }
    .Select input { background: transparent !important; color: #f4f4f5 !important; }
    .Select-menu-outer { 
        background: #15171c !important; 
        border: 1px solid rgba(255,255,255,0.08) !important; 
        border-radius: 10px !important; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
        margin-top: 4px !important;
    }
    .Select-menu { background: transparent !important; }
    .Select-option { 
        background: transparent !important; 
        color: #d4d4d8 !important; 
        padding: 10px 14px !important;
        transition: all 0.15s ease !important;
    }
    .Select-option:hover { background: rgba(16,185,129,0.12) !important; color: #10b981 !important; }
    .Select-option.is-selected { background: rgba(16,185,129,0.2) !important; color: #10b981 !important; }
    .Select-value { background: transparent !important; border: none !important; }
    .Select-value-label { color: #f4f4f5 !important; font-weight: 500 !important; }
    .Select-arrow { border-color: #71717a transparent transparent !important; }
    
    /* Tab styling */
    .tab { 
        border: none !important; 
        background: transparent !important;
        color: #71717a !important;
        font-weight: 500 !important;
        padding: 12px 24px !important;
        transition: all 0.2s ease !important;
    }
    .tab:hover { color: #10b981 !important; }
    .tab--selected { 
        color: #10b981 !important; 
        border-bottom: 2px solid #10b981 !important;
        background: transparent !important;
    }

    /* Panel animations */
    .panel-hover { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
    .panel-hover:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    .fade-in { animation: fadeIn 0.5s ease forwards; }
    @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(8px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 12px; height: 12px; }
    ::-webkit-scrollbar-track { background: #0a0b0d; }
    ::-webkit-scrollbar-thumb { background: rgba(16, 185, 129, 0.4); border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(16, 185, 129, 0.7); }
    
    /* Analytics section scrollbar - different green shade */
    .analytics-scroll::-webkit-scrollbar { width: 12px; height: 12px; }
    .analytics-scroll::-webkit-scrollbar-track { background: #0a0b0d; }
    .analytics-scroll::-webkit-scrollbar-thumb { background: rgba(74, 222, 128, 0.5); border-radius: 6px; }
    .analytics-scroll::-webkit-scrollbar-thumb:hover { background: rgba(74, 222, 128, 0.8); }
    </style>
    {%favicon%}
    {%scripts%}
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""

NAV_STYLE = {
    "height": "70px", "display": "flex", "alignItems": "center",
    "justifyContent": "space-between", "padding": "0 32px",
    "background": "linear-gradient(180deg, #12141a 0%, #0d0e12 100%)",
    "borderBottom": f"1px solid {BORDER}",
    "position": "sticky", "top": 0, "zIndex": 100
}
DD_STYLE = {
    "width": "100%",
    "color": TEXT,
    "backgroundColor": "transparent",
    "border": "none",
    "fontFamily": FONT_FAMILY,
    "fontSize": "14px",
}
BTN_ICON = {
    "position": "absolute",
    "top": "12px",
    "right": "12px",
    "zIndex": 5,
    "background": "rgba(16,185,129,0.1)",
    "border": f"1px solid rgba(16,185,129,0.3)",
    "borderRadius": "8px",
    "cursor": "pointer",
    "color": ACCENT,
    "fontSize": "14px",
    "padding": "6px 10px",
    "transition": "all 0.2s ease",
}
CARD_STYLE = {
    "background": "linear-gradient(180deg, #15171c 0%, #12141a 100%)",
    "border": f"1px solid {BORDER}",
    "borderRadius": "16px",
    "padding": "20px 24px",
    "boxShadow": CARD_SHADOW,
    "transition": "all 0.3s ease",
}

# -----------------------------
# LAYOUT
# -----------------------------
app.layout = html.Div(style={"background": BG, "color": TEXT, "minHeight": "100vh", "fontFamily": FONT_FAMILY}, children=[
    dcc.Store(id="selected-iso", data=None),        # << global clicked ISO-3 from Map/Scatter
    dcc.Store(id="pc-brush", data={}),              # << stores parcoords brush constraints
    dcc.Store(id="selected-iso-store", data=[]),       # list of iso_alpha for highlight/selection
    dcc.Store(id="highlight-country-names", data=[]),  # list of _country_fixed names

    # Professional Header
    html.Div(style=NAV_STYLE, children=[
        # Logo/Brand - Centered and larger title
        html.Div(style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "12px", "width": "100%"}, children=[

            html.Div(style={"textAlign": "center"}, children=[
                html.Div("Energy Analytics", style={"fontWeight": 700, "fontSize": "32px", "color": TEXT_BRIGHT, "letterSpacing": "-0.5px", "marginBottom": "4px"}),
                html.Div("Global Energy Intelligence Platform", style={"fontSize": "14px", "color": TEXT_DIM, "fontWeight": 400})
            ])
        ]),
    ]),

    # Control Bar with labels
    html.Div(style={
        "display": "grid", "gridTemplateColumns": "1fr 1fr 160px",
        "gap": "20px", "padding": "20px 32px",
        "background": "linear-gradient(180deg, #0d0e12 0%, #0a0b0d 100%)",
        "borderBottom": f"1px solid {BORDER}"
    }, children=[
        html.Div(children=[
            html.Label("Select Indicator", style={"fontSize": "11px", "color": TEXT_DIM, "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "8px", "display": "block", "fontWeight": 600}),
            dcc.Dropdown(id="ind", placeholder="Choose an indicator...", clearable=False, style=DD_STYLE),
        ]),
        html.Div(children=[
            html.Label("Filter by Region", style={"fontSize": "11px", "color": TEXT_DIM, "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "8px", "display": "block", "fontWeight": 600}),
            dcc.Dropdown(id="region", options=CONTINENT_OPTIONS, value="All", clearable=False, style=DD_STYLE),
        ]),
        html.Div(style={"display": "flex", "alignItems": "flex-end"}, children=[
            dcc.Checklist(id="log-scale", options=[{"label": " Log Scale", "value": "log"}],
                          value=[], style={"display": "flex", "alignItems": "center", "fontSize": "13px", "color": TEXT_DIM})
        ])
    ]),

    # KPI Cards Row
    html.Div(id="kpi-cards", style={
        "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "20px",
        "padding": "24px 32px", "background": BG
    }),

    # Tabs
    dcc.Tabs(id="tabs", value="tab-map", parent_style={"background": BG}, children=[
        dcc.Tab(label="Map", value="tab-map", selected_style={"background": PANEL}, style={"background": PANEL},
                children=[
                    html.Div(style={"padding": "10px 20px"}, children=[
                        html.Div(style={"position": "relative"}, children=[
                            html.Button("⤢", id="open-fullscreen-map", style=BTN_ICON),
                            dcc.Graph(id="map", style={"height": "70vh"}, config={"displayModeBar": False})
                        ])
                    ]),
                    html.Div(id="note", style={"color": TEXT_DIM, "fontSize": "12px", "padding": "0 20px 16px"})
                ]),

        dcc.Tab(label="Analytics", value="tab-analytics", selected_style={"background": PANEL}, style={"background": PANEL},
                children=[
                    html.Div(className="analytics-scroll", style={"padding": "24px", "height": "82vh", "overflowY": "auto", "scrollBehavior": "smooth"}, children=[

                        # Scatter Panel
                        html.Div(className="panel-hover fade-in", style={
                            "background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "14px",
                            "padding": "18px", "marginBottom": "28px", "boxShadow": "0 6px 20px rgba(0,0,0,0.25)",
                            "transition": "0.25s ease",
                        }, children=[
                            html.H3("Scatter Analysis", style={"margin": "0 0 4px 0", "fontSize": "18px", "opacity": 0.85}),
                            html.P("💡 Click legend to filter • Click points to highlight • Drag to zoom • Double-click to reset",
                                   style={"fontSize": "12px", "color": TEXT_DIM, "margin": "0 0 16px 0"}),
                            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginBottom": "12px"}, children=[
                                dcc.Dropdown(id="scatter-x", clearable=False, style=DD_STYLE),
                                dcc.Dropdown(id="scatter-y", clearable=False, style=DD_STYLE),
                            ]),
                            html.Div(style={"position": "relative"}, children=[
                                html.Button("⤢", id="open-fullscreen-scatter", style=BTN_ICON),
                                dcc.Graph(id="scatter", style={"height": "45vh"}, config={"displayModeBar": False}),
                            ]),
                        ]),

                        # Ranking Panel
                        html.Div(className="panel-hover fade-in", style={
                            "background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "14px",
                            "padding": "18px", "marginBottom": "28px", "boxShadow": "0 6px 20px rgba(0,0,0,0.25)",
                            "transition": "0.25s ease",
                        }, children=[
                            html.H3(id="rank-heading", children="Top-N Ranking", style={"margin": "0 0 12px 0", "fontSize": "18px", "opacity": 0.85}),
                            html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "12px", "alignItems": "center"}, children=[
                                html.Div("Top N:", style={"color": TEXT_DIM}),
                                dcc.Input(id="topn", type="number", value=10, min=3, max=50, step=1,
                                          style={"width": "130px", "background": "#1b1d22", "color": TEXT,
                                                 "border": f"1px solid {BORDER}", "borderRadius": "6px", "padding": "6px"}),
                            ]),
                            html.Div(style={"position": "relative"}, children=[
                                html.Button("⤢", id="open-fullscreen-rank", style=BTN_ICON),
                                dcc.Graph(id="rank", style={"height": "45vh"}, config={"displayModeBar": False}),
                            ]),
                        ]),

                        # Correlation Panel
                        html.Div(className="panel-hover fade-in", style={
                            "background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "14px",
                            "padding": "18px", "marginBottom": "28px", "boxShadow": "0 6px 20px rgba(0,0,0,0.25)",
                            "transition": "0.25s ease",
                        }, children=[
                            html.H3("Correlation Heat map", style={"margin": "0 0 12px 0", "fontSize": "18px", "opacity": 0.85}),
                            html.Div(style={"marginBottom": "12px"}, children=[
                                html.Div("Select Parameters:", style={"color": TEXT_DIM, "marginBottom": "6px", "fontSize": "12px"}),
                                dcc.Dropdown(id="corr-params", multi=True, style=DD_STYLE),
                            ]),
                            html.Div(style={"position": "relative"}, children=[
                                html.Button("⤢", id="open-fullscreen-corr", style=BTN_ICON),
                                dcc.Graph(id="corr", style={"height": "50vh"}, config={"displayModeBar": False}),
                            ]),
                        ]),

                        # SPLOM (Scatter Plot Matrix) Panel - Multivariate pairwise relationships
                        html.Div(className="panel-hover fade-in", style={
                            "background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "14px",
                            "padding": "18px", "marginBottom": "28px", "boxShadow": "0 6px 20px rgba(0,0,0,0.25)",
                            "transition": "0.25s ease", "display": "none"
                        }, children=[
                            html.H3("Scatter Plot Matrix (SPLOM)", style={"margin": "0 0 12px 0", "fontSize": "18px", "opacity": 0.85}),
                            html.P("Pairwise relationships between top 5 energy indicators", style={"color": TEXT_DIM, "fontSize": "12px", "margin": "0 0 12px 0"}),
                            html.Div(style={"position": "relative"}, children=[
                                html.Button("⤢", id="open-fullscreen-splom", style=BTN_ICON),
                                dcc.Graph(id="splom", style={"height": "70vh"}, config={"displayModeBar": False}),
                            ]),
                        ]),


                        # Parallel Coordinates Panel (enhanced)
                        html.Div(className="panel-hover fade-in", style={
                            "background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "14px",
                            "padding": "18px", "marginBottom": "10px", "boxShadow": "0 6px 20px rgba(0,0,0,0.25)",
                            "transition": "0.25s ease",
                        }, children=[
                            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                                html.H3("Parallel Coordinates", style={"margin": "0 0 12px 0"}),
                                html.Div(children=[
                                    html.Button("Reset changes", id="reset-parcoords", n_clicks=0,
                                                style={"background": ACCENT, "border": f"1px solid {BORDER}", "color": "white",
                                                    "padding": "6px 10px", "borderRadius": "8px", "cursor": "pointer",
                                                    "fontSize": "13px", "marginRight": "8px"}),
                                ], style={"position": "relative", "height": "0"})
                            ]),

                            # Parcoords container — allow horizontal scrolling and let graph be wider
                            html.Div(
                                style={
                                    # allow horizontal scroll when figure is wider than panel
                                    "overflowX": "auto",
                                    "overflowY": "hidden",
                                    "whiteSpace": "nowrap",
                                    "border": f"1px solid {BORDER}",
                                    "borderRadius": "12px",
                                    "padding": "6px",
                                    "width": "100%",
                                },
                                children=[
                                    dcc.Graph(
                                        id="parcoords",
                                        style={
                                            # default size; will be updated dynamically by callback
                                            "height": "52vh",
                                            "width": "4200px",
                                        },
                                        config={"displayModeBar": False, "responsive": True}
                                    )
                                ]
                            ),

                            # Linked Table Header
                            html.Div(style={"marginTop": "16px", "marginBottom": "8px"}, children=[
                                html.H4("📊 Linked Table - Brushed Countries", style={
                                    "margin": "0", "fontSize": "14px", "fontWeight": "600", "color": ACCENT,
                                    "display": "flex", "alignItems": "center", "gap": "8px"
                                }),
                                html.P("Brush the axes above to filter and see matching countries with their attributes below.", 
                                       style={"margin": "4px 0 0 0", "fontSize": "12px", "color": TEXT_DIM})
                            ]),

                            # Country session list (replaced Tabs to avoid rendering tab header classes)
                            html.Div(id="pc-countries-list", style={
                                "padding": "16px", 
                                "color": TEXT_DIM, 
                                "fontSize": "13px", 
                                "minHeight": "150px",
                                "maxHeight": "300px",
                                "marginTop": "16px",
                                "background": PANEL_DARK,
                                "borderRadius": "12px",
                                "border": f"1px solid {BORDER}",
                                "overflowY": "auto",
                                "overflowX": "auto"
                            }),
                        ]),

                            ])
                        ]),
  

        dcc.Tab(label="PCA / Clusters", value="tab-pca", selected_style={"background": PANEL}, style={"background": PANEL},
                children=[
                    html.Div(style={"padding": "16px 20px"}, children=[
                        html.Div(style={"marginBottom": "12px"}, children=[
                            html.Div("Select Parameters:", style={"color": TEXT_DIM, "marginBottom": "6px", "fontSize": "12px"}),
                            dcc.Dropdown(id="pca-params", multi=True, style=DD_STYLE),
                        ]),
                        html.Div(style={"position": "relative"}, children=[
                            html.Button("⤢", id="open-fullscreen-pca", style=BTN_ICON),
                            dcc.Graph(id="pca", style={"height": "75vh"}, config={"displayModeBar": False})
                        ])
                    ])
                ]),
    ]),

    # GLOBAL FULLSCREEN MODAL
    html.Div(
        id="fullscreen-modal",
        style={
            "position": "fixed", "top": "50%", "left": "50%", "transform": "translate(-50%, -50%)",
            "width": "95vw", "height": "95vh", "maxWidth": "95vw", "maxHeight": "95vh",
            "background": "rgba(0,0,0,0.92)", "zIndex": 9999, "display": "none",
            "padding": "20px", "backdropFilter": "blur(6px)", "borderRadius": "10px",
        },
        children=[
            html.Button("×", id="close-fullscreen",
                        style={"position": "absolute", "top": "10px", "right": "10px", "fontSize": "30px",
                               "background": "none", "border": "none", "color": "white", "cursor": "pointer"}),
            dcc.Graph(id="fullscreen-graph", style={"height": "calc(100% - 40px)", "width": "100%"}, config={"displayModeBar": False}),
        ]
    ),
])

# -----------------------------
# CALLBACKS — Controls
# -----------------------------
@app.callback(
    Output("ind", "options"), Output("ind", "value"),
    Output("scatter-x", "options"), Output("scatter-x", "value"),
    Output("scatter-y", "options"), Output("scatter-y", "value"),
    Input("region", "value"), State("ind", "value"), State("scatter-x", "value"), State("scatter-y", "value")
)
def set_indicators(region, ind_cur, x_cur, y_cur):
    ds = FIRST_DS  # Use single dataset
    cols = loaded[ds]["num_cols"]
    opts = [{"label": c.replace("_"," ").title(), "value": c} for c in cols]
    x = x_cur if x_cur in cols else (cols[0] if cols else None)
    y = y_cur if (y_cur in cols and y_cur != x) else (cols[1] if len(cols) > 1 else (cols[0] if cols else None))
    ind_val = ind_cur if ind_cur in cols else (cols[0] if cols else None)
    return opts, ind_val, opts, x, opts, y

# -----------------------------
# UPDATE HIGHLIGHT COUNTRIES DROPDOWN OPTIONS BASED ON REGION
# --------------------
# MAP CLICK SELECTION (for multi-select highlighting)
# --------------------
@app.callback(
    Output("highlight-country-names", "data"),
    Output("selected-iso-store", "data"),
    Input("map", "clickData"),
    Input("scatter", "clickData"),
    Input("rank", "clickData"),
    Input("pca", "clickData"),
    State("highlight-country-names", "data"),
    State("selected-iso-store", "data"),
    prevent_initial_call=True
)
def sync_highlights(map_click, scatter_click, rank_click, pca_click, prev_names, prev_iso):
    trigger = ctx.triggered_id
    names = set(prev_names or [])
    isos = set(prev_iso or [])
    
    # No need to load df here as we use customdata for ISOs
    iso = None
    
    if trigger == "map" and map_click:
        iso = map_click["points"][0].get("location")
        
    elif trigger == "scatter" and scatter_click:
        iso = scatter_click["points"][0].get("customdata")
                
    elif trigger == "rank" and rank_click:
        cd = rank_click["points"][0].get("customdata")
        # px.bar customdata is usually a list
        if isinstance(cd, list) and len(cd) > 0:
            iso = cd[0]
        else:
            iso = cd
                
    elif trigger == "pca" and pca_click:
        iso = pca_click["points"][0].get("customdata")

    if iso:
        # toggle iso; update names set accordingly (if known)
        if iso in isos:
            isos.remove(iso)
            nm = ISO_TO_NAME.get(iso)
            if nm in names:
                names.remove(nm)
        else:
            isos.add(iso)
            nm = ISO_TO_NAME.get(iso)
            if nm:
                names.add(nm)

    return sorted(names), sorted(isos)

    if iso:
        # toggle iso; update names set accordingly (if known)
        if iso in isos:
            isos.remove(iso)
            nm = ISO_TO_NAME.get(iso)
            if nm in names:
                names.remove(nm)
        else:
            isos.add(iso)
            nm = ISO_TO_NAME.get(iso)
            if nm:
                names.add(nm)

    return sorted(names), sorted(isos)
# MAP
# --------------------
@app.callback(
    Output("map", "figure"), Output("note", "children"),
    Input("ind", "value"), Input("region", "value"), Input("log-scale", "value"),
    Input("selected-iso-store", "data"),  # highlight overlay
    State("map", "figure")
)
def render_map(ind, region, logv, selected_iso, cur_fig):
    ds = FIRST_DS  # Use single dataset
    if not ind: return go.Figure(), "No indicator."
    df = loaded[ds]["df"][["_country_fixed", "iso_alpha", "continent", ind]].copy()
    if region and region != "All":
        df = df[df["continent"] == region]
    notes = []
    if "log" in logv:
        nonpos = df[ind] <= 0
        if nonpos.any():
            notes.append(f"{int(nonpos.sum())} values ≤ 0 hidden for log10.")
            df.loc[nonpos, ind] = np.nan
        df["val"] = np.log10(df[ind])
        lab = f"Log10 {ind.replace('_',' ').title()}"
    else:
        df["val"] = df[ind]
        lab = ind.replace("_"," ").title()

    fig = px.choropleth(
        df, locations="iso_alpha", color="val", hover_name="_country_fixed",
        custom_data=[ind],
        color_continuous_scale="Viridis", labels={"val": lab}, locationmode="ISO-3", title=""
    )
    # Standardize hover to ensure country name and raw value are always shown
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br>" +
                      f"{ind.replace('_',' ').title()}: %{{customdata[0]:,.2f}}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)", uirevision="map",
        coloraxis_colorbar=dict(title=dict(text=lab, font=dict(color=TEXT)))
    )
    fig.update_geos(showframe=False, showcoastlines=True, coastlinecolor="#333",
                    showland=True, landcolor="#1a1a1d", showocean=True, oceancolor=BG)
    if cur_fig and "layout" in cur_fig and "geo" in cur_fig["layout"]:
        geo = cur_fig["layout"]["geo"]
        fig.update_geos(center=geo.get("center"), projection_scale=geo.get("projection_scale"), fitbounds=None)

    # overlay for highlighted iso (gold outline, transparent fill)
    sel = [iso for iso in (selected_iso or []) if iso in df["iso_alpha"].values]
    if sel:
        overlay = go.Choropleth(
            locations=sel, z=[1]*len(sel), locationmode="ISO-3",
            showscale=False, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker_line_width=2.5, marker_line_color=HILITE,
            hoverinfo="skip"
        )
        fig.add_trace(overlay)

    unmatched = loaded[FIRST_DS]["df"][loaded[FIRST_DS]["df"]["iso_alpha"].isna()]["_country_fixed"].unique()
    if len(unmatched) > 0:
        sample = ", ".join(unmatched[:5])
        notes.append(f"Unmatched regions: {sample}{'...' if len(unmatched)>5 else ''}")
    return fig, "  |  ".join(notes)

# -----------------------------
# GLOBAL CLICK SELECTION (C): Map + Scatter
# -----------------------------
@app.callback(
    Output("selected-iso", "data"),
    Input("map", "clickData"),
    Input("scatter", "clickData"),
    Input("rank", "clickData"),
    Input("pca", "clickData"),
    prevent_initial_call=True
)
def update_selected_iso(map_click, scatter_click, rank_click, pca_click):
    ds = FIRST_DS  # Use single dataset
    trigger = ctx.triggered_id
    # No need to load df for lookup if we use customdata
    
    if trigger == "map" and map_click:
        iso = map_click["points"][0].get("location")
        return iso
        
    if trigger == "scatter" and scatter_click:
        return scatter_click["points"][0].get("customdata")
                
    if trigger == "rank" and rank_click:
        cd = rank_click["points"][0].get("customdata")
        if isinstance(cd, list) and len(cd) > 0:
            return cd[0]
        return cd
                
    if trigger == "pca" and pca_click:
        return pca_click["points"][0].get("customdata")
                
    return no_update

# CORRELATION MATRIX PARAMETERS
# -----------------------------
@app.callback(
    Output("corr-params", "options"),
    Output("corr-params", "value"),
    Input("region", "value")
)
def update_corr_params(region):
    ds = FIRST_DS
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]
    
    num_cols = sorted([c for c in base.select_dtypes(include="number").columns if c != "iso_alpha"])
    options = [{"label": c.replace("_", " ").title(), "value": c} for c in num_cols]
    # Default: select first 8 parameters
    default_value = num_cols[:8] if len(num_cols) > 0 else []
    return options, default_value

# Update correlation matrix based on selected parameters
@app.callback(
    Output("corr", "figure"),
    Input("corr-params", "value"),
    Input("region", "value")
)
def update_corr_matrix(selected_params, region):
    ds = FIRST_DS
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]
    
    cfig = go.Figure()
    
    # Use selected parameters, or all numeric columns if none selected
    if not selected_params:
        num_cols = [c for c in base.select_dtypes(include="number").columns if c != "iso_alpha"]
    else:
        num_cols = selected_params
    
    if len(num_cols) >= 2:
        corr = base[num_cols].corr().round(2)
        cfig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", origin="lower")
        cfig.update_layout(margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    else:
        cfig.update_layout(title="Select at least 2 parameters", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    return cfig

# PCA PARAMETERS
# -----------------------------
@app.callback(
    Output("pca-params", "options"),
    Output("pca-params", "value"),
    Input("region", "value")
)
def update_pca_params(region):
    ds = FIRST_DS
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]
    
    num_cols = sorted([c for c in base.select_dtypes(include="number").columns if c != "iso_alpha"])
    options = [{"label": c.replace("_", " ").title(), "value": c} for c in num_cols]
    # Default: select first 8 parameters
    default_value = num_cols[:8] if len(num_cols) > 0 else []
    return options, default_value

# -----------------------------
# ANALYTICS (Rank, Scatter, Parcoords)
# -----------------------------
@app.callback(
    Output("rank", "figure"), Output("scatter", "figure"),
    Output("parcoords", "figure"), Output("parcoords", "style"),
    Output("pc-brush", "data"), Output("rank-heading", "children"),  # keep last brush ranges (also resettable)
    Input("ind", "value"), Input("region", "value"),
    Input("topn", "value"), Input("scatter-x", "value"), Input("scatter-y", "value"),
    Input("parcoords", "restyleData"),  # Changed from relayoutData to restyleData
    Input("reset-parcoords", "n_clicks"),
    Input("selected-iso", "data"), Input("selected-iso-store", "data"),
    State("pc-brush", "data")
)
def analytics(ind, region, topn, sx, sy, pc_restyle, reset_clicks, selected_iso, selected_iso_store, pc_brush_state):
    ds = FIRST_DS  # Use single dataset
    # Safety check: if no indicator selected, return empty figures
    if not ind:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return empty_fig, empty_fig, empty_fig, {"height": "52vh", "width": "100%"}, {}, "Top-N Ranking"
    
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]

    # Map selected ISO → name set (using multi-select store)
    sel_iso = set(selected_iso_store or [])
    sel_names = {ISO_TO_NAME[i] for i in sel_iso if i in ISO_TO_NAME}

    # Determine what triggered the callback
    trigger_id = ctx.triggered_id
    
    # Initialize figures with existing state if possible (to avoid rebuilding)
    # Note: In a real app we'd pass current figures as State, but for now we'll rebuild 
    # unless we can skip. 
    
    # ... (existing setup code) ...
    
    # ----- Rank & Scatter -----
    # Only rebuild Rank and Scatter if their specific inputs changed or if it's an initial load
    # Inputs affecting Rank: ind, topn, region
    # Inputs affecting Scatter: sx, sy, region, selected_iso, selected_iso_store
    
    rank_inputs = ["ind", "topn", "region", "selected-iso", "selected-iso-store"]
    scatter_inputs = ["scatter-x", "scatter-y", "region", "selected-iso", "selected-iso-store"]
    
    # Always build if no trigger (initial) or if relevant inputs changed
    build_rank = not trigger_id or any(x in trigger_id for x in rank_inputs)
    build_scatter = not trigger_id or any(x in trigger_id for x in scatter_inputs)
    
    # If we're just brushing or resetting parcoords, we might want to preserve existing figures
    # But since we don't have them as State, we have to rebuild them. 
    # However, the lag is likely due to the heavy parcoords logic.
    
    # Let's proceed with rebuilding for now but optimize the parcoords part below.
    
    # ----- Rank -----
    if build_rank:
        topn = max(3, min(50, int(topn or 10)))
        bar_df = base[["_country_fixed", "iso_alpha", ind]].dropna().sort_values(ind, ascending=False).head(topn).copy()
        bar_df["is_sel"] = bar_df["_country_fixed"].isin(sel_names)
        colors = np.where(bar_df["is_sel"], HILITE, ACCENT)
        rfig = px.bar(bar_df, x=ind, y="_country_fixed", orientation="h", color_discrete_sequence=[ACCENT], custom_data=["iso_alpha"])
        rfig.update_traces(marker_color=colors, hovertemplate="%{y}: %{x:,.2f}<extra></extra>")
        rfig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(autorange="reversed"),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        heading = f"Top-N ranking based on {ind.replace('_', ' ').title()}" if ind else "Top-N Ranking"
    else:
        rfig = no_update
        heading = no_update

    # Determine what triggered the callback
    trigger_id = ctx.triggered_id
    
    # ---- Parse Brush Ranges (Moved to top to ensure figure reflects new state) ----
    new_brush_state = pc_brush_state or {}
    
    # restyleData format: [{property: value, ...}, [trace_indices]] or None
    if pc_restyle and isinstance(pc_restyle, list) and len(pc_restyle) > 0:
        print(f"[DEBUG] pc_restyle received: {pc_restyle}")
        restyle_dict = pc_restyle[0] if isinstance(pc_restyle[0], dict) else {}
        
        # We need num_cols to map indices to column names
        num_cols = loaded[ds]["num_cols"]
        use_cols = [c for c in num_cols if c != "iso_alpha"]  # Show all attributes (no limit)

        # Look for dimensions.constraintrange updates
        for k, v in restyle_dict.items():
            if "dimensions" in k and "constraintrange" in k:
                try:
                    # Extract dimension index from key like "dimensions[0].constraintrange"
                    idx = int(k.split("[")[1].split("]")[0])
                    if 0 <= idx < len(use_cols):
                        col_name = use_cols[idx]
                        
                        # Handle clearing brush (v is None or empty)
                        if v is None or (isinstance(v, list) and len(v) == 0):
                            if col_name in new_brush_state:
                                del new_brush_state[col_name]
                            print(f"[DEBUG] Cleared brush for {col_name}")
                        
                        # Handle setting brush
                        elif isinstance(v, list):
                            # Handle nested array format: [[min, max]] -> [min, max]
                            if len(v) > 0 and isinstance(v[0], list):
                                v = v[0]
                            
                            new_brush_state[col_name] = v
                            print(f"[DEBUG] Brush constraint for {col_name}: {v}")
                except Exception as e:
                    print(f"[DEBUG] Error parsing brush constraint: {e}")
                    pass

    # If the reset button was pressed, clear stored brush ranges
    if ctx.triggered_id == "reset-parcoords":
        new_brush_state = {}

    # Initialize figures with existing state if possible (to avoid rebuilding)
    # Note: In a real app we'd pass current figures as State, but for now we'll rebuild 
    # unless we can skip. 
    
    # ... (existing setup code) ...
    
    # ----- Rank & Scatter -----
    # Only rebuild Rank and Scatter if their specific inputs changed or if it's an initial load
    # Inputs affecting Rank: ind, topn, region
    # Inputs affecting Scatter: sx, sy, region, selected_iso, selected_iso_store
    
    rank_inputs = ["ind", "topn", "region", "selected-iso", "selected-iso-store"]
    scatter_inputs = ["scatter-x", "scatter-y", "region", "selected-iso", "selected-iso-store"]
    
    # Always build if no trigger (initial) or if relevant inputs changed
    build_rank = not trigger_id or any(x in trigger_id for x in rank_inputs)
    build_scatter = not trigger_id or any(x in trigger_id for x in scatter_inputs)
    
    # If we're just brushing or resetting parcoords, we might want to preserve existing figures
    # But since we don't have them as State, we have to rebuild them. 
    # However, the lag is likely due to the heavy parcoords logic.
    
    # Let's proceed with rebuilding for now but optimize the parcoords part below.
    
    # ----- Rank -----
    if build_rank:
        topn = max(3, min(50, int(topn or 10)))
        bar_df = base[["_country_fixed", "iso_alpha", ind]].dropna().sort_values(ind, ascending=False).head(topn).copy()
        bar_df["is_sel"] = bar_df["_country_fixed"].isin(sel_names)
        colors = np.where(bar_df["is_sel"], HILITE, ACCENT)
        rfig = px.bar(bar_df, x=ind, y="_country_fixed", orientation="h", color_discrete_sequence=[ACCENT], custom_data=["iso_alpha"])
        rfig.update_traces(marker_color=colors, hovertemplate="%{y}: %{x:,.2f}<extra></extra>")
        rfig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(autorange="reversed"),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        heading = f"Top-N ranking based on {ind.replace('_', ' ').title()}" if ind else "Top-N Ranking"
    else:
        rfig = no_update
        heading = no_update

    # ----- Scatter -----
    if build_scatter:
        sfig = go.Figure()
        if sx and sy and sx in base.columns and sy in base.columns:
            df_all = base[["_country_fixed", "iso_alpha", sx, sy, "Government_Type"]].dropna(subset=[sx, sy]).copy()
            df_all["Government_Type"] = df_all["Government_Type"].fillna("Unknown")
            df_all["is_sel"] = df_all["_country_fixed"].isin(sel_names)

            # Create color map for government types
            palette = px.colors.qualitative.Set2
            gov_types = sorted(df_all["Government_Type"].unique())
            gov_color_map = {gov: palette[i % len(palette)] for i, gov in enumerate(gov_types)}
            
            # Plot each government type - combine selected and non-selected in one trace
            for gov_type in gov_types:
                gov_data = df_all[df_all["Government_Type"] == gov_type]
                
                # Separate selected and non-selected
                non_sel = gov_data[~gov_data["is_sel"]]
                sel = gov_data[gov_data["is_sel"]]
                
                # Combine all points for this government type
                combined = pd.concat([non_sel, sel])
                
                # Create size and line properties based on selection
                sizes = [13 if is_sel else 9 for is_sel in combined["is_sel"]]
                line_colors = [HILITE_BRIGHT if is_sel else "rgba(0,0,0,0)" for is_sel in combined["is_sel"]]
                line_widths = [2 if is_sel else 0 for is_sel in combined["is_sel"]]
                
                sfig.add_trace(go.Scatter(
                    x=combined[sx],
                    y=combined[sy],
                    mode="markers",
                    name=gov_type,
                    marker=dict(size=sizes, opacity=0.85, color=gov_color_map[gov_type],
                               line=dict(color=line_colors, width=line_widths)),
                    hovertext=combined["_country_fixed"],
                    customdata=combined["iso_alpha"],  # Add ISO for reliable click handling
                    hoverinfo="text+x+y"
                ))
            
            sfig.update_layout(
                margin=dict(l=10,r=10,t=40,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            sfig.update_xaxes(title_text=sx.replace("_"," ").title(), tickangle=0)
            sfig.update_yaxes(title_text=sy.replace("_"," ").title())
    else:
        sfig = no_update

    # ----- Parallel Coordinates -----
    # Get numeric columns from the dataset
    num_cols = loaded[ds]["num_cols"]
    use_cols = [c for c in num_cols if c != "iso_alpha"]  # Show all attributes (no limit)
    pfig = go.Figure()
    brush_names = []
    # new_brush_state is already computed above
    
    # default parcoords graph style (will be updated based on number of axes)
    pstyle = {"height": "52vh", "width": "100%"}

    if len(use_cols) >= 2:
        sub = base[["_country_fixed", "iso_alpha"] + use_cols].dropna()

        if not sub.empty:
            # Selected line highlight (support multi-selection via selected-iso-store)
            sel_set = set(selected_iso_store or [])
            color_mask = sub["iso_alpha"].isin(sel_set).astype(int) if sel_set else np.zeros(len(sub), int)

            # Build dimensions (VALID properties only!)
            # IMPORTANT: Must include 'range' property to enable brushing
            # Also restore previous brush constraints if they exist
            dims = []
            for c in use_cols:
                dim_config = {
                    'label': c.replace("_", " ").title(),
                    'values': sub[c],
                    'range': [sub[c].min(), sub[c].max()]  # Required for brushing to work!
                }
                # Restore previous brush constraint if it exists
                if c in new_brush_state and isinstance(new_brush_state[c], list) and len(new_brush_state[c]) == 2:
                    dim_config['constraintrange'] = new_brush_state[c]
                dims.append(dim_config)

            pfig = go.Figure(go.Parcoords(
                line=dict(
                    color=color_mask,
                    colorscale=[[0, FADE_LINE], [1, ACCENT_SOFT]],
                    showscale=False
                ),
                dimensions=dims
            ))

            # Layout – increase top margin & use global font settings and add wider side margins
            # so first/last axis labels are not clipped by the canvas.
            pfig.update_layout(
                margin=dict(l=88, r=88, t=60, b=18),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT, size=12),  # slightly smaller to reduce overlap
                autosize=True,
                # Include reset_clicks in uirevision to force reset when button is clicked
                uirevision=f"parcoords-{reset_clicks}" 
            )


            # ---- Compute Brushed Countries ----
            brushed_mask = np.ones(len(sub), dtype=bool)
            for col, rng in new_brush_state.items():
                if isinstance(rng, list) and len(rng) == 2:
                    lo, hi = float(rng[0]), float(rng[1])
                    brushed_mask &= (sub[col] >= lo) & (sub[col] <= hi)

            if brushed_mask.any() and any(new_brush_state.values()):
                brush_names = ["• " + n for n in sub.loc[brushed_mask, "_country_fixed"].tolist()[:40]]

            # Compute dynamic width to increase axis spacing: roughly 300px per axis
            try:
                axis_count = max(1, len(use_cols))
                # increase spacing: use 300px per axis with a larger minimum width
                width_px = max(1400, int(300 * axis_count))
                pstyle = {"height": "52vh", "width": f"{width_px}px", "minWidth": f"{width_px}px"}
            except Exception:
                pstyle = {"height": "52vh", "width": "100%"}

    brush_list = (["Brushed countries:"] + brush_names) if brush_names else "Brush on axes to see selected countries."
    print(f"[DEBUG] Returning new_brush_state: {new_brush_state}")  # Debug logging
    return rfig, sfig, pfig, pstyle, new_brush_state, heading

# -----------------------------
# POPULATE PC-COUNTRIES-LIST TAB (Linked Table)
# This creates a linked table showing brushed countries and their attribute values
@app.callback(
    Output("pc-countries-list", "children"),
    Input("selected-iso", "data"),
    Input("selected-iso-store", "data"),
    Input("pc-brush", "data"),
    Input("region", "value"),
    prevent_initial_call=False
)
def update_countries_tab(selected_iso, selected_iso_store, pc_brush_state, region):
    """Display selected and brushed countries in a linked table format."""
    print(f"[DEBUG] update_countries_tab called with pc_brush_state: {pc_brush_state}, region: {region}")  # Debug logging
    ds = FIRST_DS  # Use single dataset
    
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]
    
    result = []
    
    # Display selected country
    # Show multi-selected countries (from selected-iso-store) when present
    if selected_iso_store is not None:
        sel_isos = list(selected_iso_store)
        if sel_isos:
            sel_rows = base[base["iso_alpha"].isin(sel_isos)]["_country_fixed"].dropna().unique().tolist()
            if sel_rows:
                result.append(html.Div([
                    html.Span("★ Selected Countries: ", style={"color": ACCENT, "fontWeight": "600"}),
                    html.Span(", ".join(sel_rows), style={"fontWeight": "500"})
                ], style={"marginBottom": "12px", "fontSize": "14px"}))
    else:
        # fallback to single selected_iso only if store is None (never initialized)
        if selected_iso:
            selected_country = base[base["iso_alpha"] == selected_iso]["_country_fixed"].values
            if len(selected_country) > 0:
                result.append(html.Div([
                    html.Span("★ Selected Country: ", style={"color": ACCENT, "fontWeight": "600"}),
                    html.Span(selected_country[0], style={"fontWeight": "500"})
                ], style={"marginBottom": "12px", "fontSize": "14px"}))
    
    # Display brushed countries with linked table
    if pc_brush_state and isinstance(pc_brush_state, dict) and len(pc_brush_state) > 0:
        print(f"[DEBUG] Processing brush state with {len(pc_brush_state)} constraints")  # Debug logging
        try:
            # Use same column definition as analytics callback to ensure consistency
            num_cols = loaded[ds]["num_cols"]
            use_cols = [c for c in num_cols if c != "iso_alpha"]
            sub = base[["_country_fixed", "iso_alpha", "continent"] + use_cols].dropna()
            
            if not sub.empty and len(pc_brush_state) > 0:
                brushed_mask = np.ones(len(sub), dtype=bool)
                print(f"[DEBUG] Starting with {len(sub)} countries")  # Debug logging
                
                # Filter by brush constraints
                for col, rng in pc_brush_state.items():
                    if col in sub.columns and isinstance(rng, (list, tuple)) and len(rng) == 2:
                        try:
                            lo = float(rng[0])
                            hi = float(rng[1])
                            col_data = sub[col].fillna(np.nan)
                            col_mask = (col_data >= lo) & (col_data <= hi)
                            brushed_mask &= col_mask
                            print(f"[DEBUG] Applied constraint {col}: [{lo}, {hi}] -> {col_mask.sum()} countries match")  # Debug logging
                        except (ValueError, TypeError) as e:
                            print(f"[DEBUG] Error applying constraint for {col}: {e}")  # Debug logging
                            pass
                
                print(f"[DEBUG] After all constraints: {brushed_mask.sum()} countries match")  # Debug logging
                
                # If we have any brushed countries, display them
                if brushed_mask.any():
                    brushed_df = sub.loc[brushed_mask].copy()
                    brushed_df = brushed_df.head(40)  # Limit to 40 rows
                    
                    # Create header
                    result.append(html.Div([
                        html.Div("🔍 Brushed Countries (Linked Table):", 
                                style={"fontWeight": "700", "marginBottom": "12px", "color": ACCENT, "fontSize": "13px"})
                    ]))
                    
                    # Create table
                    table_header = html.Thead(
                        html.Tr([
                            html.Th("Country", style={"padding": "8px", "textAlign": "left", "borderBottom": f"1px solid {BORDER}", "color": TEXT_DIM, "fontSize": "12px", "fontWeight": "600"}),
                            html.Th("Region", style={"padding": "8px", "textAlign": "left", "borderBottom": f"1px solid {BORDER}", "color": TEXT_DIM, "fontSize": "12px", "fontWeight": "600"}),
                        ] + [
                            html.Th(col.replace("_", " ").title()[:15], style={"padding": "8px", "textAlign": "right", "borderBottom": f"1px solid {BORDER}", "color": TEXT_DIM, "fontSize": "11px", "fontWeight": "600"})
                            for col in use_cols  # Show all attributes
                        ])
                    )
                    
                    # Create table rows
                    table_rows = []
                    for idx, row in brushed_df.iterrows():
                        row_cells = [
                            html.Td(row["_country_fixed"], style={"padding": "8px", "borderBottom": f"1px solid {BORDER}", "color": TEXT, "fontSize": "12px"}),
                            html.Td(row.get("continent", "N/A"), style={"padding": "8px", "borderBottom": f"1px solid {BORDER}", "color": TEXT_DIM, "fontSize": "11px"}),
                        ] + [
                            html.Td(
                                f"{row[col]:,.1f}" if pd.notna(row[col]) else "—",
                                style={"padding": "8px", "textAlign": "right", "borderBottom": f"1px solid {BORDER}", "color": TEXT_BRIGHT if pd.notna(row[col]) else TEXT_DIM, "fontSize": "11px"}
                            )
                            for col in use_cols
                        ]
                        # Check if row is selected
                        is_selected = row["iso_alpha"] in (sel_isos or [])
                        row_style = {"background": "rgba(16, 185, 129, 0.2)"} if is_selected else {}
                        
                        table_rows.append(html.Tr(row_cells, style=row_style))
                    
                    table_body = html.Tbody(table_rows)
                    
                    table = html.Table(
                        [table_header, table_body],
                        style={
                            "minWidth": "100%",
                            "width": "max-content",
                            "borderCollapse": "collapse",
                            "fontSize": "11px",
                            "marginTop": "8px",
                            "whiteSpace": "nowrap"
                        }
                    )
                    
                    result.append(table)
                    
                    # Add summary info
                    result.append(html.Div(
                        f"Showing {len(brushed_df)} of {brushed_mask.sum()} brushed countries",
                        style={"marginTop": "12px", "fontSize": "11px", "color": TEXT_DIM, "fontStyle": "italic"}
                    ))
        except Exception as e:
            print(f"Error in update_countries_tab: {e}")
            import traceback
            traceback.print_exc()
    
    if not result:
        result.append(html.Div("No country selected. Click on the map or scatter plot to select a country, or brush axes to filter.", 
                              style={"color": TEXT_DIM, "fontStyle": "italic"}))
    
    return result

# NOTE: Reset behaviour is now handled in the main `analytics` callback
# (listens to `reset-parcoords` clicks and clears `pc-brush`).

# -----------------------------
# KPI CARDS
# -----------------------------
@app.callback(
    Output("kpi-cards", "children"),
    Input("region", "value"),
    Input("ind", "value")
)
def render_kpi_cards(region, ind):
    ds = FIRST_DS
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]
    
    num_countries = len(base)
    num_indicators = len(loaded[ds]["num_cols"])
    
    # Calculate KPIs
    avg_val = base[ind].mean() if ind and ind in base.columns else 0
    max_val = base[ind].max() if ind and ind in base.columns else 0
    min_val = base[ind].min() if ind and ind in base.columns else 0
    
    # Find top country (maximum)
    top_country = ""
    if ind and ind in base.columns:
        top_row = base.loc[base[ind].idxmax()] if base[ind].notna().any() else None
        if top_row is not None:
            top_country = top_row.get("_country_fixed", "")
    
    # Find bottom country (minimum)
    bottom_country = ""
    if ind and ind in base.columns:
        bottom_row = base.loc[base[ind].idxmin()] if base[ind].notna().any() else None
        if bottom_row is not None:
            bottom_country = bottom_row.get("_country_fixed", "")
    
    def kpi_card(title, value, subtitle="", color=ACCENT, accent_bg=None):
        bg_color = accent_bg or f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.08)"
        return html.Div(className="panel-hover", style={
            "background": "linear-gradient(180deg, #15171c 0%, #12141a 100%)",
            "border": f"1px solid {BORDER}",
            "borderRadius": "16px",
            "padding": "20px 24px",
            "position": "relative",
            "overflow": "hidden",
        }, children=[
            # Accent glow
            html.Div(style={
                "position": "absolute", "top": "-20px", "right": "-20px",
                "width": "80px", "height": "80px", "borderRadius": "50%",
                "background": bg_color, "filter": "blur(30px)", "opacity": "0.6"
            }),
            # Content
            html.Div(style={"display": "flex", "alignItems": "flex-start", "justifyContent": "space-between"}, children=[
                html.Div(children=[
                    html.Div(title, style={"color": TEXT_DIM, "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "8px", "fontWeight": 600}),
                    html.Div(f"{value:,.2f}" if isinstance(value, float) else str(value), 
                             style={"color": TEXT_BRIGHT, "fontSize": "28px", "fontWeight": 700, "letterSpacing": "-1px"}),
                    html.Div(subtitle, style={"color": TEXT_DIM, "fontSize": "11px", "marginTop": "4px"}) if subtitle else None
                ]),
            ])
        ])
    
    return [
        kpi_card("Countries", num_countries, f"{num_indicators} indicators", ACCENT),
        kpi_card("Average", avg_val, "Selected indicator", "#10b981"),
        kpi_card("Maximum", max_val, top_country[:20] if top_country else "", "#f59e0b"),
        kpi_card("Minimum", min_val, bottom_country[:20] if bottom_country else "", "#ef4444"),
    ]

# -----------------------------
# SPLOM (Scatter Plot Matrix)
# -----------------------------
@app.callback(
    Output("splom", "figure"),
    Input("region", "value"),
    Input("selected-iso-store", "data")
)
def render_splom(region, selected_iso_store):
    ds = FIRST_DS
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]
    
    num_cols = loaded[ds]["num_cols"]
    # Pick top 5 indicators for SPLOM (to keep it manageable)
    splom_cols = num_cols[:5] if len(num_cols) >= 5 else num_cols
    
    if len(splom_cols) < 2 or len(base) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for SPLOM", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=TEXT_DIM))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    
    # Prepare data
    plot_df = base[["_country_fixed", "iso_alpha", "continent"] + splom_cols].dropna()
    
    # Create SPLOM
    fig = px.scatter_matrix(
        plot_df,
        dimensions=splom_cols,
        color="continent",
        hover_name="_country_fixed",
        labels={c: c.replace("_", " ").title()[:15] for c in splom_cols}
    )
    
    fig.update_traces(
        diagonal_visible=False,
        marker=dict(size=5, opacity=0.7)
    )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# -----------------------------
# PCA VIEW (Region-respecting, with selected country highlight)
# -----------------------------
@app.callback(
    Output("pca", "figure"),
    Input("region", "value"), Input("selected-iso-store", "data"),
    Input("pca-params", "value")
)
def pca_view(region, selected_iso_store, pca_params):
    ds = FIRST_DS  # Use single dataset
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]
    
    # Use selected parameters if provided, otherwise use all numeric columns
    if pca_params and len(pca_params) > 0:
        num_cols = [c for c in pca_params if c in base.columns and base[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    else:
        num_cols = [c for c in base.select_dtypes(include="number").columns if c != "iso_alpha"]
    
    if len(num_cols) < 3 or len(base) < 3:
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric data for PCA",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    try:
        X = base[num_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.loc[:, X.notna().any()]
        variances = X.var()
        keep = variances[variances > 0].index.tolist()
        X = X[keep]
        if X.shape[1] < 2 or len(X) < 3:
            fig = go.Figure()
            fig.update_layout(title="Not enough variance for PCA",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig
        top_cols = variances.sort_values(ascending=False).index[:12]
        X = X[top_cols.intersection(X.columns)]
        X = X.fillna(X.median())
        X = (X - X.mean()) / (X.std(ddof=0) + 1e-9)

        use_sklearn = True
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
        except Exception:
            use_sklearn = False

        if use_sklearn:
            Z = PCA(n_components=2, random_state=42).fit_transform(X.values)
            k = min(5, max(2, int(np.sqrt(len(base))//1)))
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X.values)
            labels = km.labels_
        else:
            u, s, vh = np.linalg.svd(X.values, full_matrices=False)
            Z = (u[:, :2] * s[:2])
            k = min(5, max(2, int(np.sqrt(len(base))//1)))
            rng = np.random.default_rng(42)
            centroids = Z[rng.choice(len(Z), size=k, replace=False)]
            for _ in range(8):
                d2 = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
                labels = d2.argmin(axis=1)
                new_centroids = np.vstack([Z[labels == i].mean(axis=0) if (labels == i).any() else centroids[i] for i in range(k)])
                if np.allclose(new_centroids, centroids, atol=1e-6): break
                centroids = new_centroids

        df_plot = pd.DataFrame({
            "PC1": Z[:, 0],
            "PC2": Z[:, 1],
            "_country_fixed": base["_country_fixed"].values,
            "iso_alpha": base["iso_alpha"].values,
            "cluster": labels.astype(str),
        })

        # prepare cluster palette
        palette = px.colors.qualitative.Set1
        clusters = sorted(df_plot["cluster"].unique())

        # selected ISOs set (multi-select)
        sel_iso = set(selected_iso_store or [])

        fig = go.Figure()
        for i, cl in enumerate(clusters):
            cdf = df_plot[df_plot["cluster"] == cl]
            base_color = palette[i % len(palette)]
            # per-point fill colors (cluster color) and border for selected points
            fill_colors = [base_color] * len(cdf)
            line_colors = [HILITE_YELLOW if iso in sel_iso else "rgba(0,0,0,0)" for iso in cdf["iso_alpha"]]
            line_widths = [2 if iso in sel_iso else 0 for iso in cdf["iso_alpha"]]
            sizes = [11 if iso in sel_iso else 9 for iso in cdf["iso_alpha"]]

            fig.add_trace(go.Scatter(
                x=cdf["PC1"], y=cdf["PC2"], mode="markers",
                name=f"Cluster {cl}",
                marker=dict(size=sizes, color=fill_colors,
                            line=dict(color=line_colors, width=line_widths), opacity=0.6),
                text=cdf["_country_fixed"], hoverinfo="text+x+y",
                customdata=cdf["iso_alpha"]  # Add ISO for reliable click handling
            ))
        
        fig.update_layout(
            margin=dict(l=10,r=10,t=10,b=10), 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Principal Component 1 (PC1)",
            yaxis_title="Principal Component 2 (PC2)",
            font=dict(color=TEXT, size=12)
        )
        fig.update_xaxes(title_font=dict(color=TEXT), tickfont=dict(color=TEXT))
        fig.update_yaxes(title_font=dict(color=TEXT), tickfont=dict(color=TEXT))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"PCA unavailable: {e}",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

# -----------------------------
# FULLSCREEN CONTROLLER
# -----------------------------
@app.callback(
    Output("fullscreen-modal", "style"),
    Output("fullscreen-graph", "figure"),
    Input("open-fullscreen-map", "n_clicks"),
    Input("open-fullscreen-scatter", "n_clicks"),
    Input("open-fullscreen-rank", "n_clicks"),
    Input("open-fullscreen-corr", "n_clicks"),
    Input("open-fullscreen-splom", "n_clicks"),
    Input("open-fullscreen-pca", "n_clicks"),
    # Input("open-fullscreen-bubble", "n_clicks"),  # COMMENTED OUT
    Input("close-fullscreen", "n_clicks"),
    State("map", "figure"),
    State("scatter", "figure"),
    State("rank", "figure"),
    State("corr", "figure"),
    State("splom", "figure"),
    State("parcoords", "figure"),
    State("pca", "figure"),
    # State("bubble-chart", "figure"),  # COMMENTED OUT
    prevent_initial_call=True
)
def open_fullscreen(btn_map, btn_scatter, btn_rank, btn_corr, btn_splom, btn_pca, btn_close,
                    fig_map, fig_scatter, fig_rank, fig_corr, fig_splom, fig_par, fig_pca):
    trigger = ctx.triggered_id
    if trigger == "close-fullscreen":
        return {"display": "none"}, go.Figure()
    mapping = {
        "open-fullscreen-map": fig_map,
        "open-fullscreen-scatter": fig_scatter,
        "open-fullscreen-rank": fig_rank,
        "open-fullscreen-corr": fig_corr,
        "open-fullscreen-splom": fig_splom,
        "open-fullscreen-pca": fig_pca,
        # "open-fullscreen-bubble": fig_bubble,  # COMMENTED OUT
    }
    if trigger in mapping:
        return (
            {
                "display": "flex",
                "position": "fixed",
                "top": "50%",
                "left": "50%",
                "transform": "translate(-50%, -50%)",
                "width": "85vw",
                "height": "85vh",
                "maxWidth": "85vw",
                "maxHeight": "85vh",
                "background": "rgba(0,0,0,0.92)",
                "zIndex": 9999,
                "padding": "20px",
                "backdropFilter": "blur(6px)",
                "borderRadius": "10px"
            },
            mapping[trigger]
        )
    return no_update, no_update

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False)
