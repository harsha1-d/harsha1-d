# =============================================================================
# CIA World Stats
# =============================================================================
# What’s included:
# - Top nav + dark control bar
# - Global region filter (propagates to Map, Analytics, PCA)
# - Map + Scatter click -> highlight that country in Parcoords
# - Reset Selection button clears click + parcoords brushes
# - Parcoords: white labels/ticks, axis titles above each axis, fits container
# - Brushed range -> list of country names below parcoords
# - Fullscreen modal for Map / Scatter / Rank / Corr / Parcoords / PCA
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# -----------------------------
# THEME CONSTANTS
# -----------------------------
BG          = "#0b0c0f"
PANEL       = "#16171b"
BORDER      = "rgba(255,255,255,0.08)"
TEXT        = "#f1f1f1"
TEXT_DIM    = "#9a9a9a"
ACCENT      = "#2d7df4"
ACCENT_SOFT = "rgba(45,125,244,0.95)"
HILITE      = "#ffd166"  # gold for highlights
FADE_LINE   = "rgba(255,255,255,0.18)"
FONT_FAMILY = "'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"

pio.templates.default = "plotly_dark"

# -----------------------------
# DATASET CONFIG
# -----------------------------
DATASETS = {
    "Energy": "Datasets/energy_data.csv",
    "Communications": "Datasets/communications_data.csv",
    "Demographics": "Datasets/demographics_data.csv",
    "Economy": "Datasets/economy_data.csv",
    "Geography": "Datasets/geography_data.csv",
    "Transportation": "Datasets/transportation_data.csv",
    "Government & Civics": "Datasets/government_and_civics_data.csv",
}
COUNTRY_COL = "Country"
BASE_DIR = Path(__file__).resolve().parent

# Base ISO & continent from gapminder
GAP = px.data.gapminder()[["country", "iso_alpha", "continent"]].drop_duplicates()

# CIA → gapminder name fixes
CIA_NAME_FIXES = {
    "BOLIVIA": "Bolivia",
    "BOSNIA AND HERZEGOVINA": "Bosnia and Herzegovina",
    "BRUNEI": "Brunei Darussalam",
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
    "LAOS": "Lao PDR",
    "MACEDONIA": "North Macedonia",
    "MICRONESIA, FEDERATED STATES OF": "Micronesia, Fed. Sts.",
    "RUSSIA": "Russian Federation",
    "SYRIA": "Syrian Arab Republic",
    "TANZANIA": "Tanzania",
    "UNITED STATES": "United States",
    "VENEZUELA": "Venezuela, RB",
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
                return np.nan
        missing_iso = out["iso_alpha"].isna()
        if missing_iso.any():
            out.loc[missing_iso, "iso_alpha"] = out.loc[missing_iso, "_country_fixed"].apply(lookup_iso)
    except Exception:
        pass
    # pycountry_convert fallback for continent
    try:
        import pycountry_convert as pcc
        miss_cont = out["continent"].isna() & out["iso_alpha"].notna()
        if miss_cont.any():
            def iso3_to_cont(i3):
                try:
                    a2 = pcc.country_alpha3_to_country_alpha2(i3)
                    code = pcc.country_alpha2_to_continent_code(a2)
                    return dict(AF="Africa", AS="Asia", EU="Europe", NA="North America",
                                SA="South America", OC="Oceania", AN="Antarctica").get(code)
                except Exception:
                    return np.nan
            out.loc[miss_cont, "continent"] = out.loc[miss_cont, "iso_alpha"].apply(iso3_to_cont)
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
        for c in df.columns:
            if c not in skip:
                col = clean_numeric_column(df[c])
                if col.notna().any():
                    df[c] = col
        df = df.dropna(subset=["iso_alpha"]).drop_duplicates(subset=["iso_alpha"])
        num_cols = [c for c in df.select_dtypes(include="number").columns if c not in skip]
        num_cols.sort()
        loaded[name] = {"df": df, "num_cols": num_cols}
        print(f"Loaded {name}: {len(df)} rows, {len(num_cols)} numeric indicators")
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
]

# -----------------------------
# APP + GLOBAL DARK CSS
# -----------------------------
external_css = ['https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap']
app = Dash(__name__, external_stylesheets=external_css)
app.title = "CIA World Stats — Enterprise"

app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>CIA World Stats — Enterprise</title>
    {%css%}
    <style>
    body { margin: 0; padding: 0; }
    /* Dropdown dark theming */
    .Select-control {
        background-color: #16171b !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: #f2f2f2 !important;
        border-radius: 8px !important;
        min-height: 34px !important;
    }
    .Select--single > .Select-control .Select-value, .Select-placeholder { color: #cfcfcf !important; }
    .Select input { background-color: #16171b !important; color: #f2f2f2 !important; }
    .Select-menu-outer { background-color: #16171b !important; border: 1px solid rgba(255,255,255,0.18) !important; color: #f1f1f1 !important; border-radius: 8px !important; }
    .Select-menu { background-color: #16171b !important; }
    .Select-option { background-color: #16171b !important; color: #e5e5e5 !important; }
    .Select-option:hover { background-color: #2a313a !important; color: #ffffff !important; }
    .Select-option.is-selected { background-color: #2d7df4 !important; color: white !important; }
    .Select-value { background-color: #16171b !important; border: 1px solid rgba(255,255,255,0.18) !important; }
    .Select-value-label { color: #e5e5e5 !important; }
    .Select-value-icon { border-right-color: rgba(255,255,255,0.15) !important; }
    .Select-arrow { border-color: #f1f1f1 transparent transparent !important; }

    /* Panel animations */
    .panel-hover:hover {
        transform: translateY(-4px);
        transition: 0.25s ease;
        box-shadow: 0 10px 28px rgba(0,0,0,0.35);
    }
    .fade-in { animation: fadeIn 0.6s ease forwards; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
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
    "height": "62px", "display": "flex", "alignItems": "center",
    "justifyContent": "space-between", "padding": "0 20px",
    "background": PANEL, "borderBottom": f"1px solid {BORDER}",
    "position": "sticky", "top": 0, "zIndex": 50
}
DD_STYLE = {
    "width": "100%",
    "color": "#f1f1f1",
    "backgroundColor": "#1b1d22",
    "border": "1px solid rgba(255,255,255,0.18)",
    "borderRadius": "8px",
    "padding": "4px",
    "fontFamily": FONT_FAMILY,
}
BTN_ICON = {
    "position": "absolute",
    "top": "10px",
    "right": "10px",
    "zIndex": 5,
    "background": "rgba(255,255,255,0.1)",
    "border": f"1px solid {BORDER}",
    "borderRadius": "6px",
    "cursor": "pointer",
    "color": "white",
    "fontSize": "16px",
    "padding": "4px 8px",
}

# -----------------------------
# LAYOUT
# -----------------------------
app.layout = html.Div(style={"background": BG, "color": TEXT, "minHeight": "100vh", "fontFamily": FONT_FAMILY}, children=[
    dcc.Store(id="selected-iso", data=None),        # << global clicked ISO-3 from Map/Scatter
    dcc.Store(id="pc-brush", data={}),              # << stores parcoords brush constraints
    dcc.Store(id="selected-iso-store", data=[]),       # list of iso_alpha for highlight/selection
    dcc.Store(id="highlight-country-names", data=[]),  # list of _country_fixed names

    # Top Navigation
    html.Div(style=NAV_STYLE, children=[
        html.Div(children=[ html.Span("CIA World Stats", style={"fontWeight": 800, "fontSize": "20px"}) ])
    ]),

    # Control Bar
    html.Div(style={
        "display": "grid", "gridTemplateColumns": "240px 1fr 1fr 140px",
        "gap": "12px", "padding": "16px 20px", "borderBottom": f"1px solid {BORDER}",
        "background": "#0f1014"
    }, children=[
        dcc.Dropdown(id="ds", options=[{"label": k, "value": k} for k in loaded.keys()], value=list(loaded.keys())[0], clearable=False, style=DD_STYLE),
        dcc.Dropdown(id="ind", placeholder="Indicator", clearable=False, style=DD_STYLE),
        dcc.Dropdown(id="region", options=CONTINENT_OPTIONS, value="All", clearable=False, style=DD_STYLE),
        dcc.Checklist(id="log-scale", options=[{"label": " log10", "value": "log"}],
                      value=[], style={"display": "flex", "alignItems": "center"})
    ]),

    # Tabs
    dcc.Tabs(id="tabs", value="tab-map", parent_style={"background": BG}, children=[
        dcc.Tab(label="Map", value="tab-map", selected_style={"background": PANEL}, style={"background": PANEL},
                children=[
                    html.Div(style={"padding": "10px 20px"}, children=[
                        html.Div(style={"position": "relative"}, children=[
                            html.Button("⤢", id="open-fullscreen-map", style=BTN_ICON),
                            dcc.Graph(id="map", style={"height": "76vh"}, config={"displayModeBar": False})
                        ])
                    ]),
                    html.Div(id="note", style={"color": TEXT_DIM, "fontSize": "12px", "padding": "0 20px 16px"})
                ]),

        dcc.Tab(label="Analytics", value="tab-analytics", selected_style={"background": PANEL}, style={"background": PANEL},
                children=[
                    html.Div(style={"padding": "24px", "height": "82vh", "overflowY": "auto", "scrollBehavior": "smooth"}, children=[

                        # Scatter Panel
                        html.Div(className="panel-hover fade-in", style={
                            "background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "14px",
                            "padding": "18px", "marginBottom": "28px", "boxShadow": "0 6px 20px rgba(0,0,0,0.25)",
                            "transition": "0.25s ease",
                        }, children=[
                            html.H3("Scatter Analysis", style={"margin": "0 0 12px 0", "fontSize": "18px", "opacity": 0.85}),
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
                            html.H3("Top-N Ranking", style={"margin": "0 0 12px 0", "fontSize": "18px", "opacity": 0.85}),
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
                            html.H3("Correlation Matrix", style={"margin": "0 0 12px 0", "fontSize": "18px", "opacity": 0.85}),
                            html.Div(style={"position": "relative"}, children=[
                                html.Button("⤢", id="open-fullscreen-corr", style=BTN_ICON),
                                dcc.Graph(id="corr", style={"height": "50vh"}, config={"displayModeBar": False}),
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
                                    html.Button("Reset selection", id="reset-parcoords", n_clicks=0,
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

                            # Country session list (replaced Tabs to avoid rendering tab header classes)
                            html.Div(id="pc-countries-list", style={"padding": "12px", "color": TEXT_DIM, "fontSize": "13px", "minHeight": "100px", "marginTop": "12px"}),
                        ]),

                            ])
                        ]),
  

        dcc.Tab(label="PCA / Clusters", value="tab-pca", selected_style={"background": PANEL}, style={"background": PANEL},
                children=[
                    html.Div(style={"padding": "16px 20px"}, children=[
                        html.Div(style={"position": "relative"}, children=[
                            html.Button("⤢", id="open-fullscreen-pca", style=BTN_ICON),
                            dcc.Graph(id="pca", style={"height": "75vh"}, config={"displayModeBar": False})
                        ])
                    ])
                ]),

        dcc.Tab(label="Compare", value="tab-compare", selected_style={"background": PANEL}, style={"background": PANEL},
                children=[
                    html.Div(style={"padding": "16px 20px", "display": "flex", "flexDirection": "column", "gap": "16px", "height": "90vh", "overflowY": "auto"}, children=[
                        # Country selection section
                        html.Div(style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "12px", "padding": "12px", "flexShrink": 0}, children=[
                            html.Div("Select countries", style={"color": TEXT_DIM, "marginBottom": "6px", "fontWeight": "600"}),
                            dcc.Dropdown(id="compare-countries", options=COUNTRY_OPTIONS, multi=True, style=DD_STYLE),
                        ]),
                        
                        # Comparison table section (scrollable)
                        html.Div(id="compare-table", style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "12px", "padding": "12px", "overflowX": "auto", "flex": "3 1 auto", "minHeight": "300px"}),
                        
                        # Bubble Chart section
                        html.Div(style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": "12px", "padding": "18px", "boxShadow": "0 6px 20px rgba(0,0,0,0.25)", "flex": "2 1 400px"}, children=[
                            html.H3("Bubble Chart Analysis (3-Variable Scatter)", style={"margin": "0 0 12px 0", "fontSize": "18px", "opacity": 0.85}),
                            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px", "marginBottom": "12px"}, children=[
                                html.Div([
                                    html.Div("X-Axis", style={"color": TEXT_DIM, "fontSize": "12px", "marginBottom": "4px"}),
                                    dcc.Dropdown(id="bubble-x", clearable=False, style=DD_STYLE),
                                ]),
                                html.Div([
                                    html.Div("Y-Axis", style={"color": TEXT_DIM, "fontSize": "12px", "marginBottom": "4px"}),
                                    dcc.Dropdown(id="bubble-y", clearable=False, style=DD_STYLE),
                                ]),
                                html.Div([
                                    html.Div("Bubble Size", style={"color": TEXT_DIM, "fontSize": "12px", "marginBottom": "4px"}),
                                    dcc.Dropdown(id="bubble-size", clearable=False, style=DD_STYLE),
                                ]),
                            ]),
                            html.Div(style={"position": "relative", "background": "transparent", "height": "100%", "minHeight": "320px"}, children=[
                                html.Button("⤢", id="open-fullscreen-bubble", style=BTN_ICON),
                                html.Div(id="bubble-header", style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"8px"}, children=[
                                    html.Div(id="bubble-count", style={"color": TEXT_DIM}),
                                ]),
                                dcc.Graph(id="bubble-chart", style={"height": "480px", "backgroundColor": "transparent", "width": "100%"}, config={"displayModeBar": False, "responsive": True}),
                            ]),
                        ]),
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
    Input("ds", "value"), State("ind", "value"), State("scatter-x", "value"), State("scatter-y", "value")
)
def set_indicators(ds, ind_cur, x_cur, y_cur):
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
    State("highlight-country-names", "data"),
    State("selected-iso-store", "data"),
    prevent_initial_call=True
)
def sync_highlights(map_click, prev_names, prev_iso):
    names = set(prev_names or [])
    isos = set(prev_iso or [])

    if map_click and "points" in map_click and map_click["points"]:
        iso = map_click["points"][0].get("location")
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
    Input("ds", "value"), Input("ind", "value"), Input("region", "value"), Input("log-scale", "value"),
    Input("selected-iso-store", "data"),  # highlight overlay
    State("map", "figure")
)
def render_map(ds, ind, region, logv, selected_iso, cur_fig):
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
        hover_data={ind: True, "val": False, "iso_alpha": False},
        color_continuous_scale="Viridis", labels={"val": lab}, locationmode="ISO-3", title=""
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
            marker_line_width=2.5, marker_line_color=HILITE
        )
        fig.add_trace(overlay)

    unmatched = loaded[ds]["df"][loaded[ds]["df"]["iso_alpha"].isna()]["_country_fixed"].unique()
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
    State("ds", "value"),
    prevent_initial_call=True
)
def update_selected_iso(map_click, scatter_click, ds):
    trigger = ctx.triggered_id
    df = loaded[ds]["df"]
    if trigger == "map" and map_click:
        iso = map_click["points"][0].get("location")
        return iso
    if trigger == "scatter" and scatter_click:
        name = scatter_click["points"][0].get("hovertext")
        if name:
            row = df[df["_country_fixed"] == name]
            if not row.empty:
                return row.iloc[0]["iso_alpha"]
    return no_update

# -----------------------------
# ANALYTICS (Rank, Scatter, Corr, Parcoords)
# -----------------------------
@app.callback(
    Output("rank", "figure"), Output("scatter", "figure"),
    Output("corr", "figure"),
    Output("parcoords", "figure"), Output("parcoords", "style"),
    Output("pc-brush", "data"),  # keep last brush ranges (also resettable)
    Input("ds", "value"), Input("ind", "value"), Input("region", "value"),
    Input("topn", "value"), Input("scatter-x", "value"), Input("scatter-y", "value"),
    Input("parcoords", "relayoutData"),
    Input("reset-parcoords", "n_clicks"),
    Input("selected-iso", "data"), Input("selected-iso-store", "data"),
    State("pc-brush", "data")
)
def analytics(ds, ind, region, topn, sx, sy, pc_relayout, reset_clicks, selected_iso, selected_iso_store, pc_brush_state):
    # Safety check: if no indicator selected, return empty figures
    if not ind:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return empty_fig, empty_fig, empty_fig, empty_fig, {"height": "52vh", "width": "100%"}, {}
    
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]

    # Map selected ISO → name set (using multi-select store)
    sel_iso = set(selected_iso_store or [])
    sel_names = {ISO_TO_NAME[i] for i in sel_iso if i in ISO_TO_NAME}

    # ----- Rank -----
    topn = max(3, min(50, int(topn or 10)))
    bar_df = base[["_country_fixed", "iso_alpha", ind]].dropna().sort_values(ind, ascending=False).head(topn).copy()
    bar_df["is_sel"] = bar_df["_country_fixed"].isin(sel_names)
    colors = np.where(bar_df["is_sel"], HILITE, ACCENT)
    rfig = px.bar(bar_df, x=ind, y="_country_fixed", orientation="h", color_discrete_sequence=[ACCENT])
    rfig.update_traces(marker_color=colors, hovertemplate="%{y}: %{x:,.2f}<extra></extra>")
    rfig.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis=dict(autorange="reversed"),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # ----- Scatter -----
    sfig = go.Figure()
    if sx and sy and sx in base.columns and sy in base.columns:
        df_all = base[["_country_fixed", "iso_alpha", sx, sy]].dropna().copy()
        df_all["is_sel"] = df_all["_country_fixed"].isin(sel_names)

        # base points
        sfig.add_trace(go.Scatter(
            x=df_all.loc[~df_all["is_sel"], sx],
            y=df_all.loc[~df_all["is_sel"], sy],
            mode="markers",
            name="Others",
            marker=dict(size=9, opacity=0.85, color="#4ade80"),
            hovertext=df_all.loc[~df_all["is_sel"], "_country_fixed"],
            hoverinfo="text+x+y"
        ))
        # highlighted points with labels
        sfig.add_trace(go.Scatter(
            x=df_all.loc[df_all["is_sel"], sx],
            y=df_all.loc[df_all["is_sel"], sy],
            mode="markers+text",
            name="Highlighted",
            text=df_all.loc[df_all["is_sel"], "_country_fixed"],
            textposition="top center",
            marker=dict(size=13, opacity=0.98, color=HILITE, line=dict(width=1.2, color="#222")),
            hovertext=df_all.loc[df_all["is_sel"], "_country_fixed"],
            hoverinfo="text+x+y"
        ))
        sfig.update_layout(
            margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        sfig.update_xaxes(title_text=sx.replace("_"," ").title())
        sfig.update_yaxes(title_text=sy.replace("_"," ").title())

    # ----- Correlation -----
    cfig = go.Figure()
    num_cols = [c for c in base.select_dtypes(include="number").columns if c != "iso_alpha"]
    if len(num_cols) >= 2:
        corr = base[num_cols].corr().round(2)
        cfig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", origin="lower")
        cfig.update_layout(margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # ----- Parallel Coordinates -----
    use_cols = [c for c in num_cols if c != "iso_alpha"][:12]  # can increase later
    pfig = go.Figure()
    brush_names = []
    new_brush_state = pc_brush_state or {}
    # default parcoords graph style (will be updated based on number of axes)
    pstyle = {"height": "52vh", "width": "100%"}

    if len(use_cols) >= 2:
        sub = base[["_country_fixed", "iso_alpha"] + use_cols].dropna()

        if not sub.empty:
            # Selected line highlight (support multi-selection via selected-iso-store)
            sel_set = set(selected_iso_store or [])
            color_mask = sub["iso_alpha"].isin(sel_set).astype(int) if sel_set else np.zeros(len(sub), int)

            # Build dimensions (VALID properties only!)
            dims = [
                dict(
                    label=c.replace("_", " ").title(),
                    values=sub[c]
                )
                for c in use_cols
            ]

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
            )

            # ---- Parse Brush Ranges ----
            if pc_relayout and isinstance(pc_relayout, dict):
                for k, v in pc_relayout.items():
                    if "dimensions[" in k and "constraintrange" in k:
                        try:
                            idx = int(k.split("dimensions[")[1].split("]")[0])
                            if 0 <= idx < len(use_cols):
                                new_brush_state[use_cols[idx]] = v
                        except Exception:
                            pass

            # If the reset button was pressed, clear stored brush ranges
            if ctx.triggered_id == "reset-parcoords":
                new_brush_state = {}

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
    return rfig, sfig, cfig, pfig, pstyle, new_brush_state

# -----------------------------
# POPULATE PC-COUNTRIES-LIST TAB
# -----------------------------
@app.callback(
    Output("pc-countries-list", "children"),
    Input("selected-iso", "data"),
    Input("selected-iso-store", "data"),
    Input("pc-brush", "data"),
    State("ds", "value"),
    prevent_initial_call=False
)
def update_countries_tab(selected_iso, selected_iso_store, pc_brush_state, ds):
    """Display selected and brushed countries in a nicely formatted list."""
    if not ds or ds not in loaded:
        return html.Div("No dataset selected.", style={"color": TEXT_DIM})
    
    base = loaded[ds]["df"].copy()
    result = []
    
    # Display selected country
    # Show multi-selected countries (from selected-iso-store) when present
    sel_isos = list(selected_iso_store or [])
    if sel_isos:
        sel_rows = base[base["iso_alpha"].isin(sel_isos)]["_country_fixed"].dropna().unique().tolist()
        if sel_rows:
            result.append(html.Div([
                html.Span("★ Selected Countries: ", style={"color": ACCENT, "fontWeight": "600"}),
                html.Span(", ".join(sel_rows), style={"fontWeight": "500"})
            ], style={"marginBottom": "12px", "fontSize": "14px"}))
    else:
        # fallback to single selected_iso for older interactions
        if selected_iso:
            selected_country = base[base["iso_alpha"] == selected_iso]["_country_fixed"].values
            if len(selected_country) > 0:
                result.append(html.Div([
                    html.Span("★ Selected Country: ", style={"color": ACCENT, "fontWeight": "600"}),
                    html.Span(selected_country[0], style={"fontWeight": "500"})
                ], style={"marginBottom": "12px", "fontSize": "14px"}))
    
    # Display brushed countries
    if pc_brush_state:
        num_cols = [c for c in base.select_dtypes(include="number").columns if c != "iso_alpha"]
        use_cols = [c for c in num_cols if c != "iso_alpha"][:12]
        sub = base[["_country_fixed", "iso_alpha"] + use_cols].dropna()
        
        if not sub.empty:
            brushed_mask = np.ones(len(sub), dtype=bool)
            for col, rng in pc_brush_state.items():
                if isinstance(rng, list) and len(rng) == 2 and col in sub.columns:
                    lo, hi = float(rng[0]), float(rng[1])
                    brushed_mask &= (sub[col] >= lo) & (sub[col] <= hi)
            
            if brushed_mask.any():
                brushed_countries = sub.loc[brushed_mask, "_country_fixed"].tolist()[:40]
                result.append(html.Div([
                    html.Div("Brushed Countries (from axis constraints):", 
                            style={"fontWeight": "600", "marginBottom": "8px", "color": ACCENT_SOFT})
                ]))
                for country in brushed_countries:
                    result.append(html.Div(f"  • {country}", style={"paddingLeft": "12px", "marginBottom": "4px"}))
    
    if not result:
        result.append(html.Div("No country selected. Click on the map or scatter plot to select a country, or brush axes to filter.", 
                              style={"color": TEXT_DIM, "fontStyle": "italic"}))
    
    return result

# NOTE: Reset behaviour is now handled in the main `analytics` callback
# (listens to `reset-parcoords` clicks and clears `pc-brush`).

# -----------------------------
# PCA VIEW (Region-respecting, with selected country highlight)
# -----------------------------
@app.callback(
    Output("pca", "figure"),
    Input("ds", "value"), Input("region", "value"), Input("selected-iso-store", "data")
)
def pca_view(ds, region, selected_iso_store):
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]
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
            line_colors = [HILITE if iso in sel_iso else "rgba(0,0,0,0)" for iso in cdf["iso_alpha"]]
            line_widths = [2 if iso in sel_iso else 0 for iso in cdf["iso_alpha"]]
            sizes = [11 if iso in sel_iso else 9 for iso in cdf["iso_alpha"]]

            fig.add_trace(go.Scatter(
                x=cdf["PC1"], y=cdf["PC2"], mode="markers",
                name=f"Cluster {cl}",
                marker=dict(size=sizes, color=fill_colors,
                            line=dict(color=line_colors, width=line_widths), opacity=0.9),
                text=cdf["_country_fixed"], hoverinfo="text+x+y"
            ))
        
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"PCA unavailable: {e}",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

# -----------------------------
# COMPARE TABLE
# -----------------------------
@app.callback(
    Output("compare-countries", "options"),
    Input("region", "value"),
    Input("ds", "value")
)
def update_compare_options(region, ds):
    """Update the compare countries dropdown to show only countries from selected region."""
    base = loaded[ds]["df"].copy()
    if region and region != "All":
        base = base[base["continent"] == region]
    
    countries = sorted(base["_country_fixed"].dropna().unique())
    return [{"label": c, "value": c} for c in countries]

@app.callback(
    Output("compare-countries", "value"),
    Input("region", "value"),
    prevent_initial_call=True
)
def clear_compare_on_region_change(region):
    """Clear compare selections when region filter changes."""
    return []

@app.callback(
    Output("compare-table", "children"),
    Input("compare-countries", "value"),
    Input("ds", "value")
)
def compare_countries(countries, ds):
    if not countries or len(countries) < 2:
        return html.Div("Select at least two countries.", style={"color": TEXT_DIM})
    blocks = []
    for ds_name, data in loaded.items():
        df = data["df"]; nums = data["num_cols"]
        if not nums: continue
        subset = df[df["_country_fixed"].isin(countries)].set_index("_country_fixed")
        if subset.empty: continue
        header = [html.Th("Indicator", style={"textAlign":"left", "padding":"8px"})] + [html.Th(c, style={"textAlign":"right", "padding":"8px"}) for c in countries]
        rows = []
        for col in nums[:12]:
            pretty = col.replace("_"," ").title()
            cells = [html.Td(pretty, style={"padding":"8px", "borderBottom": f"1px solid {BORDER}"})]
            for ctry in countries:
                val = subset.loc[ctry, col] if ctry in subset.index else np.nan
                cells.append(html.Td(f"{val:,.2f}" if pd.notna(val) else "-", style={"textAlign":"right","padding":"8px","borderBottom": f"1px solid {BORDER}"}))
            rows.append(html.Tr(cells))
        table = html.Div([
            html.Div(ds_name, style={"fontWeight":700, "color": TEXT, "margin":"8px 0"}),
            html.Table([html.Thead(html.Tr(header)), html.Tbody(rows)], style={"width":"100%","borderCollapse":"collapse","fontSize":"14px"})
        ], style={"marginBottom":"14px", "background": PANEL, "padding":"8px", "borderRadius":"8px", "border": f"1px solid {BORDER}"})
        blocks.append(table)
    return blocks or html.Div("No comparable indicators found.", style={"color": TEXT_DIM})

# -----------------------------
# BUBBLE CHART CALLBACKS
# -----------------------------
@app.callback(
    Output("bubble-x", "options"), Output("bubble-x", "value"),
    Output("bubble-y", "options"), Output("bubble-y", "value"),
    Output("bubble-size", "options"), Output("bubble-size", "value"),
    Input("ds", "value"),
    State("bubble-x", "value"), State("bubble-y", "value"), State("bubble-size", "value")
)
def update_bubble_indicators(ds, x_cur, y_cur, z_cur):
    """Populate bubble chart indicator dropdowns with all indicators from all datasets."""
    # Collect all unique numeric indicators from all datasets
    all_cols = set()
    for data in loaded.values():
        all_cols.update(data["num_cols"])
    all_cols = sorted(list(all_cols))
    
    opts = [{"label": c.replace("_", " ").title(), "value": c} for c in all_cols]
    
    # Ensure selected values are in the available columns, otherwise pick defaults
    x = x_cur if x_cur in all_cols else (all_cols[0] if all_cols else None)
    y = y_cur if (y_cur in all_cols and y_cur != x) else (all_cols[1] if len(all_cols) > 1 else (all_cols[0] if all_cols else None))
    z = z_cur if (z_cur in all_cols and z_cur not in [x, y]) else (all_cols[2] if len(all_cols) > 2 else (all_cols[0] if all_cols else None))
    
    return opts, x, opts, y, opts, z

@app.callback(
    Output("bubble-chart", "figure"),
    Input("compare-countries", "value"),
    Input("ds", "value"),
    Input("bubble-x", "value"),
    Input("bubble-y", "value"),
    Input("bubble-size", "value"),
)
def render_bubble_chart(countries, ds, bx, by, bz):
    """Render a 3-variable bubble chart comparing selected countries."""
    if not countries or len(countries) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Select at least 2 countries to visualize", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=TEXT_DIM))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    
    if not bx or not by or not bz:
        fig = go.Figure()
        fig.add_annotation(text="Select X, Y, and Size indicators", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=TEXT_DIM))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    
    # For cross-dataset indicator support: lookup each indicator per country across all datasets
    rows = []
    for ctry in countries:
        xval = None; yval = None; zval = None
        x_src = None; y_src = None; z_src = None
        # search for bx value across datasets
        for ds_name, data in loaded.items():
            df = data["df"]
            if bx in df.columns:
                r = df[df["_country_fixed"] == ctry]
                if not r.empty:
                    v = r.iloc[0].get(bx)
                    if pd.notna(v):
                        xval = float(v)
                        x_src = ds_name
                        break
        # search for by
        for ds_name, data in loaded.items():
            df = data["df"]
            if by in df.columns:
                r = df[df["_country_fixed"] == ctry]
                if not r.empty:
                    v = r.iloc[0].get(by)
                    if pd.notna(v):
                        yval = float(v)
                        y_src = ds_name
                        break
        # search for bz (size)
        for ds_name, data in loaded.items():
            df = data["df"]
            if bz in df.columns:
                r = df[df["_country_fixed"] == ctry]
                if not r.empty:
                    v = r.iloc[0].get(bz)
                    if pd.notna(v):
                        zval = float(v)
                        z_src = ds_name
                        break

        # require at least x and y to plot
        if xval is None or yval is None:
            continue

        # if size missing, set a small default so point is visible
        if zval is None:
            zval = 1.0
            z_src = "(missing)"

        rows.append({"_country_fixed": ctry, "x": xval, "y": yval, "size": zval,
                     "x_src": x_src or "(missing)", "y_src": y_src or "(missing)", "z_src": z_src or "(missing)"})

    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No data available for selected indicators across datasets", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=TEXT_DIM))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    df_bubble = pd.DataFrame(rows)
    
    # Create bubble chart with color by country and size by third variable
    fig = px.scatter(
        df_bubble,
        x="x",
        y="y",
        size="size",
        color="_country_fixed",
        hover_name="_country_fixed",
        hover_data={"x": ":.2f", "y": ":.2f", "size": ":.2f", "_country_fixed": False},
        title="",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # include source info as customdata for hover
    fig.update_traces(
        marker=dict(opacity=0.75, line=dict(width=1, color=BORDER)),
        customdata=df_bubble[["x_src", "y_src", "z_src"]].values,
        hovertemplate=("<b>%{hovertext}</b><br>"
                       f"{bx.replace('_', ' ').title()}: %{{x:.2f}} (src: %{{customdata[0]}})<br>"
                       f"{by.replace('_', ' ').title()}: %{{y:.2f}} (src: %{{customdata[1]}})<br>"
                       f"{bz.replace('_', ' ').title()}: %{{marker.size:.2f}} (src: %{{customdata[2]}})<extra></extra>")
    )
    
    # set paper_bgcolor to PANEL so the chart always matches the panel background
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PANEL,
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(color=TEXT, size=12)
    )
    
    fig.update_xaxes(title_text=bx.replace("_", " ").title(), gridcolor=FADE_LINE)
    fig.update_yaxes(title_text=by.replace("_", " ").title(), gridcolor=FADE_LINE)
    
    return fig


# show number of plotted countries in bubble chart
@app.callback(
    Output("bubble-count", "children"),
    Input("compare-countries", "value"),
    Input("bubble-x", "value"),
    Input("bubble-y", "value"),
    Input("bubble-size", "value"),
)
def update_bubble_count(countries, bx, by, bz):
    if not countries:
        return ""
    plotted = 0
    for ctry in countries:
        x_found = any((bx in d["df"].columns) and (not d["df"][d["df"]["_country_fixed"]==ctry][bx].dropna().empty) for d in loaded.values()) if bx else False
        y_found = any((by in d["df"].columns) and (not d["df"][d["df"]["_country_fixed"]==ctry][by].dropna().empty) for d in loaded.values()) if by else False
        if x_found and y_found:
            plotted += 1
    return f"Plotted countries: {plotted} / Selected: {len(countries)}"

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
    Input("open-fullscreen-pca", "n_clicks"),
    Input("open-fullscreen-bubble", "n_clicks"),
    Input("close-fullscreen", "n_clicks"),
    State("map", "figure"),
    State("scatter", "figure"),
    State("rank", "figure"),
    State("corr", "figure"),
    State("parcoords", "figure"),
    State("pca", "figure"),
    State("bubble-chart", "figure"),
    prevent_initial_call=True
)
def open_fullscreen(btn_map, btn_scatter, btn_rank, btn_corr, btn_pca, btn_bubble, btn_close,
                    fig_map, fig_scatter, fig_rank, fig_corr, fig_par, fig_pca, fig_bubble):
    trigger = ctx.triggered_id
    if trigger == "close-fullscreen":
        return {"display": "none"}, go.Figure()
    mapping = {
        "open-fullscreen-map": fig_map,
        "open-fullscreen-scatter": fig_scatter,
        "open-fullscreen-rank": fig_rank,
        "open-fullscreen-corr": fig_corr,
        "open-fullscreen-pca": fig_pca,
        "open-fullscreen-bubble": fig_bubble,
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
