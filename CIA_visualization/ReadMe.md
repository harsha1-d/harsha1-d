# Energy Analytics Dashboard

A professional interactive visualization tool for energy analysts to explore the CIA World Factbook energy dataset across 260+ countries.

## Features

- **Choropleth Map**: Visualize global energy indicator distributions geographically.
- **Scatter Plot Analysis**: Explore bivariate relationships between indicators with dynamic X/Y axis selection.
- **Bar Chart Rankings**: View Top-N countries for any indicator with adjustable depth.
- **Correlation Heatmap**: Discover pairwise correlations between all indicators in a selectable matrix.
- **Parallel Coordinates**: Multivariate exploration with high-performance interactive brushing and linked table results.
- **SPLOM (Scatter Plot Matrix)**: Pairwise relationships across multiple dimensions (hidden by default).
- **PCA & K-Means**: Identify country groupings with automated dimensionality reduction and clustering.
- **Bidirectional Highlighting**: Click on any chart (Map, Scatter, Rank, PCA) to highlight that country across all views.
- **Region Filtering**: Filter by continent (Africa, Asia, Europe, North America, South America, Oceania) across all visualizations.
- **KPI Dashboard**: Quick summary statistics including Min, Max, and Mean for the selected indicator.

## Installation

### Requirements

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install dash plotly pandas numpy pycountry pycountry-convert scikit-learn
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application

```bash
python cia_v1.py
```

The dashboard will start at **http://127.0.0.1:8050**

### User Guide

For a detailed walkthrough on how to use the dashboard for energy analysis, see:
- [Energy Analyst Guide](file:///e:/Projects/Vis/energy_analyst_guide.txt)

### Datasets Required

Place the following CSV files in the `Datasets/` directory:

- `energyAnalyst.csv` - Main energy dataset with merged indicators.
- `geography_data.csv` - Geographic attributes (Area, Coastline, Land Use).
- `government_and_civics_data.csv` - Government type information.

## Data Categories

The tool integrates 21 indicators across 5 categories:

| Category | Indicators |
|----------|------------|
| **Energy** | Electricity access, generating capacity, coal, petroleum, natural gas, CO2 emissions |
| **Economy** | Real GDP (PPP), GDP per Capita, Exports, Imports |
| **Demographics** | Total Population |
| **Transport** | Gas pipelines, Oil pipelines (km) |
| **Geography** | Total Area, Land Area, Coastline, Forest Land, Agricultural Land |

## Project Structure

```
Vis/
├── cia_v1.py                  # Main application (Dash + Plotly)
├── ReadMe.md                  # This file
├── energy_analyst_guide.txt   # Step-by-step user guide
├── Datasets/                  # CSV Data source directory
│   ├── energyAnalyst.csv
│   ├── geography_data.csv
│   └── government_and_civics_data.csv
└── Report/                    # Academic reporting files
    ├── template.tex           # LaTeX source
    ├── template.pdf           # Generated PDF report
    └── *.jpg/*.jpeg           # Visualization exports
```

## Technical Implementation

### What We Built (Own Code)

| Component | Description |
|-----------|-------------|
| **Data Cleaning Pipeline** | Custom regex-based normalization of CIA dataset (units, currencies, missing values) in `clean_numeric_column()` |
| **Country Name Matching** | Manual CIA → Gapminder mapping table (`CIA_NAME_FIXES`) for 30+ naming discrepancies |
| **ISO Code Resolution** | Fallback lookup using pycountry with fuzzy matching and manual overrides |
| **Coordinated Multiple Views** | Bidirectional click/brush synchronization across 6+ visualizations using Dash callbacks and `dcc.Store` |
| **Parallel Coordinates Brushing** | Custom constraint extraction from `restyleData` and linked table filtering |
| **PCA + K-Means Integration** | Wrapper around sklearn with dynamic parameter selection and cluster coloring |
| **Professional Dark Theme** | Custom CSS for scrollbars, dropdowns, panels, hover effects, and animations |
| **State Management** | Cross-callback synchronization using `dcc.Store` for selections, brushes, and filters |

### External Libraries Used

- **Dash**: Web application framework for Python.
- **Plotly**: High-performance interactive visualization.
- **Pandas/NumPy**: Data processing and numeric normalization.
- **scikit-learn**: Dimensionality reduction (PCA) and K-Means clustering.
- **pycountry**: ISO alpha-3 code mapping and continent conversion.

### Core Architecture

- **Data Cleaning**: Unified normalization of units (sq km, kW, bbl/day, etc.) using regex-based cleaning.
- **State Management**: Uses `dcc.Store` for cross-callback synchronization (selections, brushes, filters).
- **Responsive Layout**: Custom CSS for a professional "Dark Mode" aesthetic and optimized panel layouts.

## Authors

- Harshavardhan Dharman
- Surya Kannan
- Shashank Venkatesha
- Leela Karthikeyan Haribabu

*Eindhoven University of Technology - Visualization Course (January 2026)*

## License

Academic project - All rights reserved.
