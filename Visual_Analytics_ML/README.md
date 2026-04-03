# Marketing Analytics Dashboard — React

A fully interactive 2-page marketing analytics dashboard built with **React 18 + Recharts**.  
Four classification ML models are trained on 2019–2020 supermarket sales data and used to predict 2021–2025 values. Switching models instantly re-renders every chart on both pages.

---

## Project Structure

```
marketing-analytics-dashboard/
├── public/
│   └── index.html               # HTML shell
├── src/
│   ├── index.js                 # React DOM entry point
│   ├── App.jsx                  # Root shell — page state, filter state
│   │
│   ├── data/
│   │   └── dataset.js           # Full 3 000-row dataset (2019-2025)
│   │
│   ├── ml/
│   │   ├── models.js            # NaiveBayes, DecisionTree, KNN, RandomForest classes
│   │   └── engine.js            # runModel(), validationAccuracy(), pre-run exports
│   │
│   ├── components/
│   │   ├── Header.jsx           # Top nav bar (logo + page pills + badges)
│   │   ├── ModelBar.jsx         # Horizontal model selector (4 classifier cards)
│   │   ├── FilterPanel.jsx      # Right sidebar (green expandable filter buttons)
│   │   ├── KPICard.jsx          # Blue KPI metric card
│   │   ├── DropFilter.jsx       # Green dropdown chip filter (Branch / Product / Payment)
│   │   └── Toggle.jsx           # Animated on/off toggle switch
│   │
│   ├── pages/
│   │   ├── Page1.jsx            # Sales Overview (KPIs + full-width chart + comparison)
│   │   └── Page2.jsx            # Segment Analysis (5 breakdown charts)
│   │
│   └── styles/
│       └── index.css            # Full design system (dark theme, CSS variables)
│
├── package.json
└── README.md
```

---

## Quick Start

### Prerequisites
- Node.js ≥ 16
- npm ≥ 8

### Install & run

```bash
# 1. Unzip the project
unzip marketing-analytics-dashboard.zip
cd marketing-analytics-dashboard

# 2. Install dependencies
npm install

# 3. Start dev server
npm start
# Opens http://localhost:3000
```

### Build for production

```bash
npm run build
# Output goes to /build — serve with any static host
```

---

## Features

### Model Selector Bar
Four classifier cards sit below the header. Click any to instantly re-predict **all charts** on both pages:

| # | Model | Strategy |
|---|-------|----------|
| 1 | **Naive Bayes** | Gaussian NB with log-likelihood class scoring |
| 2 | **Decision Tree** | CART algorithm, information gain, max depth 5 |
| 3 | **KNN (k=5)** | k-Nearest Neighbours, Euclidean distance |
| 4 | **Random Forest** | 12 trees, depth 4, sqrt feature sampling, bootstrap |

All models classify each transaction into **Low / Med / High** sales, then map predictions back to the class mean for plotting.

---

### Page 1 — Sales Overview

| Section | Content |
|---------|---------|
| **KPI Row** | Total Actual Sales · Predicted Sales · Avg Rating — with delta % vs prior period |
| **Green filters** | Branch · Product Line · Payment Method (dropdown multi-select) |
| **Sales Prediction Graph** | Full-width area chart — actual (solid) vs model prediction (dashed). Training / prediction zones separated by reference line |
| **Model Comparison Graph** | All 4 models overlaid as lines vs actual — active model full opacity |
| **Value Comparison Table** | Year-by-year predicted avg sale for all models, WIN badge marks closest to actual |

---

### Page 2 — Segment Analysis

| Chart | Dimensions |
|-------|-----------|
| Customer Type + Payment + Sales | Grouped bars per customer × payment combination |
| Product Line + Customer + Sales | Actual Member/Normal bars + predicted overlay |
| Product Line + Gender + Sales | Actual Female/Male bars + predicted overlay |
| Gender + Sales + City (Branch) | Per-branch Female vs Male actual + predicted |
| Sales + Branch & City + Customer | Stacked annual bars — actual stack A, predicted stack B |

---

### Right Filter Panel

All green buttons expand in-place (accordion style):

- **Date & Time** — year range slider + shortcut chips (Full / Train / Pred / Recent)
- **Customer Type** — Member / Normal chips
- **Branch / City** — Alex / Cairo / Giza chips *(page 1 only)*
- **Payment Method** — Cash / Credit Card / E-Wallet chips *(page 1 only)*
- **Display Options** — toggle Actual line and Predicted line independently

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Supermarket Sales 2019–2025 |
| Total rows | 3 000 |
| Training set | 865 rows (2019–2020) |
| Test set | 2 135 rows (2021–2025) |
| Features used | branch, customer type, gender, product line, payment, unit price, quantity, month (sin/cos), year |
| Target | Sales amount → binned Low / Med / High |

---

## Design System

All colours, spacing, and typography are managed via CSS custom properties in `src/styles/index.css`:

```css
--bg, --surface, --card, --card2   /* dark background layers      */
--border, --border2                /* subtle dividers             */
--text, --text2, --muted           /* text hierarchy              */
--green, --green-bg, --green-border /* filter panel accent        */
--blue, --blue-bg                  /* KPI card accent             */
--c0 #38bdf8  --c1 #34d399  --c2 #fbbf24  --c3 #f87171  /* model colours */
--actual #6ea8fe                   /* actual data line            */
--font-body: 'Plus Jakarta Sans'
--font-mono: 'IBM Plex Mono'
```
