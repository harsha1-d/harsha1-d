# Decision Tree Model Visualizer
### Visual Analytics System for Supermarket Sales Classification

An interactive Visual Analytics dashboard that trains a CART Decision Tree classifier on 3,000 supermarket transactions (2019–2025) and classifies each transaction as **Low**, **Medium**, or **High** value — with every model decision rendered as a live, interactive visual.

---

## What This Project Does

- Trains a Decision Tree classifier entirely in the browser — no backend, no server
- Renders the full model as an interactive SVG tree with plain-English YES/NO questions
- Provides 8 coordinated visualisations that update instantly when any parameter changes
- Includes a **What-If Transaction Predictor** — type in items and price, get a step-by-step prediction trace
- Gives a market analyst full model control through 6 sliders with no coding required

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | JavaScript (ES6+) | — | All logic, ML, and UI |
| UI Framework | React | 18.2.0 | Component-based interface |
| Charts | Recharts | 2.5.0 | All statistical charts |
| Tree Visual | Raw SVG | — | Decision tree diagram |
| Build Tool | Create React App | 5.0.1 | Bundling and dev server |
| Runtime | Node.js | 16+ | Development environment only |
| Package Manager | npm | 8+ | Dependency management |
| Styling | CSS Variables | — | Dark theme, consistent colours |
| Fonts | Google Fonts | — | IBM Plex Mono + Plus Jakarta Sans |
| ML Algorithm | Custom CART | — | Built from scratch in JavaScript |

> **No external ML library is used.** The entire CART Decision Tree — including weighted entropy, information gain, recursive splitting, and monotonic threshold constraints — is implemented from scratch in `src/ml/models.js`.

---

## Project Structure

```
react-dashboard/
├── package.json                  — dependencies and scripts
├── public/
│   └── index.html                — HTML entry point
└── src/
    ├── App.jsx                   — root component, lifts shared state
    ├── index.js                  — React DOM entry point
    ├── data/
    │   └── dataset.js            — 3,000 transaction rows (embedded)
    ├── ml/
    │   ├── models.js             — CART Decision Tree implementation
    │   └── engine.js             — buildAndEval(), computeDepthCurve(),
    │                               computeCorrelations(), getTopPaths()
    ├── components/
    │   ├── Header.jsx            — top navigation bar
    │   ├── TreeViz.jsx           — SVG decision tree renderer
    │   ├── ConfusionMatrix.jsx   — 3x3 confusion matrix
    │   ├── MetricsReport.jsx     — Precision/Recall/F1/Specificity + CI
    │   ├── TweakPanel.jsx        — all model controls + What-If Predictor
    │   ├── KPICard.jsx           — metric summary cards
    │   ├── Toggle.jsx            — feature toggle component
    │   └── DropFilter.jsx        — dropdown filter component
    ├── pages/
    │   ├── Page1.jsx             — Tree Visualizer page
    │   └── Page2.jsx             — Prediction Analysis page
    └── styles/
        └── index.css             — global CSS variables and theme
```

---

## Prerequisites — Install These First

### 1. Node.js (version 16 or higher)

Node.js is the only thing you need to install on your machine. It includes npm automatically.

**Check if already installed:**
```bash
node --version
npm --version
```

**If not installed — download from:**
```
https://nodejs.org/en/download
```
Choose the **LTS (Long Term Support)** version. Install it like any normal application.

**Verify after installation:**
```bash
node --version    # should show v16.x.x or higher
npm --version     # should show 8.x.x or higher
```

> That is the only installation required. Everything else (React, Recharts, etc.)
> is installed automatically in the next step via npm install.

---

## How to Run the Project

### Step 1 — Extract the zip file

Unzip the submitted file to any folder on your computer.

```
react-dashboard/
    ├── package.json
    ├── public/
    └── src/
```

### Step 2 — Open a terminal in the project folder

**Windows:** Right-click inside the `react-dashboard` folder
             → "Open in Terminal" or "Open PowerShell window here"

**Mac:**     Right-click the `react-dashboard` folder
             → "New Terminal at Folder"

**Linux:**   Open terminal and navigate to the folder:
```bash
cd path/to/react-dashboard
```

### Step 3 — Install dependencies

Run this once. It downloads React, Recharts, and all packages listed in package.json.

```bash
npm install
```

This takes 1–3 minutes. A `node_modules` folder will appear. This is expected.

### Step 4 — Start the application

```bash
npm start
```

The terminal will show:
```
Compiled successfully!
Local:   http://localhost:3000
```

Your browser will open automatically. If it does not, go to:
```
http://localhost:3000
```

### Step 5 — Stop the application

Press `Ctrl + C` in the terminal.

---

## Common Problem — Node 17+ Error

If you see this error after running `npm start`:
```
Error: error:0308010C:digital envelope routines::unsupported
```

Use one of these instead:

**Mac / Linux:**
```bash
NODE_OPTIONS=--openssl-legacy-provider npm start
```

**Windows Command Prompt:**
```bash
set NODE_OPTIONS=--openssl-legacy-provider && npm start
```

**Windows PowerShell:**
```powershell
$env:NODE_OPTIONS="--openssl-legacy-provider"; npm start
```

This is a known compatibility issue between Node 17+ and Create React App 5.

---

## Build for Production (Optional)

Creates an optimised static build that runs without a development server:

```bash
npm run build
```

Open `build/index.html` directly in any browser — no server needed.

---

## How the Machine Learning Works

The CART algorithm lives entirely in `src/ml/models.js`. Three custom constraints
are added on top of standard CART:

**1. Alternating feature rule**
Even depths (0, 2, 4) split only on Quantity.
Odd depths (1, 3, 5) split only on Unit Price.
This creates a legible visual narrative in the tree.

**2. Monotonic threshold constraint**
Each child node uses a threshold strictly greater than its parent for the same
feature. The tree always reads as a natural escalation — lesser value at parent,
higher value at child.

**3. Weighted entropy**
The three class weight sliders feed directly into the entropy formula during
training. Increasing Low Sale weight forces the algorithm to work harder to
separate Low Sale transactions, restructuring the entire tree.

Every slider change triggers a full model re-train via React useMemo. On 865
training rows this completes in under 10 milliseconds.

---

## Dataset

Embedded directly in `src/data/dataset.js` — no file loading or API calls needed.

| Property | Value |
|---|---|
| Total rows | 3,000 |
| Training rows | 865 (years 2019-2020) |
| Test rows | 2,135 (years 2021-2025) |
| Target variable | Sales = Unit Price x Quantity x 1.05 |
| Low Sale class | Sales <= $168 (33rd percentile) |
| Medium Sale class | $168 < Sales <= $385 (66th percentile) |
| High Sale class | Sales > $385 |

---

## Browser Compatibility

| Browser | Minimum Version |
|---|---|
| Google Chrome | 110+ (recommended) |
| Mozilla Firefox | 110+ |
| Microsoft Edge | 110+ |
| Safari | 16+ |

---

## Quick Reference

```bash
npm install        # install all dependencies  (run once after unzipping)
npm start          # start dev server at http://localhost:3000
npm run build      # create production build in /build folder
```
