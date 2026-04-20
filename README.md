# 💰 Personal Finance Intelligence System

A complete data-driven personal finance analyzer built with Python and Streamlit.

## Features
- Smart auto-categorization of transactions
- Monthly spending analytics & heatmaps
- Anomaly detection (spending spikes, duplicates)
- Financial Health Score (0–100)
- Expense forecasting (ML-based)
- Budget optimization engine (50-30-20 rule)
- Goal tracking with deadline alerts
- What-If scenario simulator
- Personalized AI-style insights
- Risk radar chart
- 4 integrated datasets

## Quick Start

### Step 1 — Install Python (if not installed)
Download from: https://www.python.org/downloads/

### Step 2 — Open terminal / CMD in this folder
Right-click the `finance_project` folder → Open in Terminal

### Step 3 — Install dependencies
```
pip install -r requirements.txt
```

### Step 4 — Run the app
```
streamlit run app.py
```

The app will open automatically in your browser at http://localhost:8501

## Project Structure
```
finance_project/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python packages needed
├── README.md
└── data/
    ├── transactions.csv      ← Your spending transactions
    ├── budget.csv            ← Monthly budgets per category
    ├── goals.csv             ← Financial goals & progress
    └── category_rules.csv    ← Keywords for auto-categorization
```

## Adding Your Own Data
Edit `data/transactions.csv` and add rows in this format:
```
date,description,amount,category
2024-05-01,Swiggy order,350,
```
Leave the `category` column blank — the app auto-fills it!
