# Personal Finance Intelligence System

A smart and interactive **personal finance analyzer** built with **Python** and **Streamlit** to help users understand their spending, track goals, detect unusual expenses, and make better financial decisions.

This project goes beyond simple expense tracking. It combines analytics, rule-based intelligence, and machine learning to turn raw transaction data into meaningful financial insights.

---

## Why this project?

Managing personal finances can often feel overwhelming, especially when expenses are spread across different categories and patterns are not obvious at first glance. This system is designed to make personal finance tracking more practical, visual, and intelligent.

With this app, users can:

- understand where their money is going,
- identify overspending patterns,
- monitor their financial health,
- forecast future expenses,
- optimize budgets,
- and stay on track with financial goals.

---

## Key Features

### Smart Transaction Categorization
Automatically assigns categories to transactions using keyword-based rules, reducing manual effort and improving consistency.

### Monthly Spending Analytics
Provides clear monthly summaries and visual breakdowns of spending behavior across categories.

### Heatmaps and Spending Patterns
Highlights spending trends over time so users can easily spot high-expense days, weeks, or months.

### Anomaly Detection
Detects unusual spending activity such as:
- sudden spending spikes,
- duplicate transactions,
- and abnormal expense behavior.

### Financial Health Score
Generates a score from **0 to 100** based on spending habits, savings discipline, and budget alignment.

### Expense Forecasting
Uses machine learning techniques to estimate future spending and help users plan ahead.

### Budget Optimization Engine
Applies the **50-30-20 budgeting rule** to evaluate whether spending is balanced across needs, wants, and savings.

### Goal Tracking
Tracks progress toward financial goals and alerts users when deadlines are approaching.

### What-If Scenario Simulator
Allows users to test possible financial decisions before making them, such as reducing food delivery expenses or increasing savings.

### Personalized Insights
Creates AI-style observations and suggestions based on the user’s actual financial behavior.

### Risk Radar Chart
Displays financial risk indicators visually for faster interpretation.

### Multiple Integrated Datasets
Uses **4 connected datasets** to create a complete personal finance analysis system.

---

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib / Seaborn / Plotly** *(depending on your implementation)*
- **Scikit-learn**

---

## Project Structure

```bash
finance_project/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
└── data/
    ├── transactions.csv      # User transaction history
    ├── budget.csv            # Monthly budget allocation
    ├── goals.csv             # Financial goals and progress
    └── category_rules.csv    # Rules for automatic categorization
