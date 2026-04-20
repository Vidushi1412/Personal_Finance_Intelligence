import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Intelligence System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .score-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f7ff;
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .alert-box {
        background: #fff3f0;
        border-left: 4px solid #e53935;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .success-box {
        background: #f0fff4;
        border-left: 4px solid #43a047;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #1a73e8;
        padding-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    transactions = pd.read_csv('data/transactions.csv')
    budget = pd.read_csv('data/budget.csv')
    goals = pd.read_csv('data/goals.csv')
    rules = pd.read_csv('data/category_rules.csv')
    return transactions, budget, goals, rules


# ─────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────
def clean_data(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates()
    df = df.dropna(subset=['amount'])
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    return df


# ─────────────────────────────────────────────
# CATEGORIZATION
# ─────────────────────────────────────────────
def categorize(description, rules_df):
    desc = description.lower()
    for _, row in rules_df.iterrows():
        if row['keyword'] in desc:
            return row['category']
    return 'Other'


def get_subcategory(description, rules_df):
    desc = description.lower()
    for _, row in rules_df.iterrows():
        if row['keyword'] in desc:
            return row['subcategory']
    return 'Misc'


def apply_categories(df, rules):
    df = df.copy()
    df['category'] = df['description'].apply(lambda x: categorize(x, rules))
    df['subcategory'] = df['description'].apply(lambda x: get_subcategory(x, rules))
    return df


# ─────────────────────────────────────────────
# HEALTH SCORE
# ─────────────────────────────────────────────
def calculate_health_score(df, budget_df, income):
    score = 100
    breakdown = {}

    total_months = df['month'].nunique()
    total_spent = df['amount'].sum()
    monthly_avg = total_spent / max(total_months, 1)

    # 1. Budget adherence (25 pts)
    total_budget = budget_df['budget_amount'].sum() / budget_df['month'].nunique() if 'month' in budget_df.columns else budget_df['budget_amount'].sum() / 4
    ratio = monthly_avg / total_budget
    if ratio <= 1.0:
        budget_score = 25
    elif ratio <= 1.1:
        budget_score = 18
    elif ratio <= 1.2:
        budget_score = 10
    else:
        budget_score = 0
    breakdown['Budget Adherence'] = (budget_score, 25)

    # 2. Savings rate (30 pts)
    savings = income - monthly_avg
    savings_rate = savings / income if income > 0 else 0
    if savings_rate >= 0.30:
        savings_score = 30
    elif savings_rate >= 0.20:
        savings_score = 22
    elif savings_rate >= 0.10:
        savings_score = 12
    elif savings_rate > 0:
        savings_score = 5
    else:
        savings_score = 0
    breakdown['Savings Rate'] = (savings_score, 30)

    # 3. Spending consistency (20 pts)
    monthly_totals = df.groupby('month')['amount'].sum()
    cv = monthly_totals.std() / monthly_totals.mean() if monthly_totals.mean() > 0 else 1
    if cv < 0.10:
        consistency_score = 20
    elif cv < 0.20:
        consistency_score = 15
    elif cv < 0.35:
        consistency_score = 8
    else:
        consistency_score = 0
    breakdown['Spending Consistency'] = (consistency_score, 20)

    # 4. Category diversity (25 pts) — not putting everything in one bucket
    cat_shares = df.groupby('category')['amount'].sum() / total_spent
    max_share = cat_shares.max()
    if max_share < 0.40:
        diversity_score = 25
    elif max_share < 0.55:
        diversity_score = 18
    elif max_share < 0.70:
        diversity_score = 8
    else:
        diversity_score = 0
    breakdown['Category Balance'] = (diversity_score, 25)

    total = sum(v[0] for v in breakdown.values())

    if total >= 80:
        label, color = "Excellent", "#43a047"
    elif total >= 60:
        label, color = "Good", "#1a73e8"
    elif total >= 40:
        label, color = "Moderate", "#fb8c00"
    else:
        label, color = "Needs Attention", "#e53935"

    return total, label, color, breakdown


# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────
def detect_anomalies(df):
    anomalies = []

    # Spike per category per month
    mc = df.groupby(['month', 'category'])['amount'].sum().reset_index()
    avg = mc.groupby('category')['amount'].mean()
    std = mc.groupby('category')['amount'].std().fillna(0)

    for _, row in mc.iterrows():
        cat_avg = avg[row['category']]
        cat_std = std[row['category']]
        if cat_std > 0:
            z = (row['amount'] - cat_avg) / cat_std
            pct = ((row['amount'] - cat_avg) / cat_avg) * 100
            if z > 1.5 and pct > 30:
                anomalies.append({
                    'Month': row['month'],
                    'Category': row['category'],
                    'Amount': row['amount'],
                    'Avg': round(cat_avg, 0),
                    'Change %': round(pct, 1),
                    'Severity': '🔴 High' if pct > 80 else '🟡 Medium'
                })

    # Duplicate transactions (same amount + same day)
    dupes = df[df.duplicated(subset=['date', 'amount'], keep=False)]

    return pd.DataFrame(anomalies), dupes


# ─────────────────────────────────────────────
# FORECASTING
# ─────────────────────────────────────────────
def forecast_spending(df, periods=3):
    monthly = df.groupby('month')['amount'].sum().reset_index()
    monthly['idx'] = range(len(monthly))

    X = monthly[['idx']].values
    y = monthly['amount'].values

    model = LinearRegression()
    model.fit(X, y)

    future_idx = np.array([[len(monthly) + i] for i in range(periods)])
    predictions = model.predict(future_idx)

    last_month = pd.Period(monthly['month'].iloc[-1], freq='M')
    future_months = [str(last_month + i + 1) for i in range(periods)]

    return monthly, future_months, predictions, model


def forecast_by_category(df, periods=1):
    results = {}
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat].groupby('month')['amount'].sum().reset_index()
        if len(cat_df) >= 2:
            cat_df['idx'] = range(len(cat_df))
            model = LinearRegression()
            model.fit(cat_df[['idx']], cat_df['amount'])
            pred = model.predict([[len(cat_df)]])[0]
            results[cat] = max(0, round(pred, 0))
    return results


# ─────────────────────────────────────────────
# RECURRING DETECTION
# ─────────────────────────────────────────────
def detect_recurring(df):
    rec = df.groupby('description').agg(
        Count=('amount', 'count'),
        Avg_Amount=('amount', 'mean'),
        Total=('amount', 'sum'),
        Category=('category', 'first')
    ).reset_index()
    rec = rec[rec['Count'] >= 2].sort_values('Count', ascending=False)
    rec['Avg_Amount'] = rec['Avg_Amount'].round(0)
    rec['Total'] = rec['Total'].round(0)
    return rec


# ─────────────────────────────────────────────
# PERSONALIZED INSIGHTS
# ─────────────────────────────────────────────
def generate_insights(df, budget_df, income, score):
    insights = []
    tips = []

    monthly_avg = df.groupby('month')['amount'].sum().mean()
    savings = income - monthly_avg

    # Saving insight
    if savings < 0:
        insights.append(f"🔴 You're spending ₹{abs(savings):.0f} MORE than you earn monthly. This is unsustainable.")
        tips.append("Cut discretionary spending immediately — start with Shopping and Entertainment.")
    elif savings < income * 0.1:
        insights.append(f"🟡 You're saving only ₹{savings:.0f}/month ({savings/income*100:.1f}% of income). Target 20%+.")
        tips.append("Try the 50-30-20 rule: 50% needs, 30% wants, 20% savings.")
    else:
        insights.append(f"🟢 Great! You're saving ₹{savings:.0f}/month ({savings/income*100:.1f}% of income).")

    # Category-specific insights
    cat_monthly = df.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)

    for cat in df['category'].unique():
        if cat in cat_monthly.columns:
            series = cat_monthly[cat]
            if len(series) >= 2:
                trend = series.iloc[-1] - series.iloc[-2]
                if trend > 500:
                    insights.append(f"📈 Your {cat} spending rose by ₹{trend:.0f} last month — worth reviewing.")
                if series.mean() > 0:
                    pct = (series.iloc[-1] / series.mean() - 1) * 100
                    if pct > 50:
                        tips.append(f"Your {cat} spending is {pct:.0f}% above your own average. Consider setting a firm limit.")

    # Peak spending
    peak_day = df.groupby('weekday')['amount'].mean().idxmax()
    insights.append(f"📅 You spend the most on {peak_day}s on average. Plan ahead for that day.")

    # Food delivery
    food_delivery = df[df['subcategory'] == 'Delivery']['amount'].sum()
    food_total = df[df['category'] == 'Food']['amount'].sum()
    if food_total > 0 and food_delivery / food_total > 0.5:
        insights.append(f"🍕 {food_delivery/food_total*100:.0f}% of your food budget goes to delivery apps. Cooking at home can save ₹{food_delivery*0.6:.0f}.")

    # Top spender
    top_cat = df.groupby('category')['amount'].sum().idxmax()
    top_amt = df.groupby('category')['amount'].sum().max()
    if top_cat != 'Rent':
        tips.append(f"'{top_cat}' is your biggest discretionary spend at ₹{top_amt:.0f}. Even a 15% cut saves ₹{top_amt*0.15:.0f}.")

    return insights, tips


# ─────────────────────────────────────────────
# RISK DETECTION
# ─────────────────────────────────────────────
def detect_risks(df, budget_df, income):
    risks = []
    monthly_avg = df.groupby('month')['amount'].sum().mean()

    if monthly_avg > income:
        risks.append(("CRITICAL", "Spending exceeds income", f"You spend ₹{monthly_avg:.0f} but earn ₹{income}. Deficit: ₹{monthly_avg-income:.0f}/month"))

    cat_budget = budget_df.groupby('category')['budget_amount'].mean()
    cat_spent = df.groupby(['month', 'category'])['amount'].sum().groupby('category').mean()

    for cat in cat_spent.index:
        if cat in cat_budget.index:
            overspend = cat_spent[cat] - cat_budget[cat]
            if overspend > 0:
                risks.append(("WARNING", f"{cat} over budget", f"Avg monthly overspend: ₹{overspend:.0f} ({overspend/cat_budget[cat]*100:.0f}% over limit)"))

    monthly_totals = df.groupby('month')['amount'].sum()
    if len(monthly_totals) >= 2:
        last_two = monthly_totals.iloc[-2:]
        growth = (last_two.iloc[-1] - last_two.iloc[-2]) / last_two.iloc[-2] * 100
        if growth > 20:
            risks.append(("WARNING", "Rapid spending growth", f"Spending jumped {growth:.1f}% last month"))

    savings_rate = (income - monthly_avg) / income
    if savings_rate < 0.10:
        risks.append(("WARNING", "Low savings rate", f"Only {savings_rate*100:.1f}% of income saved. Target is 20%+"))

    return risks


# ─────────────────────────────────────────────
# WHAT-IF SIMULATION
# ─────────────────────────────────────────────
def simulate_whatif(df, income, changes):
    baseline_monthly = df.groupby('month')['amount'].sum().mean()
    baseline_savings = income - baseline_monthly

    cat_monthly = df.groupby('category')['amount'].sum() / df['month'].nunique()
    new_spending = cat_monthly.copy()

    for cat, pct_change in changes.items():
        if cat in new_spending.index:
            new_spending[cat] = new_spending[cat] * (1 - pct_change / 100)

    new_total = new_spending.sum()
    new_savings = income - new_total
    extra_savings = new_savings - baseline_savings

    return baseline_monthly, new_total, baseline_savings, new_savings, extra_savings


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    income = st.number_input("💼 Monthly Income (₹)", min_value=5000, max_value=500000, value=25000, step=1000)
    st.markdown("---")
    st.markdown("## 🗂️ Navigation")
    page = st.radio("Go to", [
        "🏠 Dashboard",
        "📊 Spending Analytics",
        "🚨 Anomaly Detection",
        "🏥 Health Score",
        "🔮 Predictions",
        "💡 Budget Engine",
        "🎯 Goal Tracker",
        "🔬 What-If Simulator",
        "💬 Insights & Tips",
        "⚠️ Risk Report",
        "📋 Raw Data"
    ])
    st.markdown("---")
    st.markdown("### 📁 4 Datasets Loaded")
    st.success("✅ transactions.csv")
    st.success("✅ budget.csv")
    st.success("✅ goals.csv")
    st.success("✅ category_rules.csv")


# ─────────────────────────────────────────────
# LOAD + PROCESS
# ─────────────────────────────────────────────
transactions_raw, budget, goals, rules = load_data()
transactions = clean_data(transactions_raw)
transactions = apply_categories(transactions, rules)
budget['month'] = budget['month'].astype(str)

score, score_label, score_color, score_breakdown = calculate_health_score(transactions, budget, income)
anomaly_df, dupe_df = detect_anomalies(transactions)
monthly_data, future_months, predictions, forecast_model = forecast_spending(transactions)
risks = detect_risks(transactions, budget, income)
insights, tips = generate_insights(transactions, budget, income, score)
recurring = detect_recurring(transactions)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">💰 Personal Finance Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Data-driven analysis · Predictions · Optimization · Actionable Insights</div>', unsafe_allow_html=True)
st.markdown("---")


# ═══════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<div class="section-header">📌 At a Glance</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    total_spent = transactions['amount'].sum()
    monthly_avg = transactions.groupby('month')['amount'].sum().mean()
    savings = income - monthly_avg
    num_transactions = len(transactions)
    num_months = transactions['month'].nunique()

    col1.metric("Total Spent", f"₹{total_spent:,.0f}", f"{num_months} months")
    col2.metric("Monthly Avg", f"₹{monthly_avg:,.0f}", "average")
    col3.metric("Est. Monthly Savings", f"₹{savings:,.0f}", f"{savings/income*100:.1f}% of income")
    col4.metric("Transactions", num_transactions, f"{num_transactions/num_months:.0f}/month avg")
    col5.metric("Health Score", f"{score}/100", score_label)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Monthly Spending Trend</div>', unsafe_allow_html=True)
        monthly = transactions.groupby('month')['amount'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.fill_between(range(len(monthly)), monthly['amount'], alpha=0.15, color='#1a73e8')
        ax.plot(range(len(monthly)), monthly['amount'], color='#1a73e8', linewidth=2.5, marker='o', markersize=6)
        ax.axhline(income, color='#43a047', linestyle='--', linewidth=1.5, label=f'Income ₹{income:,}')
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(monthly['month'], rotation=45, fontsize=8)
        ax.set_ylabel('₹ Amount')
        ax.legend(fontsize=8)
        ax.set_title('Monthly Spending vs Income', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">🍕 Category Breakdown</div>', unsafe_allow_html=True)
        cat_spend = transactions.groupby('category')['amount'].sum().sort_values(ascending=False)
        colors = ['#1a73e8', '#43a047', '#fb8c00', '#e53935', '#8e24aa', '#00acc1', '#f4511e', '#7cb342', '#039be5', '#fdd835']
        fig, ax = plt.subplots(figsize=(7, 3.5))
        wedges, texts, autotexts = ax.pie(cat_spend.values, labels=cat_spend.index,
                                           autopct='%1.1f%%', colors=colors[:len(cat_spend)],
                                           startangle=140, pctdistance=0.8)
        for t in texts: t.set_fontsize(8)
        for t in autotexts: t.set_fontsize(7)
        ax.set_title('Spending by Category', fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">📅 Spending Heatmap by Weekday & Month</div>', unsafe_allow_html=True)
    heatmap_data = transactions.pivot_table(values='amount', index='weekday', columns='month', aggfunc='sum', fill_value=0)
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='Blues', ax=ax, linewidths=0.5, cbar_kws={'label': '₹ Spent'})
    ax.set_title('Daily Spending Patterns (₹)', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if len(risks) > 0:
        st.markdown('<div class="section-header">⚠️ Active Alerts</div>', unsafe_allow_html=True)
        for severity, title, detail in risks[:3]:
            icon = "🔴" if severity == "CRITICAL" else "🟡"
            st.markdown(f'<div class="alert-box">{icon} <b>{title}</b>: {detail}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE: SPENDING ANALYTICS
# ═══════════════════════════════════════════
elif page == "📊 Spending Analytics":
    st.markdown('<div class="section-header">📊 Spending Analytics Engine</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Monthly Trends", "🗂️ Category Deep Dive", "🔄 Recurring Expenses", "📆 Daily Patterns"])

    with tab1:
        monthly_cat = transactions.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 5))
        monthly_cat.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        ax.set_title('Monthly Spending by Category (Stacked)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('₹ Amount')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.markdown("**Month-over-month change**")
        monthly_total = transactions.groupby('month')['amount'].sum().reset_index()
        monthly_total['MoM Change %'] = monthly_total['amount'].pct_change() * 100
        monthly_total['amount'] = monthly_total['amount'].map('₹{:,.0f}'.format)
        monthly_total['MoM Change %'] = monthly_total['MoM Change %'].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
        monthly_total.columns = ['Month', 'Total Spent', 'MoM Change %']
        st.dataframe(monthly_total, use_container_width=True)

    with tab2:
        cat_selected = st.selectbox("Select Category", sorted(transactions['category'].unique()))
        cat_df = transactions[transactions['category'] == cat_selected]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spent", f"₹{cat_df['amount'].sum():,.0f}")
        col2.metric("Monthly Avg", f"₹{cat_df.groupby('month')['amount'].sum().mean():,.0f}")
        col3.metric("Transactions", len(cat_df))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        monthly_cat = cat_df.groupby('month')['amount'].sum()
        axes[0].bar(range(len(monthly_cat)), monthly_cat.values, color='#1a73e8', alpha=0.8)
        axes[0].set_xticks(range(len(monthly_cat)))
        axes[0].set_xticklabels(monthly_cat.index, rotation=45, fontsize=8)
        axes[0].set_title(f'{cat_selected} — Monthly Spend', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('₹')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        subcat = cat_df.groupby('subcategory')['amount'].sum()
        if len(subcat) > 1:
            axes[1].pie(subcat.values, labels=subcat.index, autopct='%1.0f%%', startangle=90)
            axes[1].set_title(f'{cat_selected} — Sub-category Split', fontsize=10, fontweight='bold')
        else:
            axes[1].bar(subcat.index, subcat.values, color='#43a047')
            axes[1].set_title('Sub-category Breakdown', fontsize=10, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"**All {cat_selected} transactions**")
        st.dataframe(cat_df[['date','description','amount','subcategory']].sort_values('date', ascending=False), use_container_width=True)

    with tab3:
        st.markdown("**🔄 Recurring / Habitual Expenses Detected**")
        st.dataframe(recurring.rename(columns={'description': 'Description', 'Count': 'Months Seen', 'Avg_Amount': 'Avg ₹', 'Total': 'Total ₹', 'Category': 'Category'}), use_container_width=True)

        fixed_total = recurring[recurring['Count'] >= 3]['Avg_Amount'].sum()
        variable_total = transactions['amount'].sum() / transactions['month'].nunique() - fixed_total
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(['Fixed (Recurring)', 'Variable'], [fixed_total, variable_total], color=['#1a73e8', '#fb8c00'])
        ax.set_xlabel('₹ Monthly')
        ax.set_title('Fixed vs Variable Expenses', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.markdown("**Peak spending by day of week**")
        day_spend = transactions.groupby('weekday')['amount'].agg(['mean', 'sum', 'count']).reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).fillna(0)
        day_spend.columns = ['Avg Spend ₹', 'Total Spend ₹', 'Transactions']

        fig, ax = plt.subplots(figsize=(10, 3.5))
        bars = ax.bar(day_spend.index, day_spend['Avg Spend ₹'], color='#1a73e8', alpha=0.85)
        peak_idx = day_spend['Avg Spend ₹'].idxmax()
        bars[list(day_spend.index).index(peak_idx)].set_color('#e53935')
        ax.set_title('Average Spending by Day of Week (🔴 = Peak Day)', fontsize=10, fontweight='bold')
        ax.set_ylabel('₹ Average')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.dataframe(day_spend.style.format('₹{:,.0f}', subset=['Avg Spend ₹', 'Total Spend ₹']), use_container_width=True)


# ═══════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ═══════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.markdown('<div class="section-header">🚨 Anomaly Detection System</div>', unsafe_allow_html=True)

    if len(anomaly_df) > 0:
        st.error(f"⚠️ {len(anomaly_df)} spending anomalies detected!")
        st.dataframe(anomaly_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        colors_map = {'🔴 High': '#e53935', '🟡 Medium': '#fb8c00'}
        bar_colors = [colors_map.get(s, '#1a73e8') for s in anomaly_df['Severity']]
        x_labels = anomaly_df['Category'] + '\n' + anomaly_df['Month']
        ax.bar(x_labels, anomaly_df['Change %'], color=bar_colors, alpha=0.85)
        ax.axhline(30, color='gray', linestyle='--', linewidth=1, label='30% threshold')
        ax.set_title('Spending Spikes by Category & Month (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('% above average')
        ax.tick_params(axis='x', rotation=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        st.pyplot(fig)
        plt.close()
    else:
        st.success("✅ No anomalies detected! Your spending is consistent.")

    st.markdown("---")
    st.markdown("**🔁 Possible Duplicate Transactions**")
    if len(dupe_df) > 0:
        st.warning(f"{len(dupe_df)} potentially duplicate transactions found")
        st.dataframe(dupe_df[['date', 'description', 'amount', 'category']], use_container_width=True)
    else:
        st.success("✅ No duplicate transactions found.")

    st.markdown("---")
    st.markdown("**📊 Spending Distribution per Category**")
    fig, ax = plt.subplots(figsize=(12, 4))
    cats = transactions['category'].unique()
    data = [transactions[transactions['category'] == c]['amount'].values for c in cats]
    bp = ax.boxplot(data, labels=cats, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#1a73e820')
    ax.set_title('Spending Distribution per Category (outliers = anomalies)', fontsize=10, fontweight='bold')
    ax.set_ylabel('₹ Amount')
    ax.tick_params(axis='x', rotation=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════
# PAGE: HEALTH SCORE
# ═══════════════════════════════════════════
elif page == "🏥 Health Score":
    st.markdown('<div class="section-header">🏥 Financial Health Score</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        theta = np.linspace(0, np.pi, 100)
        ax.plot(np.append(theta, theta[::-1]), [1]*100 + [0]*100, alpha=0)
        score_angle = np.pi * (1 - score / 100)
        ax.annotate('', xy=(np.cos(score_angle) * 0.6, np.sin(score_angle) * 0.6),
                    xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=score_color, lw=3))

        for i, (color, limit) in enumerate(zip(['#e53935', '#fb8c00', '#1a73e8', '#43a047'], [40, 60, 80, 100])):
            arc = np.linspace(np.pi * (1 - limit/100), np.pi * (1 - (limit-20)/100 if i > 0 else 1), 50)
            ax.fill_between(arc, 0.55, 1.0, alpha=0.3, color=color)

        ax.set_ylim(0, 1)
        ax.set_xlim(-1.2, 1.2)
        ax.axis('off')
        ax.text(0, -0.1, f"{score}", ha='center', va='center', fontsize=36, fontweight='bold', color=score_color, transform=ax.transData)
        ax.text(0, -0.35, score_label, ha='center', va='center', fontsize=14, color=score_color, transform=ax.transData)
        ax.set_title('Your Financial Health', fontsize=11, fontweight='bold', pad=10)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Score Breakdown**")
        for factor, (got, out_of) in score_breakdown.items():
            pct = got / out_of
            bar_color = '#43a047' if pct >= 0.8 else '#fb8c00' if pct >= 0.5 else '#e53935'
            st.markdown(f"**{factor}**: {got}/{out_of} pts")
            st.progress(pct)

        st.markdown("---")
        st.markdown("**What each factor means:**")
        st.markdown("- **Budget Adherence** — Are you staying within set limits?")
        st.markdown("- **Savings Rate** — % of income being saved (target: 20%+)")
        st.markdown("- **Spending Consistency** — Are monthly totals stable?")
        st.markdown("- **Category Balance** — Is spending spread across needs?")

    st.markdown("---")
    st.markdown('<div class="section-header">🔬 Score by Month</div>', unsafe_allow_html=True)
    months = sorted(transactions['month'].unique())
    monthly_scores = []
    for m in months:
        m_df = transactions[transactions['month'] == m]
        m_budget = budget[budget['month'] == m] if 'month' in budget.columns else budget
        s, _, _, _ = calculate_health_score(m_df, m_budget, income)
        monthly_scores.append(s)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(months, monthly_scores, marker='o', linewidth=2.5, color='#1a73e8', markersize=8)
    ax.fill_between(months, monthly_scores, alpha=0.1, color='#1a73e8')
    ax.axhline(80, color='#43a047', linestyle='--', linewidth=1, label='Excellent threshold (80)')
    ax.axhline(60, color='#fb8c00', linestyle='--', linewidth=1, label='Good threshold (60)')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Score')
    ax.set_title('Financial Health Score — Monthly Trend', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════
# PAGE: PREDICTIONS
# ═══════════════════════════════════════════
elif page == "🔮 Predictions":
    st.markdown('<div class="section-header">🔮 Predictive Analytics</div>', unsafe_allow_html=True)

    periods = st.slider("Forecast how many months ahead?", 1, 6, 3)
    _, future_months_p, predictions_p, _ = forecast_spending(transactions, periods)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Overall Spending Forecast**")
        fig, ax = plt.subplots(figsize=(8, 4))
        hist_months = monthly_data['month'].tolist()
        hist_vals = monthly_data['amount'].tolist()
        all_labels = hist_months + future_months_p
        all_vals = hist_vals + predictions_p.tolist()

        ax.plot(range(len(hist_months)), hist_vals, color='#1a73e8', linewidth=2.5, marker='o', markersize=6, label='Actual')
        ax.plot(range(len(hist_months)-1, len(all_labels)), [hist_vals[-1]] + predictions_p.tolist(),
                color='#e53935', linewidth=2, linestyle='--', marker='s', markersize=6, label='Forecast')
        ax.axvline(len(hist_months)-1, color='gray', linestyle=':', linewidth=1)
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, fontsize=8)
        ax.set_ylabel('₹')
        ax.set_title(f'Spending Forecast — Next {periods} Month(s)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Predicted Amounts**")
        for m, p in zip(future_months_p, predictions_p):
            expected_savings = income - p
            st.metric(m, f"₹{p:,.0f}", f"Est. savings: ₹{expected_savings:,.0f}")

    st.markdown("---")
    st.markdown('<div class="section-header">📂 Category-Wise Forecast (Next Month)</div>', unsafe_allow_html=True)
    cat_forecasts = forecast_by_category(transactions)

    fig, ax = plt.subplots(figsize=(10, 4))
    cats = list(cat_forecasts.keys())
    vals = list(cat_forecasts.values())
    colors_bar = ['#e53935' if v > 3000 else '#fb8c00' if v > 1500 else '#43a047' for v in vals]
    ax.barh(cats, vals, color=colors_bar, alpha=0.85)
    ax.set_xlabel('₹ Predicted Spend')
    ax.set_title('Category-wise Forecast for Next Month', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(v + 50, i, f'₹{v:,.0f}', va='center', fontsize=8)
    st.pyplot(fig)
    plt.close()

    total_forecast = sum(cat_forecasts.values())
    st.info(f"📌 Total predicted spend next month: **₹{total_forecast:,.0f}** | Projected savings: **₹{income - total_forecast:,.0f}**")


# ═══════════════════════════════════════════
# PAGE: BUDGET ENGINE
# ═══════════════════════════════════════════
elif page == "💡 Budget Engine":
    st.markdown('<div class="section-header">💡 Budget Optimization Engine</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Your Budget vs Actual Spending**")
        avg_budget = budget.groupby('category')['budget_amount'].mean()
        avg_actual = transactions.groupby('category')['amount'].sum() / transactions['month'].nunique()
        compare = pd.DataFrame({'Budget': avg_budget, 'Actual': avg_actual}).fillna(0)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(compare))
        w = 0.38
        bars1 = ax.bar(x - w/2, compare['Budget'], w, label='Budget', color='#1a73e8', alpha=0.8)
        bars2 = ax.bar(x + w/2, compare['Actual'], w, label='Actual', color='#e53935', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(compare.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('₹ Monthly')
        ax.set_title('Budget vs Actual (Monthly Avg)', fontsize=10, fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**50-30-20 Rule Allocation**")
        needs = income * 0.50
        wants = income * 0.30
        savings_amt = income * 0.20

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie([needs, wants, savings_amt], labels=[f'Needs\n₹{needs:,.0f}', f'Wants\n₹{wants:,.0f}', f'Savings\n₹{savings_amt:,.0f}'],
               colors=['#1a73e8', '#fb8c00', '#43a047'], autopct='%1.0f%%', startangle=90, textprops={'fontsize': 10})
        ax.set_title(f'Ideal Budget (Income ₹{income:,})', fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**💬 Smart Budget Suggestions**")
    for cat, actual in avg_actual.items():
        if cat in avg_budget.index:
            diff = actual - avg_budget[cat]
            if diff > 200:
                saving_potential = diff * 0.6
                st.markdown(f'<div class="alert-box">🔴 <b>{cat}</b>: Over budget by ₹{diff:.0f}/month. Reduce by 60% → save ₹{saving_potential:.0f}/month.</div>', unsafe_allow_html=True)
            elif diff < -200:
                st.markdown(f'<div class="success-box">🟢 <b>{cat}</b>: ₹{abs(diff):.0f} under budget. Great discipline!</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📌 Budget vs Actual Table**")
    compare['Over/Under ₹'] = compare['Actual'] - compare['Budget']
    compare['Status'] = compare['Over/Under ₹'].apply(lambda x: '🔴 Over' if x > 0 else '🟢 Under')
    compare = compare.round(0)
    st.dataframe(compare.style.format('₹{:,.0f}', subset=['Budget', 'Actual', 'Over/Under ₹']), use_container_width=True)


# ═══════════════════════════════════════════
# PAGE: GOAL TRACKER
# ═══════════════════════════════════════════
elif page == "🎯 Goal Tracker":
    st.markdown('<div class="section-header">🎯 Financial Goal Tracking System</div>', unsafe_allow_html=True)

    for _, goal in goals.iterrows():
        remaining = goal['target_amount'] - goal['saved_so_far']
        months_needed = remaining / goal['monthly_contribution']
        progress = goal['saved_so_far'] / goal['target_amount']
        priority_color = {'High': '#e53935', 'Medium': '#fb8c00', 'Low': '#43a047'}.get(goal['priority'], '#1a73e8')

        with st.expander(f"🎯 {goal['goal_name']} — {goal['priority']} Priority | {progress*100:.0f}% done"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Target", f"₹{goal['target_amount']:,}")
            c2.metric("Saved", f"₹{goal['saved_so_far']:,}")
            c3.metric("Remaining", f"₹{remaining:,}")
            c4.metric("Months to Goal", f"{months_needed:.1f}")

            st.progress(progress)
            deadline = pd.to_datetime(goal['deadline'])
            months_left = max(0, (deadline - pd.Timestamp.now()).days // 30)
            if months_needed > months_left:
                st.error(f"⚠️ At current pace, you'll need {months_needed:.0f} months but only have {months_left} until deadline. Increase monthly contribution to ₹{remaining/max(months_left,1):,.0f}.")
            else:
                st.success(f"✅ On track! You'll reach this goal in {months_needed:.1f} months (deadline: {goal['deadline']})")

    st.markdown("---")
    st.markdown("**📊 All Goals Overview**")
    fig, ax = plt.subplots(figsize=(10, 4))
    goal_names = goals['goal_name'].tolist()
    progress_vals = (goals['saved_so_far'] / goals['target_amount'] * 100).tolist()
    bar_colors = ['#43a047' if p >= 50 else '#fb8c00' if p >= 20 else '#e53935' for p in progress_vals]
    bars = ax.barh(goal_names, progress_vals, color=bar_colors, alpha=0.85)
    ax.axvline(100, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('% Complete')
    ax.set_title('Goal Progress Overview', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, progress_vals):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', va='center', fontsize=9)
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════
# PAGE: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════
elif page == "🔬 What-If Simulator":
    st.markdown('<div class="section-header">🔬 What-If Financial Simulator</div>', unsafe_allow_html=True)
    st.markdown("Adjust sliders to simulate how spending changes affect your savings.")

    all_cats = sorted([c for c in transactions['category'].unique() if c not in ['Rent']])
    changes = {}

    st.markdown("**Reduce spending in these categories:**")
    cols = st.columns(3)
    for i, cat in enumerate(all_cats):
        with cols[i % 3]:
            pct = st.slider(f"Cut {cat} by %", 0, 50, 0, 5)
            if pct > 0:
                changes[cat] = pct

    if st.button("🔮 Run Simulation", type="primary"):
        base_monthly, new_total, base_savings, new_savings, extra = simulate_whatif(transactions, income, changes)

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Monthly Spend", f"₹{base_monthly:,.0f}")
        col2.metric("New Monthly Spend", f"₹{new_total:,.0f}", f"-₹{base_monthly-new_total:,.0f}")
        col3.metric("Current Savings", f"₹{base_savings:,.0f}")
        col4.metric("New Savings", f"₹{new_savings:,.0f}", f"+₹{extra:,.0f}")

        fig, ax = plt.subplots(figsize=(8, 3.5))
        labels = ['Current Spend', 'New Spend', 'Current Savings', 'New Savings']
        values = [base_monthly, new_total, max(0, base_savings), max(0, new_savings)]
        colors = ['#e53935', '#fb8c00', '#1a73e8', '#43a047']
        bars = ax.bar(labels, values, color=colors, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 100, f'₹{val:,.0f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel('₹')
        ax.set_title('Before vs After Simulation', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

        if extra > 0:
            st.success(f"💰 By making these changes, you'd save an extra **₹{extra:,.0f}/month** — that's **₹{extra*12:,.0f}/year**!")
            for goal_name, target, saved in zip(goals['goal_name'], goals['target_amount'], goals['saved_so_far']):
                remaining = target - saved
                months_now = remaining / (base_savings if base_savings > 0 else 1)
                months_new = remaining / (new_savings if new_savings > 0 else 1)
                if months_now > months_new:
                    st.markdown(f'<div class="success-box">🎯 <b>{goal_name}</b>: Reach goal {months_now-months_new:.1f} months sooner!</div>', unsafe_allow_html=True)
    else:
        st.info("Adjust the sliders above and click **Run Simulation** to see results.")


# ═══════════════════════════════════════════
# PAGE: INSIGHTS & TIPS
# ═══════════════════════════════════════════
elif page == "💬 Insights & Tips":
    st.markdown('<div class="section-header">💬 Personalized Insight Generator</div>', unsafe_allow_html=True)

    st.markdown("**🧠 Key Insights About Your Spending**")
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 Actionable Tips**")
    for tip in tips:
        st.markdown(f'<div class="success-box">✅ {tip}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 Lifestyle Pattern Analysis**")
    col1, col2 = st.columns(2)
    with col1:
        food_total = transactions[transactions['category'] == 'Food']['amount'].sum()
        delivery = transactions[transactions['subcategory'] == 'Delivery']['amount'].sum()
        dining = transactions[transactions['subcategory'] == 'Dining Out']['amount'].sum()
        home_food = food_total - delivery - dining

        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ['Delivery', 'Dining Out', 'Home/Groceries']
        vals = [delivery, dining, max(0, home_food)]
        vals = [v for v in vals if v > 0]
        labels = [l for l, v in zip(labels, [delivery, dining, max(0, home_food)]) if v > 0]
        ax.pie(vals, labels=labels, autopct='%1.0f%%', colors=['#e53935', '#fb8c00', '#43a047'], startangle=90)
        ax.set_title('Food Lifestyle Breakdown', fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        monthly = transactions.groupby('month')['amount'].sum()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(monthly.index, monthly.values, color='#1a73e8', alpha=0.7)
        ax.axhline(income, color='#43a047', linestyle='--', label=f'Income ₹{income:,}')
        ax.set_title('Monthly Spend vs Income', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylabel('₹')
        ax.tick_params(axis='x', rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════
# PAGE: RISK REPORT
# ═══════════════════════════════════════════
elif page == "⚠️ Risk Report":
    st.markdown('<div class="section-header">⚠️ Risk Detection System</div>', unsafe_allow_html=True)

    if len(risks) == 0:
        st.success("✅ No significant financial risks detected. Keep it up!")
    else:
        critical = [(s, t, d) for s, t, d in risks if s == "CRITICAL"]
        warnings = [(s, t, d) for s, t, d in risks if s == "WARNING"]

        if critical:
            st.error(f"🔴 {len(critical)} CRITICAL risk(s) found")
            for _, title, detail in critical:
                st.markdown(f'<div class="alert-box">🔴 <b>{title}</b><br>{detail}</div>', unsafe_allow_html=True)

        if warnings:
            st.warning(f"🟡 {len(warnings)} warning(s) found")
            for _, title, detail in warnings:
                st.markdown(f'<div class="alert-box">🟡 <b>{title}</b><br>{detail}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 Risk Radar**")
    categories = ['Budget Control', 'Savings Rate', 'Spending Stability', 'Category Balance', 'Income Coverage']

    monthly_avg = transactions.groupby('month')['amount'].sum().mean()
    total_budget_avg = budget.groupby('month')['budget_amount'].sum().mean() if 'month' in budget.columns else budget['budget_amount'].sum() / 4
    savings_rate = max(0, (income - monthly_avg) / income)
    monthly_cv = transactions.groupby('month')['amount'].std().mean() / monthly_avg if monthly_avg > 0 else 1
    top_cat_share = transactions.groupby('category')['amount'].sum().max() / transactions['amount'].sum()
    income_coverage = min(1, income / monthly_avg)
    budget_adherence = min(1, total_budget_avg / monthly_avg)

    scores_radar = [
        budget_adherence * 10,
        savings_rate * 10,
        max(0, (1 - monthly_cv)) * 10,
        (1 - top_cat_share) * 10,
        income_coverage * 10
    ]

    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores_radar += scores_radar[:1]
    angles += angles[:1]
    ax.plot(angles, scores_radar, 'o-', linewidth=2, color='#1a73e8')
    ax.fill(angles, scores_radar, alpha=0.2, color='#1a73e8')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 10)
    ax.set_title('Financial Risk Radar (10 = best)', fontsize=10, fontweight='bold', pad=15)
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════
# PAGE: RAW DATA
# ═══════════════════════════════════════════
elif page == "📋 Raw Data":
    st.markdown('<div class="section-header">📋 All Datasets</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["💳 Transactions", "📊 Budget", "🎯 Goals", "🏷️ Category Rules"])
    with tab1:
        st.markdown(f"**{len(transactions)} transactions across {transactions['month'].nunique()} months**")
        st.dataframe(transactions.sort_values('date', ascending=False), use_container_width=True)
        st.download_button("⬇️ Download Processed Transactions", transactions.to_csv(index=False), "transactions_processed.csv", "text/csv")
    with tab2:
        st.dataframe(budget, use_container_width=True)
    with tab3:
        st.dataframe(goals, use_container_width=True)
    with tab4:
        st.dataframe(rules, use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.8rem;'>Personal Finance Intelligence System • Built with Python + Streamlit</div>",
    unsafe_allow_html=True
)
