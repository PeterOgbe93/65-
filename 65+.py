import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import fpgrowth, association_rules

st.title("🛒 Online Retail Insights Dashboard")

st.write("Designed for easy understanding of shopping patterns")

# Load data
df = pd.read_csv("your_data.csv")

# Basket encoding
basket = df.groupby(['TransactionID', 'Item'])['Item'].count().unstack().fillna(0)
basket = basket > 0

# Sidebar controls (simple for seniors)
st.sidebar.header("Controls")
min_support = st.sidebar.slider("Minimum Support", 0.001, 0.1, 0.01)

# -------------------------
# 1. Top products
# -------------------------
st.header("Top Purchased Items")

top_items = df['Item'].value_counts().head(10)

fig1 = px.bar(
    top_items,
    orientation='h',
    title="Most Popular Items"
)
st.plotly_chart(fig1)

# -------------------------
# 2. Transaction size
# -------------------------
st.header("Shopping Basket Size")

fig2 = px.histogram(
    basket.sum(axis=1),
    title="Items per Shopping Trip"
)
st.plotly_chart(fig2)

# -------------------------
# 3. ML section
# -------------------------
st.header("Patterns Found by Machine Learning")

freq = fpgrowth(basket, min_support=min_support, use_colnames=True)

st.write("Frequent item combinations found:")

st.dataframe(freq.sort_values("support", ascending=False).head(10))

rules = association_rules(freq, metric="lift", min_threshold=1)

st.write("Strong buying relationships:")
st.dataframe(rules.sort_values("lift", ascending=False).head(10))

streamlit run dashboard.py