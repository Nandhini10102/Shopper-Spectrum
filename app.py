import streamlit as st
import pandas as pd
import numpy as np
import joblib
import difflib

# -----------------------------
# Load Data & Models
# -----------------------------
df = pd.read_csv("online_retail.csv", encoding="latin1")

# Load clustering model & scaler
kmeans = joblib.load("kmeans_rfm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load item similarity matrix
item_similarity_df = pd.read_csv("item_similarity_matrix.csv", index_col=0)

# Ensure StockCodes are strings everywhere
df["StockCode"] = df["StockCode"].astype(str)
item_similarity_df.columns = item_similarity_df.columns.astype(str)
item_similarity_df.index = item_similarity_df.index.astype(str)

# Create StockCode <-> Description mapping
product_mapping = df.groupby("StockCode")["Description"].first().dropna().to_dict()
reverse_mapping = {v.upper(): k for k, v in product_mapping.items()}


# -----------------------------
# Helper Functions
# -----------------------------

def predict_cluster(recency, frequency, monetary):
    rfm_scaled = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(rfm_scaled)[0]
    return ["High-Value", "Regular", "Occasional", "At-Risk"][cluster]


def recommend_products(product_name, top_n=5):
    """Recommend top-N similar products"""
    try:
        possible_matches = list(reverse_mapping.keys())
        closest_match = difflib.get_close_matches(product_name.upper(), possible_matches, n=1, cutoff=0.6)

        if not closest_match:
            return ["❌ Product not found in database!"]

        matched_name = closest_match[0]
        stock_code = reverse_mapping[matched_name]

        if stock_code not in item_similarity_df.columns:
            return [f"⚠️ No similarity data available for {matched_name}"]

        sim_scores = item_similarity_df[stock_code].sort_values(ascending=False)[1:top_n+1]
        recommended = [product_mapping.get(str(code), str(code)) for code in sim_scores.index]
        return recommended

    except Exception as e:
        return [f"⚠️ Error: {e}"]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

st.sidebar.title("🛒 Shopper Spectrum")
menu = ["Home", "Clustering", "Recommendation"]
choice = st.sidebar.radio("Navigate", menu)

# -----------------------------
# Home
# -----------------------------
if choice == "Home":
    st.title("📊 Shopper Spectrum Dashboard")
    st.write("""
        Welcome to **Shopper Spectrum** – an E-commerce Customer Segmentation 
        and Product Recommendation System.
        
        ### Features:
        - 🎯 Customer Segmentation using RFM + KMeans
        - 🛍️ Product Recommendations using Collaborative Filtering
    """)

# -----------------------------
# Clustering
# -----------------------------
elif choice == "Clustering":
    st.title("🔍 Customer Segmentation Module")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=10.0)

    if st.button("Predict Cluster"):
        if recency > 0 or frequency > 0 or monetary > 0:
            cluster_label = predict_cluster(recency, frequency, monetary)
            st.success(f"🎯 Predicted Customer Segment: **{cluster_label}**")
        else:
            st.warning("⚠️ Please enter valid RFM values!")

# -----------------------------
# Recommendation
# -----------------------------
elif choice == "Recommendation":
    st.title("🎯 Product Recommender")
    st.write("Select a product name and get **Top 5 Recommendations.**")

    # 👉 Show ALL products in dropdown (not filtered)
    product_list = sorted(list(set(product_mapping.values())))
    product_name = st.selectbox("🔎 Choose Product", product_list)

    if st.button("Recommend"):
        if product_name:
            recommendations = recommend_products(product_name)
            st.subheader("✅ Recommended Products:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.warning("⚠️ Please select a product!")




