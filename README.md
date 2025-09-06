🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce
📖 Project Overview

This project analyzes e-commerce transaction data to understand customer behavior, segment customers based on their purchase patterns, and provide personalized product recommendations.
It demonstrates the end-to-end workflow: from data cleaning and exploratory analysis to machine learning and web app deployment.

🎯 Problem Statement

The global e-commerce industry generates massive transaction data daily.

How can we segment customers to target marketing campaigns effectively?

How can we recommend relevant products to improve customer experience and increase sales?

This project uses RFM analysis, clustering, and collaborative filtering to answer these questions.

🧠 Problem Type

Unsupervised Learning: Customer Segmentation (Clustering)

Recommendation System: Product Recommendations using Item-based Collaborative Filtering

📊 Dataset

Source: Public Online Retail Dataset

Columns:

Column	Description
InvoiceNo	Transaction number
StockCode	Unique product/item code
Description	Product name
Quantity	Number of products purchased
InvoiceDate	Date and time of transaction
UnitPrice	Price per product
CustomerID	Unique identifier for each customer
Country	Country of the customer
🛠 Technical Skills

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering (RFM)

Clustering Techniques (KMeans)

Collaborative Filtering (Cosine Similarity)

Model Evaluation & Interpretation

Streamlit App Deployment

🛠 Methodology
1️⃣ Data Cleaning & Preprocessing

Remove missing CustomerID rows

Exclude cancelled invoices

Remove negative or zero quantity/price

2️⃣ Exploratory Data Analysis

Analyze top-selling products, purchase trends, and country-wise transactions

Visualize RFM distributions

3️⃣ Customer Segmentation (Clustering)

Feature Engineering: Recency, Frequency, Monetary

Standardization and KMeans clustering

Segments: High-Value, Regular, Occasional, At-Risk

4️⃣ Recommendation System

Customer–Product matrix

Item-based Collaborative Filtering using Cosine Similarity

Recommend top 5 similar products

5️⃣ Streamlit Web App

Product Recommendation Module: Enter product → get 5 similar products

Customer Segmentation Module: Enter RFM → predict customer segment

📌 Real-World Use Cases

Targeted Marketing Campaigns

Personalized Product Recommendations

Retention Programs for At-Risk Customers

Dynamic Pricing Strategies

Inventory Management Optimization

💻 Tech Stack

Python, Pandas, NumPy

Scikit-Learn, Matplotlib, Seaborn

Streamlit for web app deployment
