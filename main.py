import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# App Title
st.set_page_config(page_title="Chipata OSR Analysis Dashboard", layout="wide")
st.subheader("Chipata OSR Analysis Dashboard")

# Load specific CSV files for each year and combine them into one DataFrame
data = []
for year in ['2021', '2022', '2023', '2024']:
    filename = f"{year}.csv"
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            df['Year'] = int(year)
            data.append(df)
        except ValueError:
            st.error(f"Error reading {filename}. Please ensure the file has a valid format.")

# Combine all data into a single DataFrame if files are found
if data:
    combined_data = pd.concat(data, ignore_index=True)

    # Display raw data if selected
    if st.checkbox("Show Raw Data"):
        st.write(combined_data)

    # Convert 'Amount' column to numeric and handle errors
    combined_data['Amount'] = pd.to_numeric(combined_data['Amount'], errors='coerce')
    combined_data.dropna(subset=['Amount'], inplace=True)

    # Create columns
    col1, col2 = st.columns(2)

    with col1:
        # Total Revenue from Licenses per Year (Ranked Top-Down)
        total_revenue = combined_data.groupby("Year")["Amount"].sum().sort_values(ascending=False)
        fig_total_revenue = px.bar(total_revenue, x=total_revenue.index, y=total_revenue.values,
                                   labels={'x': 'Year', 'y': 'Total Revenue'},
                                   title="Total Revenue from levies and fees by Year")
        st.plotly_chart(fig_total_revenue)

        # Revenue Trend Analysis by Description (Ranked Top-Down)
        revenue_by_description = combined_data.groupby(['Year', 'Description'])["Amount"].sum().reset_index()
        revenue_by_description = revenue_by_description.sort_values(by="Amount", ascending=False)
        fig_revenue_description = px.bar(revenue_by_description, x='Year', y='Amount', color='Description', barmode='stack',
                                         labels={'Amount': 'Revenue', 'Year': 'Year', 'Description': 'Description'},
                                         title="Revenue Breakdown by fees per Year")
        st.plotly_chart(fig_revenue_description)

    with col2:
        # Total Number of Licenses per Year
        license_count = combined_data.groupby("Year").size()
        fig_license_count = px.bar(license_count, x=license_count.index, y=license_count.values,
                                   labels={'x': 'Year', 'y': 'Total Licenses'},
                                   title="Total Number of Levies and fees by Year")
        st.plotly_chart(fig_license_count)

        # Revenue Trends by Description Over Time
        fig_trend_desc = px.line(combined_data, x="Year", y="Amount", color="Description",
                                 title="Revenue Trends by fees Over Time")
        st.plotly_chart(fig_trend_desc)

    # K-Nearest Neighbors (KNN) Analysis by Description
    st.header("K-Nearest Neighbors Analysis on Licenses, Levies and Fees")
    year_filter = st.selectbox("Select Year for KNN Plot", options=sorted(combined_data['Year'].unique(), reverse=True))
    filtered_data = combined_data[combined_data['Year'] == year_filter]

    label_encoder = LabelEncoder()
    filtered_data['Description_Encoded'] = label_encoder.fit_transform(filtered_data['Description'])

    knn_data = filtered_data[['Amount', 'Description_Encoded']].dropna()
    scaler = StandardScaler()
    knn_scaled = scaler.fit_transform(knn_data)

    pca = PCA(n_components=2)
    knn_pca = pca.fit_transform(knn_scaled)

    k = st.slider("Select Number of Neighbors (K)", min_value=1, max_value=10, value=3)
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(knn_pca, filtered_data['Description_Encoded'])

    knn_fig = px.scatter(x=knn_pca[:, 0], y=knn_pca[:, 1], color=filtered_data['Description'],
                         labels={'x': 'PCA Dimension 1', 'y': 'PCA Dimension 2'},
                         title=f"KNN Clustering of Licenses (Year: {year_filter})")
    knn_fig.update_traces(marker=dict(size=8, opacity=0.6))
    st.plotly_chart(knn_fig)

    # Insight Summary and Analysis
    st.header("Revenue Insights")
    total_revenue_sum = combined_data['Amount'].sum()

    # Top Revenue-Contributing Descriptions
    st.subheader("Top Revenue-Contributing Levies and Fees")
    top_descriptions = combined_data.groupby("Description")["Amount"].sum().sort_values(ascending=False)
    top_descriptions = top_descriptions.reset_index()
    top_descriptions['Percentage'] = (top_descriptions['Amount'] / total_revenue_sum * 100).round(2)
    st.write(top_descriptions)
    

else:
    st.warning("No data found. Please ensure 2021.csv, 2022.csv, 2023.csv, and 2024.csv are in the root directory.")
