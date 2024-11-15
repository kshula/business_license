import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# App Title
st.title("Business License Analysis Dashboard")
st.subheader("Analyze business license data over multiple years with insights by Description")

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
        # Total Number of Licenses per Year
        license_count = combined_data.groupby("Year").size()
        fig_license_count = px.bar(license_count, x=license_count.index, y=license_count.values,
                                   labels={'x': 'Year', 'y': 'Total Licenses'},
                                   title="Total Number of Licenses by Year")
        st.plotly_chart(fig_license_count)

        # Revenue Breakdown by Description per Year
        revenue_by_description = combined_data.groupby(['Year', 'Description'])["Amount"].sum().reset_index()
        fig_revenue_description = px.bar(revenue_by_description, x='Year', y='Amount', color='Description', barmode='stack',
                                         labels={'Amount': 'Revenue', 'Year': 'Year', 'Description': 'Description'},
                                         title="Revenue Breakdown by Description per Year")
        st.plotly_chart(fig_revenue_description)

    with col2:
        # Total Revenue from Licenses per Year
        total_revenue = combined_data.groupby("Year")["Amount"].sum()
        fig_total_revenue = px.bar(total_revenue, x=total_revenue.index, y=total_revenue.values,
                                   labels={'x': 'Year', 'y': 'Total Revenue'},
                                   title="Total Revenue from Licenses by Year")
        st.plotly_chart(fig_total_revenue)

        # Revenue Trend Analysis by Description
        fig_trend_desc = px.line(combined_data, x="Year", y="Amount", color="Description",
                                 title="Revenue Trends by Description Over Time")
        st.plotly_chart(fig_trend_desc)

    # K-Nearest Neighbors (KNN) Analysis by Description
    st.header("K-Nearest Neighbors Analysis on Licenses by Description")
    
    label_encoder = LabelEncoder()
    combined_data['Description_Encoded'] = label_encoder.fit_transform(combined_data['Description'])

    knn_data = combined_data[['Amount', 'Description_Encoded']].dropna()
    scaler = StandardScaler()
    knn_scaled = scaler.fit_transform(knn_data)

    pca = PCA(n_components=2)
    knn_pca = pca.fit_transform(knn_scaled)

    k = st.slider("Select Number of Neighbors (K)", min_value=1, max_value=10, value=3)
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(knn_pca, combined_data['Description_Encoded'])

    knn_fig = px.scatter(x=knn_pca[:, 0], y=knn_pca[:, 1], color=combined_data['Description'],
                         labels={'x': 'PCA Dimension 1', 'y': 'PCA Dimension 2'},
                         title="KNN Clustering of Licenses (by Description)")
    knn_fig.update_traces(marker=dict(size=8, opacity=0.6))
    st.plotly_chart(knn_fig)

    # Insight Summary and Analysis
    col3, col4 = st.columns(2)
    
    with col3:
        st.header("Summary of Insights")
        st.write("The KNN analysis groups licenses by their `Amount` and `Description` similarity.")
        st.write("The visualizations and trend analyses provide insights into yearly revenue types by description.")
        
        st.header("Top Revenue-Contributing Descriptions")
        top_descriptions = combined_data.groupby("Description")["Amount"].sum().sort_values(ascending=False)
        st.write("Top Revenue-Contributing Descriptions:", top_descriptions.head(10))
        
        st.header("Pareto Analysis (80/20 Rule)")
        cumulative_revenue = top_descriptions.cumsum() / top_descriptions.sum()
        pareto_threshold = cumulative_revenue[cumulative_revenue <= 0.8]
        top_20_percent = pareto_threshold.index
        st.write(f"The top {len(top_20_percent)} descriptions contribute to 80% of total revenue.")

    with col4:
        st.header("Revenue Distribution Analysis")
        percentiles = combined_data['Amount'].quantile([0.25, 0.5, 0.75, 0.9])
        st.write("Revenue Percentiles:", percentiles)

        st.header("Revenue Segmentation by Description")
        combined_data['Revenue_Segment'] = pd.qcut(combined_data['Amount'], q=3, labels=['Low', 'Medium', 'High'])
        revenue_segment_summary = combined_data.groupby(['Year', 'Revenue_Segment'])['Amount'].sum().unstack()
        fig_revenue_segment = px.bar(revenue_segment_summary, title="Revenue Segmentation Over Time by Description")
        st.plotly_chart(fig_revenue_segment)

    # Download button for combined data
    st.header("Download Combined Data")
    csv_data = combined_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Combined Data as CSV", data=csv_data, file_name="combined_data.csv", mime="text/csv")

else:
    st.warning("No data found. Please ensure 2021.csv, 2022.csv, 2023.csv, and 2024.csv are in the root directory.")
