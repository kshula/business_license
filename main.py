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

    # Dropdown for selecting the year
    selected_year = st.selectbox("Select Year to View Revenue Leading Descriptions", 
                                 options=sorted(combined_data['Year'].unique()))

    # Filter data based on the selected year
    filtered_data = combined_data[combined_data['Year'] == selected_year]

    # Display top revenue-contributing descriptions for the selected year
    st.subheader(f"Top Revenue-Contributing Descriptions for {selected_year}")
    revenue_by_description = (filtered_data.groupby("Description")["Amount"]
                               .sum()
                               .sort_values(ascending=False)
                               .reset_index())
    revenue_by_description['Percentage'] = (revenue_by_description['Amount'] / 
                                            revenue_by_description['Amount'].sum() * 100).round(2)
    st.write(revenue_by_description)

    # Create columns for visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart of revenue breakdown by description for the selected year
        fig_revenue_desc = px.bar(revenue_by_description, 
                                  x="Description", 
                                  y="Amount", 
                                  title=f"Revenue Breakdown by Description for {selected_year}",
                                  labels={"Amount": "Revenue", "Description": "Description"})
        st.plotly_chart(fig_revenue_desc)

    with col2:
        # Pie chart of revenue percentage distribution by description
        fig_pie = px.pie(revenue_by_description, 
                         names="Description", 
                         values="Amount", 
                         title=f"Revenue Percentage Distribution by Description for {selected_year}")
        st.plotly_chart(fig_pie)

    # Insight Summary and Analysis
    st.header("Revenue Insights")
    total_revenue = combined_data['Amount'].sum()
    st.write(f"**Total Revenue Across All Years:** {total_revenue}")

    # Overall top revenue-contributing descriptions
    st.subheader("Overall Top Revenue-Contributing Descriptions")
    overall_revenue_by_desc = (combined_data.groupby("Description")["Amount"]
                               .sum()
                               .sort_values(ascending=False)
                               .reset_index())
    overall_revenue_by_desc['Percentage'] = (overall_revenue_by_desc['Amount'] / 
                                             combined_data['Amount'].sum() * 100).round(2)
    st.write(overall_revenue_by_desc)

else:
    st.warning("No data found. Please ensure 2021.csv, 2022.csv, 2023.csv, and 2024.csv are in the root directory.")
