import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Page config
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecasting Dashboard")
st.write("Upload your sales dataset to analyze trends and predict sales.")

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Data Analysis", "Sales Forecast"]
)

uploaded_file = st.file_uploader("Upload Sales Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Feature Engineering
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    df['lag_1'] = df['Sales'].shift(1)
    df['lag_7'] = df['Sales'].shift(7)
    df['lag_30'] = df['Sales'].shift(30)

    df['rolling_mean_7'] = df['Sales'].rolling(7).mean()
    df['rolling_mean_30'] = df['Sales'].rolling(30).mean()

    df = df.dropna()

    # Sidebar Filters
    if "Category" in df.columns:

        categories = st.sidebar.multiselect(
            "Filter by Category",
            df["Category"].unique()
        )

        if categories:
            df = df[df["Category"].isin(categories)]

    # DASHBOARD PAGE
    if page == "Dashboard":

        st.subheader("📈 Key Metrics")

        total_sales = df["Sales"].sum()
        avg_sales = df["Sales"].mean()
        max_sales = df["Sales"].max()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Sales", f"${total_sales:,.2f}")
        col2.metric("Average Sales", f"${avg_sales:,.2f}")
        col3.metric("Highest Sale", f"${max_sales:,.2f}")

        st.subheader("Sales Trend")

        daily_sales = df.groupby("Date")["Sales"].sum().reset_index()

        fig = px.line(
            daily_sales,
            x="Date",
            y="Sales",
            title="Daily Sales Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    # DATA ANALYSIS PAGE
    elif page == "Data Analysis":

        st.subheader("Sales Distribution")

        fig = px.histogram(
            df,
            x="Sales",
            nbins=50,
            title="Sales Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Sales")

        monthly_sales = df.groupby("month")["Sales"].sum().reset_index()

        fig2 = px.bar(
            monthly_sales,
            x="month",
            y="Sales",
            title="Sales by Month"
        )

        st.plotly_chart(fig2, use_container_width=True)

        if "Category" in df.columns:

            st.subheader("Sales by Category")

            category_sales = df.groupby("Category")["Sales"].sum().reset_index()

            fig3 = px.bar(
                category_sales,
                x="Category",
                y="Sales",
                title="Category Performance"
            )

            st.plotly_chart(fig3, use_container_width=True)

    # FORECAST PAGE
    elif page == "Sales Forecast":

        st.subheader("📊 Sales Forecast")

        # Load model
        model = pickle.load(open("sales_model.pkl", "rb"))

        features = df[['dayofweek','month','is_weekend','lag_1','lag_7','lag_30','rolling_mean_7','rolling_mean_30']]

        predictions = model.predict(features)

        df["Predicted"] = predictions

        st.subheader("Actual vs Predicted Sales")

        fig4 = px.line(
            df,
            x="Date",
            y=["Sales", "Predicted"],
            title="Actual vs Predicted Sales"
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Prediction Table")

        st.dataframe(df[["Date","Sales","Predicted"]].tail(20))

        # Download forecast
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Forecast Report",
            data=csv,
            file_name="sales_forecast.csv",
            mime="text/csv"
        )

else:

    st.info("Please upload a CSV dataset to start analysis.")
