import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="StatX v1", layout="wide")

st.title("StatX v1 - Statistical Analysis Platform")

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

df = None

# -----------------------------
# DATA LOADING
# -----------------------------

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.success("Dataset Loaded Successfully!")

    except Exception as e:
        st.error(f"Error loading file: {e}")

# -----------------------------
# DATA PREVIEW
# -----------------------------

if df is not None:

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Column Names")
    st.write(df.columns)

    # -----------------------------
    # DESCRIPTIVE STATISTICS
    # -----------------------------

    st.subheader("Descriptive Statistics")

    if st.checkbox("Show Statistics"):

        st.write(df.describe())

    # -----------------------------
    # VARIABLE SELECTION
    # -----------------------------

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_columns:

        column = st.selectbox(
            "Select Numeric Variable",
            numeric_columns
        )

        # Histogram

        st.subheader("Histogram")

        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)

        st.pyplot(fig)

        # Boxplot

        st.subheader("Boxplot")

        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[column], ax=ax2)

        st.pyplot(fig2)

    else:

        st.warning("No numeric columns found in dataset")

else:

    st.info("Upload a dataset to begin analysis")
