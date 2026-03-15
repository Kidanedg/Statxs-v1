# =====================================================
# StatX v1 – Statistical Analysis Platform
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="StatX Statistical WebLab",
    layout="wide",
    page_icon="📊"
)

# =====================================================
# CUSTOM STYLE
# =====================================================

st.markdown("""
<style>

.main-title{
    font-size:60px;
    font-weight:800;
    text-align:center;
    color:#cc0000;
}

.subtitle{
    font-size:28px;
    text-align:center;
    color:black;
    font-weight:600;
}

.description{
    text-align:center;
    font-size:20px;
    color:#444;
    padding-top:10px;
}

.divider{
    height:4px;
    background-color:#cc0000;
    border-radius:5px;
    margin-top:10px;
    margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# FRONT PAGE
# =====================================================

st.markdown('<div class="main-title">StatX</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">StatX Statistical WebLab Software</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="description">

    A scientific statistical computing platform designed for  
    <b>data analysis, modeling, and intelligent statistical discovery.</b>

    Upload a dataset and perform statistical analysis with  
    <b>automatic interpretation and visualization.</b>

    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# =====================================================
# DATA UPLOAD SYSTEM
# =====================================================

st.sidebar.markdown("## 📂 Dataset Manager")

uploaded = st.sidebar.file_uploader(
    "Upload Dataset",
    type=[
        "csv", "xlsx", "xls",
        "txt", "json",
        "parquet",
        "dta", "sav"
    ]
)

df = None

# =====================================================
# LOAD DATA
# =====================================================

if uploaded:

    try:

        file_name = uploaded.name.lower()

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded)

        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)

        elif file_name.endswith(".txt"):
            df = pd.read_csv(uploaded, sep=None, engine="python")

        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded)

        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)

        elif file_name.endswith(".dta"):
            df = pd.read_stata(uploaded)

        elif file_name.endswith(".sav"):
            df = pd.read_spss(uploaded)

        st.sidebar.success("✅ Dataset Loaded")

    except Exception as e:
        st.sidebar.error(f"❌ Error loading dataset: {e}")


# =====================================================
# DATASET DISPLAY PAGE
# =====================================================

if df is not None:

    st.markdown("## 📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())

    st.divider()

    # Preview

    st.markdown("### 🔎 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Column Types

    st.markdown("### 📑 Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum().values
    })

    st.dataframe(info_df, use_container_width=True)

else:

    st.markdown(
        """
        # 📈 Welcome to **StatX v1**

        Upload a dataset from the sidebar to begin statistical analysis.

        ### Supported Formats

        - CSV
        - Excel
        - TXT
        - JSON
        - Parquet
        - SPSS (.sav)
        - Stata (.dta)
        """
    )

# =====================================================
# INTERPRETATION FUNCTIONS
# =====================================================

def interpret_p(p):

    if p < 0.01:
        return "Strong evidence against the null hypothesis (p < 0.01). Reject H0."
    elif p < 0.05:
        return "Statistically significant at 5% level (p < 0.05). Reject H0."
    else:
        return "Not statistically significant (p ≥ 0.05). Fail to reject H0."


def interpret_r2(r2):

    if r2 > 0.75:
        return "Very strong model fit."
    elif r2 > 0.50:
        return "Moderate model fit."
    elif r2 > 0.25:
        return "Weak model fit."
    else:
        return "Very weak model fit."

# =====================================================
# MAIN ANALYSIS
# =====================================================

if df is not None:

    st.subheader("Dataset Preview")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    # ============================================
    # SIDEBAR MENU
    # ============================================

    analysis_category = st.sidebar.selectbox(
        "Analysis Category",
        [
            "Descriptive Statistics",
            "Graphics",
            "Hypothesis Tests",
            "ANOVA",
            "Regression",
            "Chi-Square Tests",
            "Time Series",
            "Quality Control"
        ]
    )

# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

    if analysis_category == "Descriptive Statistics":

        st.subheader("Descriptive Statistics")

        var = st.selectbox("Variable", numeric_cols)

        data = df[var]

        summary = pd.DataFrame({
            "Mean":[np.mean(data)],
            "Median":[np.median(data)],
            "Std Dev":[np.std(data)],
            "Variance":[np.var(data)],
            "Min":[np.min(data)],
            "Max":[np.max(data)],
            "Skewness":[stats.skew(data)],
            "Kurtosis":[stats.kurtosis(data)]
        })

        st.table(summary)

# =====================================================
# GRAPHICS
# =====================================================

    elif analysis_category == "Graphics":

        st.subheader("Visualization")

        plot = st.selectbox(
            "Plot Type",
            ["Histogram","Boxplot","Scatter","Correlation Heatmap"]
        )

        if plot == "Histogram":

            var = st.selectbox("Variable", numeric_cols)

            fig, ax = plt.subplots()
            sns.histplot(df[var], kde=True)
            st.pyplot(fig)

        elif plot == "Boxplot":

            var = st.selectbox("Variable", numeric_cols)

            fig, ax = plt.subplots()
            sns.boxplot(y=df[var])
            st.pyplot(fig)

# =====================================================
# TIME SERIES
# =====================================================

    elif analysis_category == "Time Series":

        st.subheader("Time Series Analysis")

        method = st.selectbox(
            "Method",
            ["Time Series Plot","Moving Average","Trend Estimation"]
        )

        var = st.selectbox("Variable", numeric_cols)

        data = df[var]

        if method == "Time Series Plot":

            fig, ax = plt.subplots()
            ax.plot(data)
            st.pyplot(fig)

        elif method == "Moving Average":

            window = st.slider("Window",2,20,5)

            ma = data.rolling(window).mean()

            fig, ax = plt.subplots()
            ax.plot(data,label="Original")
            ax.plot(ma,label="Moving Average")
            ax.legend()

            st.pyplot(fig)

# =====================================================
# QUALITY CONTROL
# =====================================================

    elif analysis_category == "Quality Control":

        st.subheader("Statistical Quality Control")

        method = st.selectbox(
            "Method",
            ["Control Chart","Process Capability"]
        )

        var = st.selectbox("Process Variable", numeric_cols)

        data = df[var]

        if method == "Control Chart":

            mean = np.mean(data)
            std = np.std(data)

            UCL = mean + 3*std
            LCL = mean - 3*std

            fig, ax = plt.subplots()

            ax.plot(data,marker="o")
            ax.axhline(mean,label="Mean")
            ax.axhline(UCL,color="red",label="UCL")
            ax.axhline(LCL,color="red",label="LCL")

            ax.legend()

            st.pyplot(fig)

        elif method == "Process Capability":

            LSL = st.number_input("Lower Spec Limit")
            USL = st.number_input("Upper Spec Limit")

            mean = np.mean(data)
            std = np.std(data)

            Cp = (USL-LSL)/(6*std)

            st.write("Cp:",Cp)

else:

    st.info("Upload a dataset to begin analysis.")
