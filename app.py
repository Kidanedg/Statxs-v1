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
import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="StatX Statistical WebLab Software",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CUSTOM STYLE
# =====================================================

st.markdown("""
<style>

.main-title{
    font-size:64px;
    font-weight:900;
    text-align:center;
    color:#cc0000;
    letter-spacing:2px;
}

.version{
    text-align:center;
    font-size:16px;
    color:gray;
}

.subtitle{
    font-size:30px;
    text-align:center;
    color:black;
    font-weight:700;
}

.description{
    text-align:center;
    font-size:19px;
    color:#444;
    padding-top:10px;
    line-height:1.6;
}

.divider{
    height:4px;
    background-color:#cc0000;
    border-radius:5px;
    margin-top:15px;
    margin-bottom:30px;
}

.footer{
    text-align:center;
    font-size:14px;
    color:gray;
    margin-top:80px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# FRONT PAGE HEADER
# =====================================================

st.markdown('<div class="main-title">StatX</div>', unsafe_allow_html=True)

st.markdown('<div class="version">Version 1.0</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">StatX Statistical WebLab Software</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown(
"""
<div class="description">

A modern scientific environment for  
<b>statistical computing, data analysis, and data visualization.</b>

Upload a dataset to perform statistical analysis with  
<b>automatic interpretation and interactive graphics.</b>

</div>
""",
unsafe_allow_html=True
)


# =====================================================
# COPYRIGHT
# =====================================================

st.markdown(
"""
<div class="footer">
© 2024–2026 <b>Dr. Kidane Desta</b> — Founder of <b>StatX Software</b><br>
All Rights Reserved.
</div>
""",
unsafe_allow_html=True
)
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

    st.subheader("📊 Data Visualization")

    plot = st.selectbox(
        "Plot Type",
        [
            "Histogram",
            "Density Plot",
            "Boxplot",
            "Violin Plot",
            "Scatter Plot",
            "Scatter with Regression",
            "Bar Chart",
            "Line Chart",
            "Count Plot",
            "Pair Plot",
            "Correlation Heatmap"
        ]
    )

    # -----------------------------
    # HISTOGRAM
    # -----------------------------

    if plot == "Histogram":

        var = st.selectbox("Variable", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[var], kde=True)
        st.pyplot(fig)

    # -----------------------------
    # DENSITY PLOT
    # -----------------------------

    elif plot == "Density Plot":

        var = st.selectbox("Variable", numeric_cols)

        fig, ax = plt.subplots()
        sns.kdeplot(df[var], fill=True)
        st.pyplot(fig)

    # -----------------------------
    # BOXPLOT
    # -----------------------------

    elif plot == "Boxplot":

        var = st.selectbox("Variable", numeric_cols)

        fig, ax = plt.subplots()
        sns.boxplot(y=df[var])
        st.pyplot(fig)

    # -----------------------------
    # VIOLIN PLOT
    # -----------------------------

    elif plot == "Violin Plot":

        var = st.selectbox("Variable", numeric_cols)

        fig, ax = plt.subplots()
        sns.violinplot(y=df[var])
        st.pyplot(fig)

    # -----------------------------
    # SCATTER
    # -----------------------------

    elif plot == "Scatter Plot":

        x = st.selectbox("X Variable", numeric_cols)
        y = st.selectbox("Y Variable", numeric_cols)

        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x], y=df[y])
        st.pyplot(fig)

    # -----------------------------
    # SCATTER + REGRESSION
    # -----------------------------

    elif plot == "Scatter with Regression":

        x = st.selectbox("X Variable", numeric_cols)
        y = st.selectbox("Y Variable", numeric_cols)

        fig, ax = plt.subplots()
        sns.regplot(x=df[x], y=df[y])
        st.pyplot(fig)

    # -----------------------------
    # BAR CHART
    # -----------------------------

    elif plot == "Bar Chart":

        cat_cols = df.select_dtypes(include=["object","category"]).columns

        if len(cat_cols) > 0:

            cat = st.selectbox("Category", cat_cols)

            fig, ax = plt.subplots()
            df[cat].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        else:
            st.warning("No categorical variables found")

    # -----------------------------
    # LINE CHART
    # -----------------------------

    elif plot == "Line Chart":

        var = st.selectbox("Variable", numeric_cols)

        fig, ax = plt.subplots()
        ax.plot(df[var])
        ax.set_title("Line Plot")
        st.pyplot(fig)

    # -----------------------------
    # COUNT PLOT
    # -----------------------------

    elif plot == "Count Plot":

        cat_cols = df.select_dtypes(include=["object","category"]).columns

        if len(cat_cols) > 0:

            var = st.selectbox("Variable", cat_cols)

            fig, ax = plt.subplots()
            sns.countplot(x=df[var])
            plt.xticks(rotation=45)
            st.pyplot(fig)

        else:
            st.warning("No categorical variables found")

    # -----------------------------
    # PAIR PLOT
    # -----------------------------

    elif plot == "Pair Plot":

        st.write("Pairwise relationships between numeric variables")

        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig)

    # -----------------------------
    # CORRELATION HEATMAP
    # -----------------------------

    elif plot == "Correlation Heatmap":

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(fig)

# =====================================================
# HYPOTHESIS TESTING & ESTIMATION
# =====================================================

elif analysis_category == "Hypothesis Testing & Estimation":

    st.subheader("📊 Hypothesis Testing and Estimation")

    test_type = st.selectbox(
        "Select Analysis",
        [
            "One-Sample t-Test",
            "Two-Sample t-Test",
            "Paired t-Test",
            "Proportion Test",
            "Chi-Square Test",
            "One-Way ANOVA",
            "Confidence Interval (Mean)",
            "Confidence Interval (Proportion)"
        ]
    )

# =====================================================
# ONE SAMPLE T TEST
# =====================================================

    if test_type == "One-Sample t-Test":

        var = st.selectbox("Variable", numeric_cols)

        mu = st.number_input("Hypothesized Mean", value=0.0)

        alpha = st.slider("Significance Level", 0.01,0.10,0.05)

        data = df[var].dropna()

        stat, p = stats.ttest_1samp(data, mu)

        st.write("### Results")

        st.write("t statistic:", stat)
        st.write("p-value:", p)

        if p < alpha:
            st.success("Reject H₀: The sample mean is significantly different from the hypothesized mean.")
        else:
            st.info("Fail to reject H₀: No significant difference detected.")

# =====================================================
# TWO SAMPLE T TEST
# =====================================================

    elif test_type == "Two-Sample t-Test":

        var = st.selectbox("Numeric Variable", numeric_cols)

        cat_cols = df.select_dtypes(include=["object","category"]).columns

        group = st.selectbox("Grouping Variable", cat_cols)

        groups = df[group].unique()

        g1 = st.selectbox("Group 1", groups)
        g2 = st.selectbox("Group 2", groups)

        alpha = st.slider("Significance Level",0.01,0.10,0.05)

        data1 = df[df[group]==g1][var].dropna()
        data2 = df[df[group]==g2][var].dropna()

        stat,p = stats.ttest_ind(data1,data2)

        st.write("### Results")
        st.write("t statistic:", stat)
        st.write("p-value:", p)

        if p < alpha:
            st.success("Reject H₀: The group means are significantly different.")
        else:
            st.info("Fail to reject H₀: No significant difference between groups.")

# =====================================================
# PAIRED T TEST
# =====================================================

    elif test_type == "Paired t-Test":

        var1 = st.selectbox("Variable 1", numeric_cols)
        var2 = st.selectbox("Variable 2", numeric_cols)

        alpha = st.slider("Significance Level",0.01,0.10,0.05)

        data1 = df[var1].dropna()
        data2 = df[var2].dropna()

        stat,p = stats.ttest_rel(data1,data2)

        st.write("### Results")

        st.write("t statistic:", stat)
        st.write("p-value:", p)

        if p < alpha:
            st.success("Reject H₀: Significant difference between paired samples.")
        else:
            st.info("Fail to reject H₀: No significant difference detected.")

# =====================================================
# PROPORTION TEST
# =====================================================

    elif test_type == "Proportion Test":

        successes = st.number_input("Number of Successes", value=10)
        n = st.number_input("Sample Size", value=50)

        p0 = st.number_input("Hypothesized Proportion", value=0.5)

        alpha = st.slider("Significance Level",0.01,0.10,0.05)

        stat,p = stats.binom_test(successes,n,p0)

        st.write("p-value:",p)

        if p < alpha:
            st.success("Reject H₀: Proportion differs from hypothesized value.")
        else:
            st.info("Fail to reject H₀.")

# =====================================================
# CHI SQUARE TEST
# =====================================================

    elif test_type == "Chi-Square Test":

        cat_cols = df.select_dtypes(include=["object","category"]).columns

        var1 = st.selectbox("Variable 1", cat_cols)
        var2 = st.selectbox("Variable 2", cat_cols)

        table = pd.crosstab(df[var1],df[var2])

        stat,p,_,_ = stats.chi2_contingency(table)

        st.write("Contingency Table")
        st.dataframe(table)

        st.write("Chi-square statistic:", stat)
        st.write("p-value:", p)

        if p < 0.05:
            st.success("Reject H₀: Variables are associated.")
        else:
            st.info("Fail to reject H₀: No significant association.")

# =====================================================
# ANOVA
# =====================================================

    elif test_type == "One-Way ANOVA":

        var = st.selectbox("Numeric Variable", numeric_cols)

        cat_cols = df.select_dtypes(include=["object","category"]).columns
        group = st.selectbox("Grouping Variable", cat_cols)

        groups = df[group].unique()

        data = [df[df[group]==g][var].dropna() for g in groups]

        stat,p = stats.f_oneway(*data)

        st.write("F statistic:", stat)
        st.write("p-value:", p)

        if p < 0.05:
            st.success("Reject H₀: At least one group mean differs.")
        else:
            st.info("Fail to reject H₀.")

# =====================================================
# CONFIDENCE INTERVAL MEAN
# =====================================================

    elif test_type == "Confidence Interval (Mean)":

        var = st.selectbox("Variable", numeric_cols)

        conf = st.slider("Confidence Level",0.80,0.99,0.95)

        data = df[var].dropna()

        mean = np.mean(data)
        sem = stats.sem(data)

        interval = stats.t.interval(conf,len(data)-1,loc=mean,scale=sem)

        st.write("Sample Mean:",mean)
        st.write("Confidence Interval:",interval)

# =====================================================
# CONFIDENCE INTERVAL PROPORTION
# =====================================================

    elif test_type == "Confidence Interval (Proportion)":

        successes = st.number_input("Successes",10)
        n = st.number_input("Sample Size",50)

        conf = st.slider("Confidence Level",0.80,0.99,0.95)

        p = successes/n

        z = stats.norm.ppf(1-(1-conf)/2)

        se = np.sqrt(p*(1-p)/n)

        lower = p - z*se
        upper = p + z*se

        st.write("Estimated Proportion:",p)
        st.write("Confidence Interval:",(lower,upper))    

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
