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

    # -------------------------------------------
    # DATA PREVIEW
    # -------------------------------------------

    st.markdown("### 🔎 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # -------------------------------------------
    # COLUMN INFORMATION
    # -------------------------------------------

    st.markdown("### 📑 Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum().values
    })

    st.dataframe(info_df, use_container_width=True)

    st.divider()

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
        return "Strong evidence against the null hypothesis (p < 0.01). Reject H₀."
    elif p < 0.05:
        return "Statistically significant at 5% level (p < 0.05). Reject H₀."
    else:
        return "Not statistically significant (p ≥ 0.05). Fail to reject H₀."


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

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    st.sidebar.markdown("## 📊 Analysis Menu")

    analysis_category = st.sidebar.selectbox(
        "Select Analysis",
        [
            "Descriptive Statistics",
            "Graphics",
            "Hypothesis Testing",
            "Regression",
            "Chi-Square Tests"
        ]
    )

# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

    if analysis_category == "Descriptive Statistics":

        st.subheader("📑 Descriptive Statistics")

        var = st.selectbox("Select Variable", numeric_cols)

        data = df[var].dropna()

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
                "Boxplot",
                "Scatter Plot",
                "Correlation Heatmap"
            ]
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

        elif plot == "Scatter Plot":

            x = st.selectbox("X Variable", numeric_cols)
            y = st.selectbox("Y Variable", numeric_cols)

            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x], y=df[y])
            st.pyplot(fig)

        elif plot == "Correlation Heatmap":

            corr = df[numeric_cols].corr()

            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm")

            st.pyplot(fig)


# =====================================================
# HYPOTHESIS TESTING
# =====================================================

    elif analysis_category == "Hypothesis Testing":

        st.subheader("📊 Hypothesis Testing")

        test = st.selectbox(
            "Select Test",
            [
                "One-Sample t-Test",
                "Two-Sample t-Test",
                "Chi-Square Test"
            ]
        )

        alpha = st.slider("Significance Level (α)",0.01,0.10,0.05)

        # -------------------------------------

        if test == "One-Sample t-Test":

            var = st.selectbox("Variable", numeric_cols)
            mu = st.number_input("Hypothesized Mean",0.0)

            if st.button("Run Test"):

                data = df[var].dropna()

                stat,p = stats.ttest_1samp(data,mu)

                st.write("t statistic:", round(stat,4))
                st.write("p-value:", round(p,4))

                st.info(interpret_p(p))

        # -------------------------------------

        elif test == "Two-Sample t-Test":

            var = st.selectbox("Numeric Variable", numeric_cols)
            group = st.selectbox("Group Variable", categorical_cols)

            groups = df[group].unique()

            g1 = st.selectbox("Group 1", groups)
            g2 = st.selectbox("Group 2", groups)

            if st.button("Run Test"):

                d1 = df[df[group]==g1][var].dropna()
                d2 = df[df[group]==g2][var].dropna()

                stat,p = stats.ttest_ind(d1,d2)

                st.write("t statistic:", round(stat,4))
                st.write("p-value:", round(p,4))

                st.info(interpret_p(p))

        # -------------------------------------

        elif test == "Chi-Square Test":

            v1 = st.selectbox("Variable 1", categorical_cols)
            v2 = st.selectbox("Variable 2", categorical_cols)

            if st.button("Run Test"):

                table = pd.crosstab(df[v1],df[v2])

                stat,p,_,_ = stats.chi2_contingency(table)

                st.dataframe(table)

                st.write("Chi-square:", round(stat,4))
                st.write("p-value:", round(p,4))

                st.info(interpret_p(p))


# =====================================================
# REGRESSION
# =====================================================

elif analysis_category == "Regression":

    st.subheader("📈 Regression Analysis")

    model_type = st.selectbox(
        "Regression Model",
        [
            "Simple Linear Regression",
            "Multiple Linear Regression",
            "Logistic Regression",
            "Poisson Regression",
            "Stepwise Regression"
        ]
    )

# =====================================================
# SIMPLE LINEAR REGRESSION
# =====================================================

    if model_type == "Simple Linear Regression":

        if len(numeric_cols) < 2:
            st.warning("Dataset must contain at least two numeric variables.")
        else:

            x = st.selectbox("Independent Variable (X)", numeric_cols)
            y = st.selectbox("Dependent Variable (Y)", numeric_cols)

            if st.button("Run Model"):

                data = df[[x, y]].dropna()

                X = data[[x]]
                Y = data[y]

                model = LinearRegression()
                model.fit(X, Y)

                pred = model.predict(X)

                r2 = r2_score(Y, pred)

                st.markdown("### Model Results")

                col1, col2, col3 = st.columns(3)

                col1.metric("Intercept", round(model.intercept_,4))
                col2.metric("Slope", round(model.coef_[0],4))
                col3.metric("R²", round(r2,4))

                st.info(interpret_r2(r2))

                fig, ax = plt.subplots()

                sns.scatterplot(x=X[x], y=Y)
                plt.plot(X, pred)

                plt.title("Regression Fit")

                st.pyplot(fig)

# =====================================================
# MULTIPLE LINEAR REGRESSION
# =====================================================

    elif model_type == "Multiple Linear Regression":

        y = st.selectbox("Dependent Variable", numeric_cols)

        Xvars = st.multiselect(
            "Independent Variables",
            [c for c in numeric_cols if c != y]
        )

        if st.button("Run Model"):

            if len(Xvars) == 0:
                st.warning("Select at least one independent variable.")
            else:

                data = df[[y] + Xvars].dropna()

                X = sm.add_constant(data[Xvars])
                Y = data[y]

                model = sm.OLS(Y, X).fit()

                st.markdown("### Model Summary")

                st.text(model.summary())

                st.info(interpret_r2(model.rsquared))

# =====================================================
# LOGISTIC REGRESSION
# =====================================================

    elif model_type == "Logistic Regression":

        cat_cols = df.select_dtypes(include=["object","category"]).columns

        if len(cat_cols) == 0:
            st.warning("Dataset must contain a categorical variable.")
        else:

            y = st.selectbox("Binary Dependent Variable", cat_cols)

            Xvars = st.multiselect(
                "Independent Variables",
                numeric_cols
            )

            if st.button("Run Model"):

                if len(Xvars) == 0:
                    st.warning("Select predictors.")
                else:

                    data = df[[y] + Xvars].dropna()

                    if data[y].nunique() != 2:
                        st.error("Dependent variable must have exactly 2 categories.")
                    else:

                        Y = pd.factorize(data[y])[0]
                        X = sm.add_constant(data[Xvars])

                        model = sm.Logit(Y, X).fit(disp=0)

                        st.markdown("### Logistic Model Summary")

                        st.text(model.summary())

                        st.info(
                            "Interpretation: coefficients represent change in log-odds."
                        )

# =====================================================
# POISSON REGRESSION
# =====================================================

    elif model_type == "Poisson Regression":

        y = st.selectbox("Count Dependent Variable", numeric_cols)

        Xvars = st.multiselect(
            "Independent Variables",
            [c for c in numeric_cols if c != y]
        )

        if st.button("Run Model"):

            if len(Xvars) == 0:
                st.warning("Select predictors.")
            else:

                data = df[[y] + Xvars].dropna()

                if (data[y] < 0).any():
                    st.error("Poisson regression requires non-negative count data.")
                else:

                    X = sm.add_constant(data[Xvars])
                    Y = data[y]

                    model = sm.GLM(
                        Y,
                        X,
                        family=sm.families.Poisson()
                    ).fit()

                    st.markdown("### Poisson Model Summary")

                    st.text(model.summary())

                    st.info(
                        "Interpretation: coefficients show expected log-count change."
                    )

# =====================================================
# STEPWISE REGRESSION
# =====================================================

    elif model_type == "Stepwise Regression":

        y = st.selectbox("Dependent Variable", numeric_cols)

        Xvars = st.multiselect(
            "Candidate Variables",
            [c for c in numeric_cols if c != y]
        )

        if st.button("Run Stepwise Regression"):

            if len(Xvars) == 0:
                st.warning("Select candidate variables.")
            else:

                data = df[[y] + Xvars].dropna()

                remaining = list(Xvars)
                selected = []
                current_score = 0

                while remaining:

                    scores = []

                    for candidate in remaining:

                        predictors = selected + [candidate]

                        X = sm.add_constant(data[predictors])
                        Y = data[y]

                        model = sm.OLS(Y, X).fit()

                        scores.append((model.rsquared, candidate))

                    scores.sort()

                    best_score, best_candidate = scores[-1]

                    if best_score > current_score:

                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_score

                    else:
                        break

                st.write("### Selected Variables")
                st.write(selected)

                X = sm.add_constant(data[selected])
                Y = data[y]

                final_model = sm.OLS(Y, X).fit()

                st.text(final_model.summary())

                st.info(interpret_r2(final_model.rsquared))

# =====================================================
            
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
