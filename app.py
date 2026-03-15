# =====================================================
# StatX v1 – Statistical Analysis Software
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

# Time Series imports
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="StatX Statistical WebLab",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("📊 StatX Statistical WebLab")
st.markdown(
"""
### Scientific Platform for Statistical Computing

StatX provides a modern environment for:

• Data Analysis  
• Statistical Modeling  
• Scientific Visualization  
• Advanced Research Methods
"""
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("StatX Control Panel")

# =====================================================
# DATA UPLOAD
# =====================================================

uploaded = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv","xlsx","xls","txt","json","parquet","dta","sav"]
)

df = None

if uploaded:

    try:

        name = uploaded.name.lower()

        if name.endswith(".csv"):
            df = pd.read_csv(uploaded)

        elif name.endswith((".xlsx",".xls")):
            df = pd.read_excel(uploaded)

        elif name.endswith(".txt"):
            df = pd.read_csv(uploaded,sep=None,engine="python")

        elif name.endswith(".json"):
            df = pd.read_json(uploaded)

        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)

        elif name.endswith(".dta"):
            df = pd.read_stata(uploaded)

        elif name.endswith(".sav"):
            df = pd.read_spss(uploaded)

        st.sidebar.success("Dataset Loaded Successfully")

    except Exception as e:

        st.sidebar.error(f"Loading Error: {e}")

# =====================================================
# DATASET OVERVIEW
# =====================================================

if df is not None:

    st.subheader("📑 Dataset Overview")

    c1,c2,c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isna().sum().sum())

    st.divider()

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.divider()

    info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing": df.isna().sum()
    })

    st.subheader("Column Information")
    st.dataframe(info, use_container_width=True)

    # =====================================================
    # VARIABLE TYPES
    # =====================================================

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # =====================================================
    # ANALYSIS MENU
    # =====================================================

    st.sidebar.subheader("Analysis Modules")

    analysis_category = st.sidebar.selectbox(
        "Select Module",
        [
            "Descriptive Statistics",
            "Graphics",
            "Regression",
            "Hypothesis Testing",
            "Estimation",
            "Chi-Square Tests",
            "Time Series",
            "ANOVA",
            "Design of Experiments",
            "Multivariate Analysis",
            "Biostatistics",
            "Bioinformatics",
            "Biometrics",
            "Biomolecular Modeling",
            "Chemoinformatics",
            "Quality Control"
        ]
    )
# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

    if analysis_category == "Descriptive Statistics":

        st.subheader("Descriptive Statistics")

        var = st.selectbox("Variable",numeric_cols)

        data = df[var].dropna()

        summary = pd.DataFrame({
            "Statistic":[
                "Sample Size",
                "Mean",
                "Median",
                "Std Dev",
                "Variance",
                "Minimum",
                "Maximum",
                "Skewness",
                "Kurtosis"
            ],
            "Value":[
                len(data),
                np.mean(data),
                np.median(data),
                np.std(data,ddof=1),
                np.var(data,ddof=1),
                np.min(data),
                np.max(data),
                stats.skew(data),
                stats.kurtosis(data)
            ]
        })

        st.dataframe(summary.round(4),use_container_width=True)


# =====================================================
# GRAPHICS
# =====================================================

    if analysis_category == "Graphics":

        st.subheader("Data Visualization")

        plot = st.selectbox(
            "Plot Type",
            [
                "Histogram",
                "Boxplot",
                "Scatter Plot"
            ]
        )

        if plot == "Histogram":

            var = st.selectbox("Variable",numeric_cols)

            fig,ax = plt.subplots()

            sns.histplot(df[var],kde=True,ax=ax)

            st.pyplot(fig)

        elif plot == "Boxplot":

            var = st.selectbox("Variable",numeric_cols)

            fig,ax = plt.subplots()

            sns.boxplot(y=df[var],ax=ax)

            st.pyplot(fig)

        elif plot == "Scatter Plot":

            x = st.selectbox("X Variable",numeric_cols)
            y = st.selectbox("Y Variable",numeric_cols)

            fig,ax = plt.subplots()

            sns.scatterplot(x=df[x],y=df[y],ax=ax)

            st.pyplot(fig)
            
# =====================================================
# REGRESSION OUTPUT FORMATTER
# =====================================================

def regression_table(model, test_stat="t"):

    coef_table = pd.DataFrame({
        "Variable": model.params.index,
        "Estimate": model.params.values,
        "Std Error": model.bse.values,
        f"{test_stat}-value": model.tvalues.values,
        "p-value": model.pvalues.values,
        "CI Lower": model.conf_int()[0].values,
        "CI Upper": model.conf_int()[1].values
    })

    coef_table = coef_table.round(4)

    def stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "."
        else:
            return ""

    coef_table["Sig"] = coef_table["p-value"].apply(stars)

    return coef_table

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def interpret_r2(r2):

    if r2 < 0.10:
        return "Very weak model fit (R² < 0.10)"
    elif r2 < 0.30:
        return "Weak model fit"
    elif r2 < 0.50:
        return "Moderate model fit"
    elif r2 < 0.70:
        return "Good model fit"
    elif r2 < 0.90:
        return "Very good model fit"
    else:
        return "Excellent model fit"


def regression_table(model, stat_label="t"):

    coef_table = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std Error": model.bse.values,
        f"{stat_label} value": model.tvalues.values,
        "p-value": model.pvalues.values,
        "CI Lower": model.conf_int()[0].values,
        "CI Upper": model.conf_int()[1].values
    })

    return coef_table.round(4)


# =====================================================
# CATEGORY SELECTOR (IMPORTANT)
# =====================================================

analysis_category = st.sidebar.selectbox(
    "Analysis Category",
    [
        "Descriptive Statistics",
        "Graphics",
        "Regression",
        "Time Series"
    ]
)


# =====================================================
# REGRESSION
# =====================================================

if analysis_category == "Regression":

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
            st.stop()

        x = st.selectbox("Independent Variable (X)", numeric_cols)
        y = st.selectbox("Dependent Variable (Y)", [c for c in numeric_cols if c != x])

        if st.button("Run Regression"):

            data = df[[x, y]].dropna()

            X = sm.add_constant(data[x])
            Y = data[y]

            model = sm.OLS(Y, X).fit()

            st.markdown("### 📊 Model Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("R²", round(model.rsquared,4))
            col2.metric("Adj R²", round(model.rsquared_adj,4))
            col3.metric("AIC", round(model.aic,2))
            col4.metric("BIC", round(model.bic,2))

            st.info(interpret_r2(model.rsquared))

            coef_table = regression_table(model)

            st.markdown("### 📑 Coefficients")
            st.dataframe(coef_table, use_container_width=True)

            intercept = model.params["const"]
            slope = model.params[x]

            st.success(
                f"Estimated Equation: **{y} = {round(intercept,4)} + {round(slope,4)} × {x}**"
            )

            fig, ax = plt.subplots()
            sns.regplot(x=data[x], y=data[y], ci=95, ax=ax)

            ax.set_title("Simple Linear Regression")
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

        if st.button("Run Regression"):

            if len(Xvars) == 0:
                st.warning("Select at least one independent variable.")
                st.stop()

            data = df[[y] + Xvars].dropna()

            X = sm.add_constant(data[Xvars])
            Y = data[y]

            model = sm.OLS(Y, X).fit()

            st.markdown("### 📊 Model Summary")

            col1,col2,col3,col4 = st.columns(4)

            col1.metric("R²", round(model.rsquared,4))
            col2.metric("Adj R²", round(model.rsquared_adj,4))
            col3.metric("AIC", round(model.aic,2))
            col4.metric("BIC", round(model.bic,2))

            st.info(interpret_r2(model.rsquared))

            coef_table = regression_table(model)

            st.markdown("### 📑 Coefficient Table")
            st.dataframe(coef_table, use_container_width=True)


# =====================================================
# LOGISTIC REGRESSION
# =====================================================

    elif model_type == "Logistic Regression":

        y = st.selectbox("Binary Dependent Variable", categorical_cols)
        Xvars = st.multiselect("Independent Variables", numeric_cols)

        if st.button("Run Logistic Regression"):

            data = df[[y] + Xvars].dropna()

            if data[y].nunique() != 2:
                st.error("Dependent variable must have exactly 2 categories.")
                st.stop()

            Y = pd.factorize(data[y])[0]
            X = sm.add_constant(data[Xvars])

            model = sm.Logit(Y, X).fit(disp=0)

            coef_table = regression_table(model,"z")

            st.markdown("### 📑 Logistic Regression Model")
            st.dataframe(coef_table, use_container_width=True)


# =====================================================
# POISSON REGRESSION
# =====================================================

    elif model_type == "Poisson Regression":

        y = st.selectbox("Count Dependent Variable", numeric_cols)

        Xvars = st.multiselect(
            "Independent Variables",
            [c for c in numeric_cols if c != y]
        )

        if st.button("Run Poisson Regression"):

            data = df[[y] + Xvars].dropna()

            X = sm.add_constant(data[Xvars])
            Y = data[y]

            model = sm.GLM(Y, X, family=sm.families.Poisson()).fit()

            coef_table = regression_table(model,"z")

            st.markdown("### 📑 Poisson Regression Model")
            st.dataframe(coef_table, use_container_width=True)


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

            data = df[[y] + Xvars].dropna()

            remaining = list(Xvars)
            selected = []

            while remaining:

                scores = []

                for candidate in remaining:

                    predictors = selected + [candidate]

                    X = sm.add_constant(data[predictors])
                    Y = data[y]

                    model = sm.OLS(Y, X).fit()

                    scores.append((model.aic, candidate))

                scores.sort()

                best_aic, best_candidate = scores[0]

                selected.append(best_candidate)
                remaining.remove(best_candidate)

            st.success(f"Selected Variables: {selected}")

            X = sm.add_constant(data[selected])
            Y = data[y]

            final_model = sm.OLS(Y, X).fit()

            coef_table = regression_table(final_model)

            st.markdown("### 📑 Final Model")
            st.dataframe(coef_table, use_container_width=True)

            st.info(interpret_r2(final_model.rsquared))
        
# =====================================================
# CHI-SQUARE TESTS
# =====================================================

    if analysis_category == "Chi-Square Tests":

        st.subheader("📊 Chi-Square Analysis")

        test_type = st.selectbox(
            "Select Test",
            [
                "Test of Independence",
                "Goodness of Fit",
                "Test of Homogeneity"
            ]
        )

# =====================================================
# TEST OF INDEPENDENCE
# =====================================================

        if test_type == "Test of Independence":

            st.markdown("### Test of Independence")

            st.write(
            "Tests whether **two categorical variables are statistically independent**."
            )

            if len(categorical_cols) < 2:
                st.warning("Dataset must contain at least two categorical variables.")
            else:

                var1 = st.selectbox("Variable 1", categorical_cols)
                var2 = st.selectbox("Variable 2", categorical_cols)

                if st.button("Run Test"):

                    table = pd.crosstab(df[var1], df[var2])

                    chi2, p, dof, expected = stats.chi2_contingency(table)

                    st.markdown("### Contingency Table")
                    st.dataframe(table)

                    st.markdown("### Expected Frequencies")
                    st.dataframe(pd.DataFrame(
                        expected,
                        index=table.index,
                        columns=table.columns
                    ))

                    col1,col2,col3 = st.columns(3)

                    col1.metric("Chi-square", round(chi2,4))
                    col2.metric("Degrees of Freedom", dof)
                    col3.metric("p-value", round(p,4))

                    st.info(interpret_p(p))

                    if p < 0.05:
                        st.success("Conclusion: Variables are statistically associated.")
                    else:
                        st.warning("Conclusion: No significant association detected.")

# =====================================================
# GOODNESS OF FIT
# =====================================================

        elif test_type == "Goodness of Fit":

            st.markdown("### Goodness-of-Fit Test")

            st.write(
            "Tests whether **observed frequencies match an expected distribution**."
            )

            if len(categorical_cols) == 0:
                st.warning("Dataset must contain a categorical variable.")
            else:

                var = st.selectbox("Categorical Variable", categorical_cols)

                if st.button("Run Test"):

                    observed = df[var].value_counts()

                    expected = np.repeat(
                        observed.mean(),
                        len(observed)
                    )

                    chi2, p = stats.chisquare(
                        f_obs=observed.values,
                        f_exp=expected
                    )

                    freq_table = pd.DataFrame({
                        "Category": observed.index,
                        "Observed": observed.values,
                        "Expected": expected
                    })

                    st.markdown("### Frequency Table")
                    st.dataframe(freq_table)

                    col1,col2 = st.columns(2)

                    col1.metric("Chi-square", round(chi2,4))
                    col2.metric("p-value", round(p,4))

                    st.info(interpret_p(p))

                    if p < 0.05:
                        st.success("Conclusion: Observed frequencies differ from expected distribution.")
                    else:
                        st.warning("Conclusion: Observed frequencies match the expected distribution.")

# =====================================================
# TEST OF HOMOGENEITY
# =====================================================

        elif test_type == "Test of Homogeneity":

            st.markdown("### Test of Homogeneity")

            st.write(
            "Tests whether **different populations share the same distribution** of a categorical variable."
            )

            if len(categorical_cols) < 2:
                st.warning("Dataset must contain at least two categorical variables.")
            else:

                group = st.selectbox("Group Variable", categorical_cols)
                outcome = st.selectbox("Outcome Variable", categorical_cols)

                if st.button("Run Test"):

                    table = pd.crosstab(df[group], df[outcome])

                    chi2, p, dof, expected = stats.chi2_contingency(table)

                    st.markdown("### Frequency Table")
                    st.dataframe(table)

                    st.markdown("### Expected Frequencies")
                    st.dataframe(pd.DataFrame(
                        expected,
                        index=table.index,
                        columns=table.columns
                    ))

                    col1,col2,col3 = st.columns(3)

                    col1.metric("Chi-square", round(chi2,4))
                    col2.metric("Degrees of Freedom", dof)
                    col3.metric("p-value", round(p,4))

                    st.info(interpret_p(p))

                    if p < 0.05:
                        st.success("Conclusion: Distributions differ across populations.")
                    else:
                        st.warning("Conclusion: Distributions are homogeneous across groups.")


# =====================================================
# =====================================================
# TIME SERIES
# =====================================================

if analysis_category == "Time Series":

    st.subheader("📈 Time Series Analysis")

    method = st.selectbox(
        "Method",
        [
            "Time Series Plot",
            "Moving Average",
            "Trend Estimation",
            "Stationarity Test (ADF)",
            "Differencing",
            "Autocorrelation (ACF)",
            "Partial Autocorrelation (PACF)",
            "Seasonal Decomposition"
        ]
    )

    var = st.selectbox("Variable", numeric_cols)

    data = df[var].dropna().reset_index(drop=True)

    if len(data) < 10:
        st.warning("Time series analysis requires at least 10 observations.")
        st.stop()

    # =====================================================
    # TIME SERIES PLOT
    # =====================================================

    if method == "Time Series Plot":

        fig, ax = plt.subplots()

        ax.plot(data, marker="o")

        ax.set_title("Time Series Plot")
        ax.set_xlabel("Observation")
        ax.set_ylabel(var)

        st.pyplot(fig)

        st.info(
            "Interpretation: The plot shows how the variable evolves across observations. "
            "Look for patterns such as trend, seasonality, or structural breaks."
        )

    # =====================================================
    # MOVING AVERAGE
    # =====================================================

    elif method == "Moving Average":

        window = st.slider("Window Size", 2, 30, 5)

        ma = data.rolling(window).mean()

        fig, ax = plt.subplots()

        ax.plot(data, label="Original")
        ax.plot(ma, linewidth=3, label="Moving Average")

        ax.legend()
        ax.set_title("Moving Average Smoothing")

        st.pyplot(fig)

        st.info(
            "Interpretation: Moving averages smooth short-term noise and highlight "
            "the underlying long-term trend of the series."
        )

    # =====================================================
    # TREND ESTIMATION
    # =====================================================

    elif method == "Trend Estimation":

        x = np.arange(len(data))

        slope, intercept = np.polyfit(x, data, 1)

        trend = slope * x + intercept

        fig, ax = plt.subplots()

        ax.plot(data, label="Observed")
        ax.plot(trend, linewidth=3, label="Estimated Trend")

        ax.legend()

        st.pyplot(fig)

        st.info(
            f"Estimated Trend Equation:  Y = {round(intercept,4)} + {round(slope,4)}t\n\n"
            "Interpretation: The slope indicates whether the series is increasing or decreasing over time."
        )

    # =====================================================
    # STATIONARITY TEST
    # =====================================================

    elif method == "Stationarity Test (ADF)":

        result = adfuller(data)

        stat = result[0]
        p = result[1]

        col1, col2 = st.columns(2)

        col1.metric("ADF Statistic", round(stat,4))
        col2.metric("p-value", round(p,4))

        st.write("Critical Values")

        for key,value in result[4].items():
            st.write(f"{key}: {round(value,4)}")

        if p < 0.05:

            st.success(
                "Conclusion: Reject H₀. The time series is **stationary**."
            )

        else:

            st.warning(
                "Conclusion: Fail to reject H₀. The series is **non-stationary**. "
                "Differencing or transformation may be required."
            )

    # =====================================================
    # DIFFERENCING
    # =====================================================

    elif method == "Differencing":

        lag = st.slider("Lag",1,10,1)

        diff = data.diff(lag).dropna()

        fig, ax = plt.subplots()

        ax.plot(diff)

        ax.set_title("Differenced Series")

        st.pyplot(fig)

        st.info(
            "Interpretation: Differencing removes trend and stabilizes the mean, "
            "which helps achieve stationarity for ARIMA modeling."
        )

    # =====================================================
    # AUTOCORRELATION
    # =====================================================

    elif method == "Autocorrelation (ACF)":

        fig, ax = plt.subplots()

        plot_acf(data, ax=ax, lags=min(40, len(data)//2))

        st.pyplot(fig)

        st.info(
            "Interpretation: ACF measures correlation between observations and their lagged values. "
            "Useful for identifying MA components in ARIMA models."
        )

    # =====================================================
    # PARTIAL AUTOCORRELATION
    # =====================================================

    elif method == "Partial Autocorrelation (PACF)":

        fig, ax = plt.subplots()

        max_lag = min(20, len(data)//2)

        plot_pacf(data, ax=ax, lags=max_lag)

        st.pyplot(fig)

        st.info(
            "Interpretation: PACF measures the direct relationship between observations "
            "at different lags and helps identify AR model order."
        )

    # =====================================================
    # SEASONAL DECOMPOSITION
    # =====================================================

    elif method == "Seasonal Decomposition":

        period = st.slider("Seasonal Period",2,24,12)

        if len(data) < period*2:

            st.warning("Dataset must contain at least two seasonal cycles.")

        else:

            decomposition = seasonal_decompose(data, period=period)

            fig = decomposition.plot()

            st.pyplot(fig)

            st.info(
                "Interpretation: Decomposition separates the series into "
                "**Trend, Seasonal, and Residual components**, "
                "revealing hidden patterns in the data."
            )

# =====================================================
# ANOVA
# =====================================================

elif analysis_category == "ANOVA":

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    st.subheader("📊 Analysis of Variance (ANOVA)")

    anova_type = st.selectbox(
        "ANOVA Type",
        ["One-Way ANOVA", "Two-Way ANOVA"]
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# -----------------------------------------------------
# ONE WAY ANOVA
# -----------------------------------------------------

    if anova_type == "One-Way ANOVA":

        response = st.selectbox("Response Variable", numeric_cols)
        factor = st.selectbox("Factor Variable", categorical_cols)

        if st.button("Run One-Way ANOVA"):

            model = ols(f"{response} ~ C({factor})", data=df).fit()
            table = sm.stats.anova_lm(model, typ=2)

            st.write("### ANOVA Table")
            st.dataframe(table)

            p = table["PR(>F)"][0]

            if p < 0.05:

                st.success(
                    "Interpretation: There is a statistically significant difference between group means."
                )

            else:

                st.info(
                    "Interpretation: No statistically significant difference between group means."
                )

# -----------------------------------------------------
# TWO WAY ANOVA
# -----------------------------------------------------

    elif anova_type == "Two-Way ANOVA":

        response = st.selectbox("Response Variable", numeric_cols)
        factor1 = st.selectbox("Factor A", categorical_cols)
        factor2 = st.selectbox("Factor B", categorical_cols)

        if st.button("Run Two-Way ANOVA"):

            formula = f"{response} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"

            model = ols(formula, data=df).fit()

            table = sm.stats.anova_lm(model, typ=2)

            st.write("### Two-Way ANOVA Table")
            st.dataframe(table)

            st.markdown("### Interpretation")

            for factor in table.index:

                if factor != "Residual":

                    p = table.loc[factor, "PR(>F)"]

                    if p < 0.05:

                        st.success(f"{factor} has a significant effect (p < 0.05)")

                    else:

                        st.info(f"{factor} is not statistically significant")

# =====================================================
# DESIGN OF EXPERIMENTS
# =====================================================

elif analysis_category == "Design of Experiments (DOE)":

    st.subheader("🧪 Design of Experiments")

    doe_type = st.selectbox(
        "Experimental Design",
        [
            "Completely Randomized Design (CRD)",
            "Randomized Block Design (RBD)"
        ]
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# -----------------------------------------------------
# CRD
# -----------------------------------------------------

    if doe_type == "Completely Randomized Design (CRD)":

        response = st.selectbox("Response Variable", numeric_cols)
        treatment = st.selectbox("Treatment Factor", categorical_cols)

        if st.button("Run CRD Analysis"):

            model = ols(f"{response} ~ C({treatment})", data=df).fit()
            table = sm.stats.anova_lm(model, typ=2)

            st.write(table)

            p = table["PR(>F)"][0]

            if p < 0.05:

                st.success("Treatments significantly affect the response.")

            else:

                st.info("No significant treatment effect detected.")

# -----------------------------------------------------
# RBD
# -----------------------------------------------------

    elif doe_type == "Randomized Block Design (RBD)":

        response = st.selectbox("Response Variable", numeric_cols)
        treatment = st.selectbox("Treatment Factor", categorical_cols)
        block = st.selectbox("Block Factor", categorical_cols)

        if st.button("Run RBD Analysis"):

            formula = f"{response} ~ C({treatment}) + C({block})"

            model = ols(formula, data=df).fit()
            table = sm.stats.anova_lm(model, typ=2)

            st.write(table)

            st.info("Interpretation: Tests treatment effects while controlling block variability.")

# =====================================================
# MULTIVARIATE ANALYSIS
# =====================================================

elif analysis_category == "Multivariate Analysis":

    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.cluster import KMeans
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    st.subheader("📊 Multivariate Statistical Methods")

    method = st.selectbox(
        "Select Method",
        [
            "Principal Component Analysis (PCA)",
            "Factor Analysis",
            "Cluster Analysis (K-Means)",
            "Discriminant Analysis"
        ]
    )

    numeric_data = df.select_dtypes(include=np.number).dropna()

    if numeric_data.shape[1] < 2:
        st.warning("Multivariate methods require at least two numeric variables.")
        st.stop()

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_data)

# ------------------------------
# PCA
# ------------------------------

    if method == "Principal Component Analysis (PCA)":

        pca = PCA()
        components = pca.fit_transform(X)

        st.write("### Explained Variance Ratio")
        st.write(pca.explained_variance_ratio_)

        fig, ax = plt.subplots()
        ax.scatter(components[:,0], components[:,1])
        ax.set_title("PCA Score Plot")

        st.pyplot(fig)

        st.info(
        "Interpretation: PCA reduces dimensionality by transforming variables "
        "into principal components that explain maximum variance."
        )

# ------------------------------
# FACTOR ANALYSIS
# ------------------------------

    elif method == "Factor Analysis":

        n_factors = st.slider("Number of Factors",1,5,2)

        fa = FactorAnalysis(n_components=n_factors)
        factors = fa.fit_transform(X)

        st.write("Factor Loadings")
        st.write(fa.components_)

        st.info(
        "Interpretation: Factor analysis identifies latent variables "
        "that explain correlations among observed variables."
        )

# ------------------------------
# CLUSTERING
# ------------------------------

    elif method == "Cluster Analysis (K-Means)":

        k = st.slider("Number of Clusters",2,10,3)

        model = KMeans(n_clusters=k)
        clusters = model.fit_predict(X)

        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1], c=clusters)
        ax.set_title("Cluster Plot")

        st.pyplot(fig)

        st.info(
        "Interpretation: Clustering groups observations with similar characteristics."
        )

# ------------------------------
# DISCRIMINANT ANALYSIS
# ------------------------------

    elif method == "Discriminant Analysis":

        target = st.selectbox("Target Group Variable", categorical_cols)

        X = df[numeric_cols].dropna()
        y = df[target].loc[X.index]

        lda = LinearDiscriminantAnalysis()
        lda.fit(X,y)

        st.write("Classification Accuracy:", lda.score(X,y))

        st.info(
        "Interpretation: Discriminant analysis identifies variables "
        "that best separate predefined groups."
        )

# =====================================================
# BIOSTATISTICS
# =====================================================

elif analysis_category == "Biostatistics":

    st.subheader("🧬 Biostatistics Analysis")

    method = st.selectbox(
        "Select Method",
        [
            "Relative Risk",
            "Odds Ratio",
            "Kaplan-Meier Survival Estimate"
        ]
    )

# ------------------------------
# RELATIVE RISK
# ------------------------------

    if method == "Relative Risk":

        a = st.number_input("Exposed + Disease",10)
        b = st.number_input("Exposed + No Disease",20)
        c = st.number_input("Unexposed + Disease",5)
        d = st.number_input("Unexposed + No Disease",30)

        rr = (a/(a+b)) / (c/(c+d))

        st.metric("Relative Risk", round(rr,4))

        if rr > 1:
            st.warning("Exposure increases disease risk.")
        else:
            st.success("Exposure does not increase risk.")

# ------------------------------
# ODDS RATIO
# ------------------------------

    elif method == "Odds Ratio":

        a = st.number_input("Case + Exposure",10)
        b = st.number_input("Control + Exposure",20)
        c = st.number_input("Case + No Exposure",5)
        d = st.number_input("Control + No Exposure",30)

        OR = (a*d)/(b*c)

        st.metric("Odds Ratio", round(OR,4))

        st.info(
        "Interpretation: Odds ratio measures association "
        "between exposure and outcome."
        )

# ------------------------------
# SURVIVAL ANALYSIS
# ------------------------------

    elif method == "Kaplan-Meier Survival Estimate":

        st.info("Upload survival time and event variables.")

        time_var = st.selectbox("Survival Time Variable", numeric_cols)
        event_var = st.selectbox("Event Indicator (0/1)", numeric_cols)

        from lifelines import KaplanMeierFitter

        kmf = KaplanMeierFitter()
        kmf.fit(df[time_var], df[event_var])

        fig, ax = plt.subplots()
        kmf.plot(ax=ax)

        st.pyplot(fig)

        st.info(
        "Interpretation: Kaplan-Meier estimates survival probability over time."
        )

# =====================================================
# BIOINFORMATICS
# =====================================================

elif analysis_category == "Bioinformatics":

    st.subheader("🧬 Bioinformatics Tools")

    method = st.selectbox(
        "Bioinformatics Method",
        [
            "Gene Expression Heatmap",
            "Sequence Length Analysis"
        ]
    )

    if method == "Gene Expression Heatmap":

        numeric_data = df.select_dtypes(include=np.number)

        fig, ax = plt.subplots()

        sns.heatmap(numeric_data.corr(), cmap="coolwarm")

        st.pyplot(fig)

        st.info(
        "Interpretation: Heatmap reveals correlation patterns "
        "between gene expression variables."
        )

    elif method == "Sequence Length Analysis":

        st.info("Analyze length distribution of genetic sequences.")

        seq_col = st.selectbox("Sequence Column", df.columns)

        lengths = df[seq_col].astype(str).apply(len)

        fig, ax = plt.subplots()

        sns.histplot(lengths)

        st.pyplot(fig)

        st.info(
        "Interpretation: Shows distribution of sequence lengths."
        )

# =====================================================
# BIOMETRICS
# =====================================================

elif analysis_category == "Biometrics":

    st.subheader("🔐 Biometric Statistical Analysis")

    method = st.selectbox(
        "Biometric Method",
        [
            "Classification Accuracy",
            "Similarity Matrix"
        ]
    )

    if method == "Classification Accuracy":

        from sklearn.metrics import accuracy_score

        y_true = st.text_input("True Labels (comma separated)")
        y_pred = st.text_input("Predicted Labels (comma separated)")

        if st.button("Compute Accuracy"):

            y_true = y_true.split(",")
            y_pred = y_pred.split(",")

            acc = accuracy_score(y_true,y_pred)

            st.metric("Accuracy",round(acc,4))

    elif method == "Similarity Matrix":

        numeric_data = df.select_dtypes(include=np.number)

        corr = numeric_data.corr()

        fig, ax = plt.subplots()

        sns.heatmap(corr)

        st.pyplot(fig)

        st.info(
        "Interpretation: Similarity matrix reveals relationships "
        "between biometric features."
        )

# =====================================================
# BIOMOLECULAR MODELING
# =====================================================

elif analysis_category == "Biomolecular Modeling":

    st.subheader("🧪 Biomolecular Modeling")

    method = st.selectbox(
        "Modeling Tool",
        [
            "Distance Matrix",
            "Molecular Descriptor Correlation"
        ]
    )

    numeric_data = df.select_dtypes(include=np.number)

    if method == "Distance Matrix":

        from scipy.spatial.distance import pdist, squareform

        dist = squareform(pdist(numeric_data))

        st.write(dist)

        st.info(
        "Interpretation: Distance matrix measures similarity between molecules."
        )

    elif method == "Molecular Descriptor Correlation":

        corr = numeric_data.corr()

        fig, ax = plt.subplots()

        sns.heatmap(corr)

        st.pyplot(fig)

        st.info(
        "Interpretation: Correlation among molecular descriptors."
        )

# =====================================================
# CHEMOINFORMATICS
# =====================================================

elif analysis_category == "Chemoinformatics":

    st.subheader("⚗ Chemoinformatics Analysis")

    method = st.selectbox(
        "Chemoinformatics Method",
        [
            "Descriptor Correlation",
            "Molecular Clustering"
        ]
    )

    numeric_data = df.select_dtypes(include=np.number)

    if method == "Descriptor Correlation":

        corr = numeric_data.corr()

        fig, ax = plt.subplots()

        sns.heatmap(corr, cmap="coolwarm")

        st.pyplot(fig)

        st.info(
        "Interpretation: Shows relationships between chemical descriptors."
        )

    elif method == "Molecular Clustering":

        k = st.slider("Number of Clusters",2,10,3)

        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k)
        clusters = model.fit_predict(numeric_data)

        fig, ax = plt.subplots()

        ax.scatter(numeric_data.iloc[:,0], numeric_data.iloc[:,1], c=clusters)

        st.pyplot(fig)

        st.info(
        "Interpretation: Clustering groups chemically similar compounds."
        )

# =====================================================
# QUALITY CONTROL
# =====================================================

elif analysis_category == "Quality Control":

    st.subheader("🏭 Statistical Quality Control (SPC)")

    method = st.selectbox(
        "Quality Control Method",
        [
            "Individuals Control Chart (I Chart)",
            "Moving Range Chart (MR Chart)",
            "X-Bar Chart (Subgroup Mean)",
            "P Chart (Proportion Defective)",
            "Process Capability Analysis"
        ]
    )

    var = st.selectbox("Process Variable", numeric_cols)

    data = df[var].dropna()

    if len(data) < 5:
        st.warning("At least 5 observations are required.")
        st.stop()

# =====================================================
# INDIVIDUALS CONTROL CHART
# =====================================================

    if method == "Individuals Control Chart (I Chart)":

        mean = np.mean(data)
        std = np.std(data)

        UCL = mean + 3*std
        LCL = mean - 3*std

        fig, ax = plt.subplots()

        ax.plot(data, marker="o")
        ax.axhline(mean, linestyle="--", label="Mean")
        ax.axhline(UCL, color="red", label="UCL")
        ax.axhline(LCL, color="red", label="LCL")

        ax.set_title("Individuals Control Chart")

        ax.legend()

        st.pyplot(fig)

        out = ((data>UCL)|(data<LCL)).sum()

        st.metric("Out of Control Points",int(out))

        if out>0:
            st.warning(
            "Interpretation: The process shows signals of instability. "
            "Special causes may be affecting the process."
            )
        else:
            st.success(
            "Interpretation: The process appears statistically stable."
            )

# =====================================================
# MOVING RANGE CHART
# =====================================================

    elif method == "Moving Range Chart (MR Chart)":

        mr = np.abs(np.diff(data))

        MRbar = np.mean(mr)

        UCL = 3.267 * MRbar

        fig, ax = plt.subplots()

        ax.plot(mr, marker="o")

        ax.axhline(MRbar, linestyle="--", label="Mean MR")
        ax.axhline(UCL, color="red", label="UCL")

        ax.set_title("Moving Range Chart")

        ax.legend()

        st.pyplot(fig)

        st.info(
        "Interpretation: Moving Range chart evaluates variability between consecutive observations."
        )

# =====================================================
# XBAR CHART
# =====================================================

    elif method == "X-Bar Chart (Subgroup Mean)":

        subgroup = st.slider("Subgroup Size",2,10,5)

        groups = [data[i:i+subgroup] for i in range(0,len(data),subgroup)]

        means = [np.mean(g) for g in groups if len(g)==subgroup]

        xbar = np.mean(means)
        std = np.std(means)

        UCL = xbar + 3*std
        LCL = xbar - 3*std

        fig, ax = plt.subplots()

        ax.plot(means, marker="o")

        ax.axhline(xbar,label="Mean")
        ax.axhline(UCL,color="red",label="UCL")
        ax.axhline(LCL,color="red",label="LCL")

        ax.set_title("X-Bar Chart")

        ax.legend()

        st.pyplot(fig)

        st.info(
        "Interpretation: X-Bar chart monitors changes in process mean across subgroups."
        )

# =====================================================
# P CHART
# =====================================================

    elif method == "P Chart (Proportion Defective)":

        defects = st.number_input("Number of Defects",10)
        n = st.number_input("Sample Size",100)

        p = defects/n

        UCL = p + 3*np.sqrt(p*(1-p)/n)
        LCL = max(0,p - 3*np.sqrt(p*(1-p)/n))

        st.metric("Proportion Defective",round(p,4))

        st.write("UCL:",round(UCL,4))
        st.write("LCL:",round(LCL,4))

        st.info(
        "Interpretation: P chart monitors proportion of defective items."
        )

# =====================================================
# PROCESS CAPABILITY
# =====================================================

    elif method == "Process Capability Analysis":

        st.markdown("### Specification Limits")

        LSL = st.number_input("Lower Spec Limit",0.0)
        USL = st.number_input("Upper Spec Limit",1.0)

        if USL<=LSL:
            st.error("USL must be greater than LSL")
            st.stop()

        mean = np.mean(data)
        std = np.std(data)

        Cp = (USL-LSL)/(6*std)

        Cpk = min((USL-mean)/(3*std),(mean-LSL)/(3*std))

        Pp = (USL-LSL)/(6*np.std(data,ddof=1))

        Ppk = min((USL-mean)/(3*np.std(data,ddof=1)),
                  (mean-LSL)/(3*np.std(data,ddof=1)))

        col1,col2 = st.columns(2)

        col1.metric("Cp",round(Cp,4))
        col2.metric("Cpk",round(Cpk,4))

        col3,col4 = st.columns(2)

        col3.metric("Pp",round(Pp,4))
        col4.metric("Ppk",round(Ppk,4))

# =====================================================
# HISTOGRAM
# =====================================================

        fig, ax = plt.subplots()

        sns.histplot(data,kde=True)

        ax.axvline(LSL,color="red",linestyle="--",label="LSL")
        ax.axvline(USL,color="red",linestyle="--",label="USL")

        ax.set_title("Process Distribution")

        ax.legend()

        st.pyplot(fig)

# =====================================================
# INTERPRETATION
# =====================================================

        if Cpk>=1.33:
            st.success(
            "Interpretation: The process is capable and meets quality requirements."
            )

        elif Cpk>=1:
            st.info(
            "Interpretation: The process is marginally capable. Improvements recommended."
            )

        else:
            st.warning(
            "Interpretation: The process is not capable of meeting specifications."
            )

# =====================================================
# DATA NOT UPLOADED
# =====================================================

else:

    st.info("Upload a dataset to begin statistical analysis.")

