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

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.sidebar.markdown("## 📊 StatX Analysis Menu")

    analysis_category = st.sidebar.selectbox(
        "Select Analysis Module",
        [
            "Descriptive Statistics",
            "Graphics & Visualization",
            "Hypothesis Testing",
            "Estimation",
            "Regression Analysis",
            "Chi-Square & Categorical Tests",
            "ANOVA",
            "Design of Experiments (DOE)",
            "Time Series Analysis",
            "Quality Control",
            "Multivariate Analysis",
            "Biostatistics",
            "Bioinformatics",
            "Biometrics",
            "Biomolecular Modeling",
            "Chemoinformatics"
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

elif analysis_category == "Hypothesis Testing":

    st.subheader("📊 Hypothesis Testing")

    test = st.selectbox(
        "Select Statistical Test",
        [
            "One-Sample t-Test",
            "Two-Sample t-Test (Independent)",
            "Paired t-Test",
            "Z-Test (Large Sample Mean)",
            "Proportion Test",
            "Chi-Square Test (Independence)",
            "Chi-Square Goodness-of-Fit",
            "ANOVA (One-Way)"
        ]
    )

    alpha = st.slider("Significance Level (α)",0.01,0.10,0.05)

# -----------------------------------------------------
# ONE SAMPLE T TEST
# -----------------------------------------------------

    if test == "One-Sample t-Test":

        var = st.selectbox("Variable", numeric_cols)
        mu = st.number_input("Hypothesized Mean (μ₀)",0.0)

        if st.button("Run Test"):

            data = df[var].dropna()

            stat,p = stats.ttest_1samp(data,mu)

            mean = np.mean(data)

            st.write("Sample Mean:", round(mean,4))
            st.write("t statistic:", round(stat,4))
            st.write("p-value:", round(p,4))

            if p < alpha:

                st.success("Reject H₀: The population mean differs significantly from the hypothesized mean.")

            else:

                st.info("Fail to reject H₀: No significant evidence that the population mean differs.")

# -----------------------------------------------------
# TWO SAMPLE T TEST
# -----------------------------------------------------

    elif test == "Two-Sample t-Test (Independent)":

        var = st.selectbox("Numeric Variable", numeric_cols)
        group = st.selectbox("Grouping Variable", categorical_cols)

        groups = df[group].dropna().unique()

        g1 = st.selectbox("Group 1", groups)
        g2 = st.selectbox("Group 2", groups)

        equal_var = st.checkbox("Assume Equal Variances",True)

        if st.button("Run Test"):

            d1 = df[df[group]==g1][var].dropna()
            d2 = df[df[group]==g2][var].dropna()

            stat,p = stats.ttest_ind(d1,d2,equal_var=equal_var)

            st.write("Mean (Group1):", round(np.mean(d1),4))
            st.write("Mean (Group2):", round(np.mean(d2),4))
            st.write("t statistic:", round(stat,4))
            st.write("p-value:", round(p,4))

            if p < alpha:

                st.success("Reject H₀: The two group means differ significantly.")

            else:

                st.info("Fail to reject H₀: No significant difference between group means.")

# -----------------------------------------------------
# PAIRED T TEST
# -----------------------------------------------------

    elif test == "Paired t-Test":

        v1 = st.selectbox("Variable 1 (Before)", numeric_cols)
        v2 = st.selectbox("Variable 2 (After)", numeric_cols)

        if st.button("Run Test"):

            data = df[[v1,v2]].dropna()

            stat,p = stats.ttest_rel(data[v1],data[v2])

            st.write("t statistic:", round(stat,4))
            st.write("p-value:", round(p,4))

            if p < alpha:

                st.success("Reject H₀: Significant difference between paired observations.")

            else:

                st.info("Fail to reject H₀: No significant change detected.")

# -----------------------------------------------------
# Z TEST
# -----------------------------------------------------

    elif test == "Z-Test (Large Sample Mean)":

        var = st.selectbox("Variable", numeric_cols)
        mu = st.number_input("Hypothesized Mean",0.0)

        if st.button("Run Test"):

            data = df[var].dropna()

            mean = np.mean(data)
            std = np.std(data)
            n = len(data)

            z = (mean - mu)/(std/np.sqrt(n))

            p = 2*(1-stats.norm.cdf(abs(z)))

            st.write("Z statistic:", round(z,4))
            st.write("p-value:", round(p,4))

            if p < alpha:

                st.success("Reject H₀: Population mean significantly differs.")

            else:

                st.info("Fail to reject H₀.")

# -----------------------------------------------------
# PROPORTION TEST
# -----------------------------------------------------

    elif test == "Proportion Test":

        successes = st.number_input("Number of Successes",0)
        n = st.number_input("Sample Size",1)
        p0 = st.number_input("Hypothesized Proportion",0.5)

        if st.button("Run Test"):

            phat = successes/n

            z = (phat-p0)/np.sqrt((p0*(1-p0))/n)

            p = 2*(1-stats.norm.cdf(abs(z)))

            st.write("Sample Proportion:",round(phat,4))
            st.write("Z statistic:",round(z,4))
            st.write("p-value:",round(p,4))

            if p < alpha:

                st.success("Reject H₀: Population proportion differs.")

            else:

                st.info("Fail to reject H₀.")

# -----------------------------------------------------
# CHI SQUARE INDEPENDENCE
# -----------------------------------------------------

    elif test == "Chi-Square Test (Independence)":

        v1 = st.selectbox("Variable 1", categorical_cols)
        v2 = st.selectbox("Variable 2", categorical_cols)

        if st.button("Run Test"):

            table = pd.crosstab(df[v1],df[v2])

            chi2,p,dof,_ = stats.chi2_contingency(table)

            st.dataframe(table)

            st.write("Chi-square:",round(chi2,4))
            st.write("Degrees of freedom:",dof)
            st.write("p-value:",round(p,4))

            if p < alpha:

                st.success("Reject H₀: Variables are statistically associated.")

            else:

                st.info("Fail to reject H₀: Variables appear independent.")

# -----------------------------------------------------
# ANOVA
# -----------------------------------------------------

    elif test == "ANOVA (One-Way)":

        var = st.selectbox("Numeric Variable", numeric_cols)
        group = st.selectbox("Factor Variable", categorical_cols)

        if st.button("Run Test"):

            groups = [g[var].dropna().values for name,g in df.groupby(group)]

            stat,p = stats.f_oneway(*groups)

            st.write("F statistic:",round(stat,4))
            st.write("p-value:",round(p,4))

            if p < alpha:

                st.success("Reject H₀: At least one group mean differs.")

            else:

                st.info("Fail to reject H₀: No significant difference detected.")

    elif analysis_category == "Estimation":

    st.subheader("📐 Statistical Estimation")

    method = st.selectbox(
        "Estimation Method",
        [
            "Point Estimation (Mean, Variance)",
            "Confidence Interval for Mean",
            "Confidence Interval for Proportion",
            "Confidence Interval for Variance"
        ]
    )

# -----------------------------------------------------
# POINT ESTIMATION
# -----------------------------------------------------

    if method == "Point Estimation (Mean, Variance)":

        var = st.selectbox("Variable", numeric_cols)

        data = df[var].dropna()

        mean = np.mean(data)
        var_est = np.var(data)
        std = np.std(data)

        st.metric("Sample Mean",round(mean,4))
        st.metric("Sample Variance",round(var_est,4))
        st.metric("Sample Standard Deviation",round(std,4))

        st.info(
            "Interpretation: These statistics estimate the unknown population parameters "
            "using observed sample data."
        )

# -----------------------------------------------------
# CI MEAN
# -----------------------------------------------------

    elif method == "Confidence Interval for Mean":

        var = st.selectbox("Variable", numeric_cols)

        conf = st.slider("Confidence Level",0.80,0.99,0.95)

        data = df[var].dropna()

        mean = np.mean(data)
        std = np.std(data)
        n = len(data)

        t = stats.t.ppf((1+conf)/2,n-1)

        margin = t*(std/np.sqrt(n))

        lower = mean - margin
        upper = mean + margin

        st.write("Mean:",round(mean,4))
        st.write("Confidence Interval:",(round(lower,4),round(upper,4)))

        st.success(
            f"We are {int(conf*100)}% confident that the true population mean lies within this interval."
        )

# -----------------------------------------------------
# CI PROPORTION
# -----------------------------------------------------

    elif method == "Confidence Interval for Proportion":

        successes = st.number_input("Number of Successes",0)
        n = st.number_input("Sample Size",1)

        conf = st.slider("Confidence Level",0.80,0.99,0.95)

        p = successes/n

        z = stats.norm.ppf((1+conf)/2)

        margin = z*np.sqrt((p*(1-p))/n)

        lower = p-margin
        upper = p+margin

        st.write("Sample Proportion:",round(p,4))
        st.write("Confidence Interval:",(round(lower,4),round(upper,4)))

        st.info(
            f"Interpretation: With {int(conf*100)}% confidence, the population proportion lies within this interval."
        )

# -----------------------------------------------------
# CI VARIANCE
# -----------------------------------------------------

    elif method == "Confidence Interval for Variance":

        var = st.selectbox("Variable", numeric_cols)

        conf = st.slider("Confidence Level",0.80,0.99,0.95)

        data = df[var].dropna()

        n = len(data)
        s2 = np.var(data,ddof=1)

        chi1 = stats.chi2.ppf((1-conf)/2,n-1)
        chi2 = stats.chi2.ppf((1+conf)/2,n-1)

        lower = (n-1)*s2/chi2
        upper = (n-1)*s2/chi1

        st.write("Sample Variance:",round(s2,4))
        st.write("Confidence Interval:",(round(lower,4),round(upper,4)))

        st.info(
            f"Interpretation: With {int(conf*100)}% confidence, the population variance lies in this interval."
        )



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

                if st.button("Run Regression"):

                    data = df[[x,y]].dropna()

                    X = data[[x]]
                    Y = data[y]

                    model = LinearRegression()
                    model.fit(X,Y)

                    pred = model.predict(X)

                    r2 = r2_score(Y,pred)

                    st.markdown("### Model Results")

                    col1,col2,col3 = st.columns(3)

                    col1.metric("Intercept", round(model.intercept_,4))
                    col2.metric("Slope", round(model.coef_[0],4))
                    col3.metric("R²", round(r2,4))

                    st.info(interpret_r2(r2))

                    fig, ax = plt.subplots()

                    sns.scatterplot(x=X[x],y=Y)
                    ax.plot(X,pred)

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

                else:

                    data = df[[y]+Xvars].dropna()

                    X = sm.add_constant(data[Xvars])
                    Y = data[y]

                    model = sm.OLS(Y,X).fit()

                    st.markdown("### Model Summary")

                    st.text(model.summary())

                    st.info(interpret_r2(model.rsquared))

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

        elif model_type == "Logistic Regression":

            if len(categorical_cols) == 0:
                st.warning("Dataset must contain a categorical variable.")

            else:

                y = st.selectbox("Binary Dependent Variable", categorical_cols)

                Xvars = st.multiselect(
                    "Independent Variables",
                    numeric_cols
                )

                if st.button("Run Logistic Regression"):

                    if len(Xvars) == 0:
                        st.warning("Select predictors.")

                    else:

                        data = df[[y]+Xvars].dropna()

                        if data[y].nunique() != 2:
                            st.error("Dependent variable must have exactly 2 categories.")

                        else:

                            Y = pd.factorize(data[y])[0]
                            X = sm.add_constant(data[Xvars])

                            model = sm.Logit(Y,X).fit(disp=0)

                            st.markdown("### Logistic Regression Summary")

                            st.text(model.summary())

                            st.info("Interpretation: coefficients represent change in log-odds.")

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

                if len(Xvars) == 0:
                    st.warning("Select predictors.")

                else:

                    data = df[[y]+Xvars].dropna()

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

                        st.markdown("### Poisson Regression Summary")

                        st.text(model.summary())

                        st.info("Interpretation: coefficients show expected log-count change.")

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

                    data = df[[y]+Xvars].dropna()

                    remaining = list(Xvars)
                    selected = []
                    current_score = 0

                    while remaining:

                        scores = []

                        for candidate in remaining:

                            predictors = selected+[candidate]

                            X = sm.add_constant(data[predictors])
                            Y = data[y]

                            model = sm.OLS(Y,X).fit()

                            scores.append((model.rsquared,candidate))

                        scores.sort()

                        best_score,best_candidate = scores[-1]

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

                    final_model = sm.OLS(Y,X).fit()

                    st.text(final_model.summary())

                    st.info(interpret_r2(final_model.rsquared))

# =====================================================
# CHI-SQUARE TESTS
# =====================================================

    elif analysis_category == "Chi-Square Tests":

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
# TIME SERIES
# =====================================================

elif analysis_category == "Time Series":

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import seasonal_decompose

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

    data = df[var].dropna()

    if len(data) < 10:
        st.warning("Time series analysis requires at least 10 observations.")
        st.stop()

    # =====================================================
    # TIME SERIES PLOT
    # =====================================================

    if method == "Time Series Plot":

        fig, ax = plt.subplots()

        ax.plot(data)

        ax.set_title("Time Series Plot")
        ax.set_xlabel("Observation")
        ax.set_ylabel(var)

        st.pyplot(fig)

        st.info("Interpretation: Displays how the variable changes over time.")

    # =====================================================
    # MOVING AVERAGE
    # =====================================================

    elif method == "Moving Average":

        window = st.slider("Window Size", 2, 30, 5)

        ma = data.rolling(window).mean()

        fig, ax = plt.subplots()

        ax.plot(data, label="Original")
        ax.plot(ma, label="Moving Average")

        ax.legend()
        ax.set_title("Moving Average Smoothing")

        st.pyplot(fig)

        st.info("Interpretation: Moving averages smooth short-term fluctuations and reveal long-term trends.")

    # =====================================================
    # TREND ESTIMATION
    # =====================================================

    elif method == "Trend Estimation":

        x = np.arange(len(data))

        slope, intercept = np.polyfit(x, data, 1)

        trend = slope * x + intercept

        fig, ax = plt.subplots()

        ax.plot(data, label="Observed")
        ax.plot(trend, label="Trend")

        ax.legend()

        st.pyplot(fig)

        st.info("Interpretation: Trend estimation shows the long-term direction of the time series.")

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
                "Interpretation: The time series is **stationary** (reject unit root hypothesis)."
            )

        else:

            st.warning(
                "Interpretation: The series is **non-stationary**. Differencing may be required."
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

        st.info("Interpretation: Differencing removes trend and stabilizes the mean.")

    # =====================================================
    # AUTOCORRELATION
    # =====================================================

    elif method == "Autocorrelation (ACF)":

        fig, ax = plt.subplots()

        plot_acf(data, ax=ax)

        st.pyplot(fig)

        st.info("Interpretation: ACF measures correlation between observations and their lagged values.")

    # =====================================================
    # PARTIAL AUTOCORRELATION
    # =====================================================

    elif method == "Partial Autocorrelation (PACF)":

        fig, ax = plt.subplots()

        max_lag = min(20, len(data)//2)

        plot_pacf(data, ax=ax, lags=max_lag)

        st.pyplot(fig)

        st.info("Interpretation: PACF identifies the direct relationship between observations at different lags.")

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

            st.info("Interpretation: Decomposition separates trend, seasonal, and residual components.")

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

