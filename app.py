```python
# =====================================================
# StatX v1 – Statistical Analysis Platform
# With Automatic Statistical Interpretation
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

st.set_page_config(page_title="StatX", layout="wide")

st.title("StatX Scientific Statistical Platform")

st.write("Upload a dataset and perform statistical analysis with automatic interpretation.")

# =====================================================
# DATA UPLOAD
# =====================================================

st.sidebar.header("Dataset")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

df = None

if uploaded:

    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)

    else:
        df = pd.read_excel(uploaded)

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

    menu = st.sidebar.selectbox(
        "Analysis",
        [
            "Descriptive Statistics",
            "Graphics",
            "Hypothesis Tests",
            "ANOVA",
            "Regression",
            "Chi-Square Tests"
        ]
    )

# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

    if menu == "Descriptive Statistics":

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

        st.write("Interpretation")

        if summary["Skewness"][0] > 0:
            st.write("Distribution is positively skewed.")

        elif summary["Skewness"][0] < 0:
            st.write("Distribution is negatively skewed.")

        else:
            st.write("Distribution is symmetric.")

# =====================================================
# GRAPHICS
# =====================================================

    elif menu == "Graphics":

        st.subheader("Data Visualization")

        plot = st.selectbox(
            "Plot Type",
            ["Histogram","Boxplot","Scatter","Correlation Heatmap"]
        )

        if plot == "Histogram":

            var = st.selectbox("Variable", numeric_cols)

            fig, ax = plt.subplots()
            sns.histplot(df[var], kde=True)
            st.pyplot(fig)

            st.write("Interpretation")
            st.write("Histogram shows the distribution shape of the variable.")

        elif plot == "Boxplot":

            var = st.selectbox("Variable", numeric_cols)

            fig, ax = plt.subplots()
            sns.boxplot(y=df[var])
            st.pyplot(fig)

            st.write("Interpretation")
            st.write("Boxplot highlights median, spread, and potential outliers.")

        elif plot == "Scatter":

            x = st.selectbox("X variable", numeric_cols)
            y = st.selectbox("Y variable", numeric_cols)

            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x], y=df[y])
            st.pyplot(fig)

            corr = df[[x,y]].corr().iloc[0,1]

            st.write("Correlation:",corr)

            if abs(corr) > 0.7:
                st.write("Strong relationship between variables.")

            elif abs(corr) > 0.4:
                st.write("Moderate relationship.")

            else:
                st.write("Weak relationship.")

        elif plot == "Correlation Heatmap":

            corr = df[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            st.pyplot(fig)

            st.write("Interpretation")
            st.write("Heatmap shows pairwise correlations among variables.")

# =====================================================
# HYPOTHESIS TESTING
# =====================================================

    elif menu == "Hypothesis Tests":

        st.subheader("Hypothesis Testing")

        test = st.selectbox(
            "Select Test",
            ["One Sample t-test","Two Sample t-test"]
        )

        if test == "One Sample t-test":

            var = st.selectbox("Variable", numeric_cols)
            mu = st.number_input("Hypothesized Mean")

            t,p = stats.ttest_1samp(df[var], mu)

            st.write("t statistic:",t)
            st.write("p value:",p)

            st.write("Interpretation")
            st.write(interpret_p(p))

        elif test == "Two Sample t-test":

            group = st.selectbox("Grouping variable", categorical_cols)
            var = st.selectbox("Variable", numeric_cols)

            groups = df[group].unique()

            g1 = df[df[group]==groups[0]][var]
            g2 = df[df[group]==groups[1]][var]

            t,p = stats.ttest_ind(g1,g2)

            st.write("t statistic:",t)
            st.write("p value:",p)

            st.write("Interpretation")
            st.write(interpret_p(p))

# =====================================================
# ANOVA
# =====================================================

    elif menu == "ANOVA":

        st.subheader("One Way ANOVA")

        y = st.selectbox("Dependent variable", numeric_cols)
        x = st.selectbox("Factor", categorical_cols)

        model = smf.ols(f"{y} ~ C({x})", data=df).fit()

        table = sm.stats.anova_lm(model)

        st.dataframe(table)

        p = table["PR(>F)"][0]

        st.write("Interpretation")
        st.write(interpret_p(p))

# =====================================================
# REGRESSION
# =====================================================

    elif menu == "Regression":

        st.subheader("Linear Regression")

        y = st.selectbox("Dependent variable", numeric_cols)
        X = st.multiselect("Independent variables", numeric_cols)

        if X:

            Xdata = df[X]
            ydata = df[y]

            model = LinearRegression()
            model.fit(Xdata, ydata)

            pred = model.predict(Xdata)

            r2 = r2_score(ydata,pred)

            st.write("R²:",r2)

            coef = pd.DataFrame({
                "Variable":X,
                "Coefficient":model.coef_
            })

            st.table(coef)

            st.write("Interpretation")
            st.write(interpret_r2(r2))

# =====================================================
# CHI SQUARE TESTS
# =====================================================

    elif menu == "Chi-Square Tests":

        st.subheader("Chi-Square Test of Independence")

        var1 = st.selectbox("Variable 1", categorical_cols)
        var2 = st.selectbox("Variable 2", categorical_cols)

        table = pd.crosstab(df[var1], df[var2])

        chi2,p,_,_ = stats.chi2_contingency(table)

        st.dataframe(table)

        st.write("Chi-square:",chi2)
        st.write("p value:",p)

        st.write("Interpretation")
        st.write(interpret_p(p))

else:

    st.info("Upload a dataset to begin analysis.")

# =====================================================
# TIME SERIES ANALYSIS
# =====================================================

elif analysis_category == "Time Series":

    st.subheader("Time Series Analysis")

    var = st.selectbox("Select Time Series Variable", numeric_cols)

    data = df[var]

    if method == "Time Series Plot":

        fig, ax = plt.subplots()
        ax.plot(data)
        ax.set_title("Time Series Plot")
        st.pyplot(fig)

        st.write("Interpretation")
        st.write("The plot shows the evolution of the variable over time.")

    elif method == "Moving Average":

        window = st.slider("Window Size",2,20,5)

        ma = data.rolling(window).mean()

        fig, ax = plt.subplots()
        ax.plot(data,label="Original")
        ax.plot(ma,label="Moving Average")
        ax.legend()

        st.pyplot(fig)

        st.write("Interpretation")
        st.write("Moving average smooths short-term fluctuations and reveals trends.")

    elif method == "Trend Estimation":

        t = np.arange(len(data))

        model = LinearRegression()
        model.fit(t.reshape(-1,1),data)

        trend = model.predict(t.reshape(-1,1))

        fig, ax = plt.subplots()
        ax.plot(data,label="Original")
        ax.plot(trend,label="Trend")
        ax.legend()

        st.pyplot(fig)

        st.write("Interpretation")
        st.write("Trend line indicates long-term direction of the series.")

# =====================================================
# QUALITY CONTROL
# =====================================================

elif analysis_category == "Quality Control":

    st.subheader("Statistical Quality Control")

    var = st.selectbox("Process Variable", numeric_cols)

    data = df[var]

    if method == "Control Chart (X-bar)":

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

        st.write("Interpretation")

        if any(data > UCL) or any(data < LCL):

            st.write("Process may be out of control.")

        else:

            st.write("Process appears to be in statistical control.")


    elif method == "Process Capability":

        LSL = st.number_input("Lower Specification Limit")
        USL = st.number_input("Upper Specification Limit")

        mean = np.mean(data)
        std = np.std(data)

        Cp = (USL-LSL)/(6*std)

        st.write("Process Capability Cp:",Cp)

        if Cp > 1.33:
            st.write("Process capability is good.")

        elif Cp > 1:
            st.write("Process capability is acceptable.")

        else:
            st.write("Process capability is poor.")


