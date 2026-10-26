import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="2020 Turnout Explorer", layout="wide")
st.title("2020 Turnout Explorer")
st.caption(
    "Upload a CSV, explore a column with a chart, then fit a linear regression "
    "with two inputs and a target. Built with the 2020 US general election turnout rates by state."
)

st.info(
    "**What is this model for, and what happens when it is wrong?** It gives a rough estimate of "
    "a state's total ballots from its eligible population and turnout rate, useful for quick planning, "
    "but when it is wrong a campaign could misjudge how many voters a state holds, so check it "
    "against official counts before relying on it."
)


def clean_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text like '2,306,587' or '62.6%' into numbers when most of a column parses."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            text = df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
            converted = pd.to_numeric(text, errors="coerce")
            non_empty = df[col].notna().sum()
            if non_empty > 0 and converted.notna().sum() >= 0.8 * non_empty:
                df[col] = converted
    return df


uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    raw = pd.read_csv(uploaded_file)
    df = clean_numbers(raw)

    # The national total row would distort a state-level model.
    first_col = df.columns[0]
    if df[first_col].astype(str).str.strip().eq("United States").any():
        df = df[df[first_col].astype(str).str.strip() != "United States"]
        st.caption("Removed the 'United States' total row so only states are used.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("This file needs at least three numeric columns.")
        st.stop()

    # 1. Preview
    st.subheader("1. Preview")
    st.dataframe(df.head(10), hide_index=True, use_container_width=True)
    st.caption(f"{len(df)} rows, {len(df.columns)} columns, {len(numeric_cols)} numeric.")

    # 2. Chart
    st.subheader("2. Chart")
    c1, c2 = st.columns(2)
    chart_type = c1.selectbox("Chart type", ["Histogram", "Line", "Scatter"])
    column = c2.selectbox("Column", numeric_cols)
    label_col = first_col

    if chart_type == "Histogram":
        fig = px.histogram(df, x=column, nbins=15, title=f"Distribution of {column}")
    elif chart_type == "Line":
        fig = px.line(df.reset_index(drop=True), y=column, hover_name=label_col, title=f"{column} by row order")
        fig.update_xaxes(title="Row")
    else:
        other_options = [c for c in numeric_cols if c != column]
        other = st.selectbox("Compare against", other_options)
        fig = px.scatter(df, x=other, y=column, hover_name=label_col, title=f"{column} versus {other}")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Regression
    st.subheader("3. Predict with linear regression")
    default_inputs = [c for c in numeric_cols if "VEP" in c][:2]
    target_default = next((c for c in numeric_cols if "Ballots" in c), numeric_cols[-1])

    target = st.selectbox("Target: what you want to predict", numeric_cols, index=numeric_cols.index(target_default))
    input_options = [c for c in numeric_cols if c != target]
    inputs = st.multiselect(
        "Inputs: choose exactly two",
        input_options,
        default=[c for c in default_inputs if c in input_options][:2],
        max_selections=2,
    )

    if len(inputs) != 2:
        st.warning("Choose exactly two input columns.")
        st.stop()

    data = df[inputs + [target]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in all three columns.")
        st.stop()

    # Same column names and order for training and prediction.
    X = data[inputs]
    y = data[target]
    model = LinearRegression().fit(X, y)

    m1, m2, m3 = st.columns(3)
    m1.metric("R2 on this data", f"{model.score(X, y):.3f}")
    m2.metric(f"Effect of {inputs[0]}", f"{model.coef_[0]:,.4g}")
    m3.metric(f"Effect of {inputs[1]}", f"{model.coef_[1]:,.4g}")

    st.markdown("**Try it:** type values for the inputs.")
    v1, v2 = st.columns(2)
    val1 = v1.number_input(inputs[0], value=float(data[inputs[0]].median()))
    val2 = v2.number_input(inputs[1], value=float(data[inputs[1]].median()))

    new_row = pd.DataFrame([[val1, val2]], columns=inputs)
    prediction = model.predict(new_row)[0]
    st.metric(f"Predicted {target}", f"{prediction:,.1f}")

    for name, val in zip(inputs, (val1, val2)):
        low, high = data[name].min(), data[name].max()
        if val < low or val > high:
            st.warning(
                f"{name} = {val:,.2f} is outside the range the model learned from "
                f"({low:,.2f} to {high:,.2f}). Treat the prediction with caution."
            )
else:
    st.info("Upload a CSV to get started. Try the sample file `turnout_2020.csv` from the repository.")
