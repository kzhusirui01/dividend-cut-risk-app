import streamlit as st
import pandas as pd
import joblib

# --- Load models and feature info ---
sector_logreg, sector_cols = joblib.load("sector_only_logreg.pkl")
micro_model, micro_cols, micro_means = joblib.load("micro_logreg.pkl")
macro_model, macro_cols, macro_means = joblib.load("macro_logreg.pkl")

# --- Label mapping for micro inputs ---
label_map = {
    "quickRatio": "Quick Ratio",
    "currentRatio": "Current Ratio",
    "operatingCycle": "Operating Cycle",
    "cashConversionCycle": "Cash Conversion Cycle",
    "cashFlowToDebtRatio": "Cash Flow to Debt Ratio",
    "cashFlowCoverageRatios": "Cash Flow Coverage Ratio",
    "daysOfInventoryOutstanding": "Days Inventory Outstanding",
    "daysOfPayablesOutstanding": "Days Payables Outstanding",
    "debtEquityRatio": "Debt-to-Equity Ratio",
    "interestCoverage": "Interest Coverage Ratio",
    "dividendYield": "Dividend Yield",
    "payoutRatio": "Payout Ratio",
    "freeCashFlowOperatingCashFlowRatio": "Free Cash Flow / Operating Cash Flow",
    "returnOnEquity": "Return on Equity"
}

# --- Formula dictionary for micro inputs ---
formula_dict = {
    "quickRatio": "Quick\\ Ratio = \\frac{\\text{Current Assets} - \\text{Inventories}}{\\text{Current Liabilities}}",
    "currentRatio": "Current\\ Ratio = \\frac{\\text{Current Assets}}{\\text{Current Liabilities}}",
    "operatingCycle": "Operating\\ Cycle = Days\\ Inventory\\ Outstanding + Days\\ Sales\\ Outstanding",
    "cashConversionCycle": "CCC = Operating\\ Cycle - Days\\ Payable\\ Outstanding",
    "cashFlowToDebtRatio": "CF\\ to\\ Debt = \\frac{\\text{Operating Cash Flow}}{\\text{Total Debt}}",
    "cashFlowCoverageRatios": "CF\\ Coverage = \\frac{OCF + Interest + Taxes}{Interest}",
    "daysOfInventoryOutstanding": "DIO = \\frac{\\text{Avg Inventory}}{COGS} \\times 365",
    "daysOfPayablesOutstanding": "DPO = \\frac{\\text{Avg Payables}}{COGS} \\times 365",
    "debtEquityRatio": "D/E = \\frac{\\text{Total Liabilities}}{\\text{Shareholder Equity}}",
    "interestCoverage": "IC = \\frac{EBIT}{Interest}",
    "dividendYield": "Dividend\\ Yield = \\frac{\\text{Annual DPS}}{\\text{Price per Share}}",
    "payoutRatio": "Payout\\ Ratio = \\frac{Dividends}{Net Income}",
    "freeCashFlowOperatingCashFlowRatio": "FCF / OCF = \\frac{Free Cash Flow}{Operating Cash Flow}",
    "returnOnEquity": "ROE = \\frac{Net Income}{Shareholder Equity}"
}

# --- Label mapping for macro inputs ---
macro_label_map = {
    "GDP": "GDP (Quarterly)",
    "unemployment": "Unemployment (Yearly)",
    "inflation_lag": "Previous Year's Inflation Rate (Yearly)"
}

# --- Set of insignificant micro ratios ---
insignificant_ratios = {
    "debtEquityRatio",
    "interestCoverage",
    "dividendYield",
    "payoutRatio",
    "returnOnEquity",
    "freeCashFlowOperatingCashFlowRatio"
}

# --- Streamlit UI setup ---
st.set_page_config(page_title="Dividend Cut Risk App", layout="centered")
st.title("📉 Dividend Cut Risk Predictor")
st.write("Input sector, company financials, and/or macroeconomic data to estimate dividend cut risk.")

# --- Model selection ---
model_choice = st.radio("Choose Model Type:", [
    "Sector Only",
    "Micro Only",
    "Macro Only",
    "Combined (Sector + Micro + Macro)"
])

# --- Sector input ---
if model_choice in ["Sector Only", "Combined (Sector + Micro + Macro)"]:
    sector_input = st.selectbox("Select Sector:", ["(Baseline) Communication Services"] + sector_cols)
    significant_sectors = ["Financials", "Energy", "Information Technology"]
    if sector_input not in significant_sectors:
        st.info("ℹ️ This sector was not found statistically significant. Prediction may carry lower confidence.")

# --- Micro inputs ---
if model_choice in ["Micro Only", "Combined (Sector + Micro + Macro)"]:
    st.subheader("Company Financial Ratios")
    st.info("ℹ️ Some financial ratios included in this model were not found statistically significant in univariate testing. "
            "Their inclusion is based on domain relevance and may affect prediction confidence.")

    micro_inputs = {}
    for var in micro_cols:
        label = label_map.get(var, var)

        with st.expander(f"{label} – ℹ️ Click to see formula"):
            if var in formula_dict:
                st.latex(formula_dict[var])

        micro_inputs[var] = st.number_input(f"{label}", value=0.0, format="%.4f")

        if var in insignificant_ratios:
            st.caption("_Note: This ratio was not found statistically significant. Included for relevance._")

    st.markdown("*If left blank or 0, the model will use average values from training data.*")

# --- Macro inputs ---
if model_choice in ["Macro Only", "Combined (Sector + Micro + Macro)"]:
    st.subheader("Macroeconomic Indicators")
    st.markdown("""
**Indicator Descriptions**  
• **GDP (Quarterly)**: GDP growth rate for the latest quarter (% QoQ)  
• **Unemployment (Yearly)**: National unemployment rate for the current year (%)  
• **Previous Year's Inflation Rate (Yearly)**: Inflation rate for the *prior calendar year* (%)  
<br>
*If left blank or 0, the model will use average values from training data.*
""", unsafe_allow_html=True)

    macro_inputs = {}
    for i, var in enumerate(macro_cols):
        label = macro_label_map.get(var, var)
        macro_inputs[var] = st.number_input(f"{label}", value=0.0, format="%.4f")

# --- Prediction logic ---
if st.button("Predict Dividend Cut Risk"):
    if model_choice == "Sector Only":
        sec_input = {col: 0 for col in sector_cols}
        if sector_input in sector_cols:
            sec_input[sector_input] = 1
        sector_df = pd.DataFrame([sec_input])[sector_cols]
        prob = sector_logreg.predict_proba(sector_df)[0][1]

    elif model_choice == "Micro Only":
        micro_vals = {
            col: (micro_inputs[col] if micro_inputs[col] != 0.0 else micro_means[i])
            for i, col in enumerate(micro_cols)
        }
        micro_df = pd.DataFrame([micro_vals])[micro_cols]
        prob = micro_model.predict_proba(micro_df)[0][1]

    elif model_choice == "Macro Only":
        macro_vals = {
            col: (macro_inputs[col] if macro_inputs[col] != 0.0 else macro_means[i])
            for i, col in enumerate(macro_cols)
        }
        macro_df = pd.DataFrame([macro_vals])[macro_cols]
        prob = macro_model.predict_proba(macro_df)[0][1]

    else:  # Combined model
        st.warning("⚠️ Using all three models (equal weight average).")

        # Sector
        sec_input = {col: 0 for col in sector_cols}
        if sector_input in sector_cols:
            sec_input[sector_input] = 1
        sector_df = pd.DataFrame([sec_input])[sector_cols]
        prob_sec = sector_logreg.predict_proba(sector_df)[0][1]

        # Micro
        micro_vals = {
            col: (micro_inputs[col] if micro_inputs[col] != 0.0 else micro_means[i])
            for i, col in enumerate(micro_cols)
        }
        micro_df = pd.DataFrame([micro_vals])[micro_cols]
        prob_micro = micro_model.predict_proba(micro_df)[0][1]

        # Macro
        macro_vals = {
            col: (macro_inputs[col] if macro_inputs[col] != 0.0 else macro_means[i])
            for i, col in enumerate(macro_cols)
        }
        macro_df = pd.DataFrame([macro_vals])[macro_cols]
        prob_macro = macro_model.predict_proba(macro_df)[0][1]

        # Average
        prob = (prob_sec + prob_micro + prob_macro) / 3

    # --- Display result ---
    if prob >= 0.75:
        st.error(f"🔴 High Risk: {prob:.1%} chance of dividend cut")
    elif prob >= 0.4:
        st.warning(f"🟠 Medium Risk: {prob:.1%} chance of dividend cut")
    else:
        st.success(f"🟢 Low Risk: {prob:.1%} chance of dividend cut")
