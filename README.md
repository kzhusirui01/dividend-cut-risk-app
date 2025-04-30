# 📉 Dividend Cut Risk Predictor

This is an interactive Streamlit app that predicts the probability of a dividend cut for U.S. public companies, based on:

- **Sector classification** (e.g., Energy, Financials)
- **Company-specific financial ratios** (e.g., Quick Ratio, Payout Ratio)
- **Macroeconomic indicators** (GDP, Unemployment Rate, Prior-Year Inflation)

The model integrates logistic regression models trained on Russell 3000 dividend history and financial statement data. Each component was tested for statistical significance, with fallback defaults for missing data.

🧠 Built using:
- Python (scikit-learn, pandas)
- Streamlit
- joblib for model persistence

📊 Try the app: [Streamlit App Link Here](https://dividend-cut-risk-app-no3ww3xttqmjev7pmuvp8f.streamlit.app/)

---

### How to Use

1. Select a model (Sector Only, Micro Only, Macro Only, or Combined)
2. Input relevant company and macro data
3. Click **Predict** to view the estimated dividend cut risk (High, Medium, or Low)

---

### About

Developed as part of an analytics project at Columbia University in collaboration with RBC Capital Markets.  
