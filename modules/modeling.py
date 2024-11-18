import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def linear_regression_models():
    st.header("Linear Regression Models")

    if 'ml_data' not in st.session_state:
        st.warning("Please run the Data Acquisition step first.")
        return

    ml_data = st.session_state['ml_data']
    tickers = st.session_state['tickers']
    sector_map = st.session_state['sector_map']
    macro_vars = st.session_state['macro_vars']

    # Dictionary to store predictions and model summaries for each stock
    predictions_dict = {}
    models_summary = {}

    # Loop through each stock to build individual regression models and store predictions
    for ticker in tickers:
        y = ml_data[f'{ticker}_Returns']
        stock_specific_vars = [f'{ticker}_Debt_Equity', f'{ticker}_ROE', f'{ticker}_ROA',
                               f'{ticker}_12M_Momentum', f'{ticker}_MA50', f'{ticker}_MA200', f'{ticker}_RSI']
        stock_specific_vars = [var for var in stock_specific_vars if var in ml_data.columns]
        sector_var = sector_map[ticker]
        X_vars = macro_vars + stock_specific_vars + ([sector_var] if sector_var in ml_data.columns else [])
        X = ml_data[X_vars]

        # Align indices of y and X
        X = X.loc[y.index]
        y = y.loc[X.index]

        X = sm.add_constant(X)

        if not X.empty and not y.empty:
            model = sm.OLS(y, X).fit()
            predictions = model.predict(X)
            predictions_dict[ticker] = {'actual': y, 'predicted': predictions, 'model': model}
            models_summary[ticker] = {'R_squared': model.rsquared}

    # Store predictions in session state
    st.session_state['predictions'] = predictions_dict

    # Display models with R²
    models_df = pd.DataFrame.from_dict(models_summary, orient='index')

    st.write("Model R-squared values:")
    st.dataframe(models_df)

    # Filter models with R² > 0.7
    threshold = st.slider("Select R-squared threshold", min_value=0.0, max_value=1.0, value=0.7)
    high_r2_models = models_df[models_df['R_squared'] > threshold]
    st.write(f"Models with R² > {threshold}:")
    st.dataframe(high_r2_models)

    if not high_r2_models.empty:
        # Ensure the available tickers for selection match the filtered models
        available_tickers = high_r2_models.index.intersection(predictions_dict.keys())
        if not available_tickers.empty:
            selected_ticker = st.selectbox("Select a ticker to view model details", available_tickers)
            if selected_ticker:
                st.write(f"Model Summary for {selected_ticker}:")
                model_summary = predictions_dict[selected_ticker]['model'].summary().as_text()
                st.code(model_summary, language='text')

                # Calculate evaluation metrics
                y_actual = predictions_dict[selected_ticker]['actual']
                y_pred = predictions_dict[selected_ticker]['predicted']
                mae = mean_absolute_error(y_actual, y_pred)
                rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

                # Plot actual vs. predicted returns
                fig, ax = plt.subplots(figsize=(12, 6))
                y_actual.index = pd.to_datetime(ml_data.Date)  # Ensure dates are datetime
                y_pred.index =pd.to_datetime(ml_data.Date)  # Align predicted values with actual dates
                y_actual.plot(ax=ax, label='Actual Returns')
                y_pred.plot(ax=ax, label='Predicted Returns', linestyle='--')
                ax.set_title(f"{selected_ticker} - Actual vs Predicted Returns")
                ax.set_xlabel("Date")
                ax.set_ylabel("Returns")
                ax.legend()
                st.pyplot(fig)
        else:
            st.warning("No tickers available for selection.")
    else:
        st.write("No models meet the R² threshold.")

    with st.expander("Detailed Explanation of Linear Regression Models"):
        st.markdown("""
        ### What We Do in This Section:
        - **Filtering Stocks**: Run individual linear regression models for each stock and filter based on R² values.
        - **Building Multiple Models**: Create separate regression models for all stocks using macroeconomic, stock-specific, and sector variables.
        - **Evaluation**: Identify and prioritize high-performing models (R² > selected threshold) for downstream tasks like portfolio optimization.

        ### Key Steps:
        1. **Independent Variables**:
            - Macroeconomic indicators: GDP Growth, Unemployment Rate, CPI, etc.
            - Sector trends: Returns of the sector ETF for the stock.
        2. **Dependent Variable**:
            - Stock returns for each stock, calculated as percentage changes in adjusted closing prices.

        ### Insights:
        - High R² values indicate strong models but must balance with avoiding overfitting.
        - Models with R² > 0.7 are filtered for downstream optimization.
        - Visualizations, such as actual vs. predicted returns, help evaluate the models' predictive performance.

        ### Limitations:
        - Multicollinearity in independent variables could affect the regression model's stability.
        - Linear regression assumes a linear relationship, which may not fully capture complex market dynamics.
        """)

    st.session_state['models_summary'] = models_summary