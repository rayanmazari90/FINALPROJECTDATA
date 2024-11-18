# modules/stepwise_regression.py

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def stepwise_regression():
    st.header("Stepwise Regression for Feature Selection")

    if 'ml_data' not in st.session_state or 'models_summary' not in st.session_state:
        st.warning("Please run the Data Acquisition and Linear Regression steps first.")
        return

    ml_data = st.session_state['ml_data']
    tickers = st.session_state['tickers']
    sector_map = st.session_state['sector_map']
    macro_vars = st.session_state['macro_vars']
    models_summary = st.session_state['models_summary']  # Get the R-squared values from linear regression

    # Filter tickers based on R-squared threshold
    st.subheader("Filter Models Based on R² Value from Linear Regression")
    threshold = st.slider("Select R-squared threshold", min_value=0.0, max_value=1.0, value=0.7)
    high_r2_tickers = [ticker for ticker, summary in models_summary.items() if summary['R_squared'] > threshold]

    if not high_r2_tickers:
        st.write("No tickers meet the R-squared threshold.")
        return
    st.session_state['ticker_models'] = {}  # Initialize storage for all models

    for ticker in high_r2_tickers:


        y = ml_data[f'{ticker}_Returns']
        stock_specific_vars = [
            f'{ticker}_Debt_Equity',
            f'{ticker}_ROE',
            f'{ticker}_ROA',
            f'{ticker}_12M_Momentum',
            f'{ticker}_MA50',
            f'{ticker}_MA200',
            f'{ticker}_RSI'
        ]
        stock_specific_vars = [var for var in stock_specific_vars if var in ml_data.columns]
        sector_var = sector_map[ticker]
        X_vars = macro_vars + stock_specific_vars + ([sector_var] if sector_var in ml_data.columns else [])

        X = ml_data[X_vars].copy()
        X = X.dropna()
        y = y.loc[X.index]

        # Build initial model with all variables
        X_with_const = sm.add_constant(X)
        initial_model = sm.OLS(y, X_with_const).fit()
        initial_r_squared = initial_model.rsquared


        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Linear regression model for feature selection
        lr = LinearRegression()

        # Perform forward or backward stepwise regression
        sfs = SequentialFeatureSelector(
            lr,
            n_features_to_select="auto",
            direction="forward",
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1
        )

        
        sfs.fit(X_scaled, y)
        selected_features = X.columns[sfs.get_support()]

        # Build final model with selected features
        X_selected = X[selected_features]
        X_selected_const = sm.add_constant(X_selected)
        final_model = sm.OLS(y, X_selected_const).fit()

        # Store the final model details
        st.session_state['ticker_models'][ticker] = {
            'intercept': final_model.params.get('const', 0.0),
            'coefficients': final_model.params.drop('const', errors='ignore').to_dict(),
            'features': list(selected_features),
            'r_squared': final_model.rsquared
        }

        # Optional: Display summary for debugging


    st.success("Stepwise Regression completed for all tickers!")

    # Option to select a ticker from filtered tickers
    selected_ticker = st.selectbox("Select a ticker for stepwise regression", high_r2_tickers)

    if selected_ticker:
        y = ml_data[f'{selected_ticker}_Returns']
        stock_specific_vars = [
            f'{selected_ticker}_Debt_Equity',
            f'{selected_ticker}_ROE',
            f'{selected_ticker}_ROA',
            f'{selected_ticker}_12M_Momentum',
            f'{selected_ticker}_MA50',
            f'{selected_ticker}_MA200',
            f'{selected_ticker}_RSI'
        ]
        stock_specific_vars = [var for var in stock_specific_vars if var in ml_data.columns]
        sector_var = sector_map[selected_ticker]
        X_vars = macro_vars + stock_specific_vars + ([sector_var] if sector_var in ml_data.columns else [])

        X = ml_data[X_vars].copy()
        X = X.dropna()
        y = y.loc[X.index]

        # Build initial model with all variables
        X_with_const = sm.add_constant(X)
        initial_model = sm.OLS(y, X_with_const).fit()
        initial_r_squared = initial_model.rsquared

        st.subheader("Initial Model Summary (Before Stepwise Regression):")
        model_summ= initial_model.summary().as_text()
        st.code(model_summ, language="text")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Linear regression model for feature selection
        lr = LinearRegression()

        # Option to select direction of stepwise regression
        direction = st.selectbox("Select stepwise regression direction", ["forward", "backward"])

        # Define the SequentialFeatureSelector
        if direction == "forward":
            sfs = SequentialFeatureSelector(
                lr,
                n_features_to_select="auto",
                direction='forward',
                scoring='neg_mean_squared_error',
                cv=5,
                n_jobs=-1
            )
        else:
            sfs = SequentialFeatureSelector(
                lr,
                n_features_to_select="auto",
                direction='backward',
                scoring='neg_mean_squared_error',
                cv=5,
                n_jobs=-1
            )

        # Perform stepwise regression
        with st.spinner("Performing stepwise regression..."):
            sfs.fit(X_scaled, y)
            selected_features = X.columns[sfs.get_support()]
            removed_features = X.columns[~sfs.get_support()]

        st.subheader("Selected Features After Stepwise Regression:")
        st.write(list(selected_features))
        st.subheader("Removed Features:")
        st.write(list(removed_features))

        # Build model with selected features
        X_selected = X[selected_features]
        X_selected_const = sm.add_constant(X_selected)
        final_model = sm.OLS(y, X_selected_const).fit()
        final_r_squared = final_model.rsquared

        st.subheader("Final Model Summary (After Stepwise Regression):")
        st.code(final_model.summary().as_text(), language="text")

        # Display R-squared comparison
        st.subheader("R² Value Comparison:")
        st.write(f"**Before Stepwise Regression:** R² = {initial_r_squared:.6f}")
        st.write(f"**After Stepwise Regression:** R² = {final_r_squared:.6f}")

        # Predict and evaluate
        predictions = final_model.predict(X_selected_const)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        st.write(f"Mean Absolute Error (MAE): {mae:.6f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

        st.subheader("Actual vs. Predicted Returns After Stepwise Regression")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Align actual and predicted values with the Date column
        date_column = ml_data.loc[y.index, 'Date']  # Extract the corresponding Date column
        y.index = date_column  # Align the index of y with the Date column
        predictions.index = date_column  # Align the index of predictions with the Date column

        # Plot the actual and predicted returns
        y.plot(ax=ax, label='Actual Returns', color='blue')
        predictions.plot(ax=ax, label='Predicted Returns', linestyle='--', color='orange')

        # Add title, labels, and legend
        ax.set_title(f'{selected_ticker} Actual vs. Predicted Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.legend()

        # Render the plot in Streamlit
        st.pyplot(fig)

        # Store the final model and selected features in session state
        st.session_state[f'{selected_ticker}_stepwise_model'] = final_model
        st.session_state[f'{selected_ticker}_selected_features'] = selected_features

    else:
        st.write("Please select a ticker to proceed.")
    # Add a single expander for detailed explanation
    with st.expander("Detailed Explanation of Stepwise Regression"):
        st.markdown("""
        ### Stepwise Regression: What We Do
        - **Purpose**: Select the most significant features for predicting stock returns using stepwise regression.
        - **Process**:
          1. Start with either no features (forward selection) or all features (backward elimination).
          2. Add or remove features one by one based on their statistical significance (p-values).
        - **Key Variables**:
          - **Macroeconomic Variables**: Economic indicators like GDP Growth, Unemployment Rate, etc.
          - **Stock-Specific Features**: Technical indicators (RSI, Moving Averages) and financial ratios (Debt-to-Equity, ROE, ROA).
          - **Sector Variables**: Performance of sector ETFs.

        ### Benefits:
        - Reduces overfitting by selecting only the most relevant variables.
        - Simplifies the model without compromising much predictive power.

        ### Outputs:
        - Final selected features for each stock.
        - Updated models stored in the session state for further use.
        """)