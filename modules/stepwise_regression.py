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

def backward_elimination(model, X, y, significance_level=0.05):
    """
    Perform backward elimination by iteratively removing the least significant variable.

    Parameters:
    - model: The fitted OLS model.
    - X: Feature DataFrame.
    - y: Target Series.
    - significance_level: Threshold for p-value to retain variables.

    Returns:
    - model: The final fitted OLS model with significant variables.
    - X: The DataFrame with only significant variables.
    - removed_features: List of variables removed during elimination.
    """
    removed_features = []  # Initialize list to track removed features
    while True:
        p_values = model.pvalues.drop('const', errors='ignore')
        max_pval = p_values.max()
        if max_pval > significance_level:
            excluded_var = p_values.idxmax()
            removed_features.append(excluded_var)  # Track the removed variable
            X = X.drop(columns=[excluded_var])
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()
        else:
            break
    return model, X, removed_features  # Return the list of removed features

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

        # Perform forward stepwise regression
        sfs = SequentialFeatureSelector(
            lr,
            n_features_to_select="auto",
            direction="forward",
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1
        )

        sfs.fit(X_scaled, y)
        selected_features_stepwise = X.columns[sfs.get_support()].tolist()
        removed_features_stepwise = X.columns[~sfs.get_support()].tolist()

        # Build initial model with selected features
        X_selected = X[selected_features_stepwise]
        X_selected_const = sm.add_constant(X_selected)
        initial_selected_model = sm.OLS(y, X_selected_const).fit()

        # Perform backward elimination based on p-values and capture removed features
        final_model, X_final, removed_features_backward = backward_elimination(initial_selected_model, X_selected, y)

        # Store the final model details along with removed features and model summary
        st.session_state['ticker_models'][ticker] = {
            'intercept': final_model.params.get('const', 0.0),
            'coefficients': final_model.params.drop('const', errors='ignore').to_dict(),
            'features': list(X_final.columns),
            'r_squared': final_model.rsquared,
            'removed_features_backward': removed_features_backward,
            'removed_features_stepwise': removed_features_stepwise,
            'model_summary': final_model.summary().as_text()
        }

    st.success("Stepwise Regression completed for all tickers!")

    # Option to select a ticker from filtered tickers
    selected_ticker = st.selectbox("Select a ticker for stepwise regression", high_r2_tickers)

    if selected_ticker:
        # Retrieve model details from session state
        model_details = st.session_state['ticker_models'][selected_ticker]

        y = ml_data[f'{selected_ticker}_Returns']
        X = ml_data[macro_vars + stock_specific_vars + ([sector_map[selected_ticker]] if sector_map[selected_ticker] in ml_data.columns else [])].copy()
        X = X.dropna()
        y = y.loc[X.index]

        st.subheader("Final Model Summary (After Stepwise Regression and Backward Elimination):")
        st.code(model_details['model_summary'], language="text")

        st.subheader("Selected Features:")
        st.write(model_details['features'])

        st.subheader("Removed Features During Stepwise Regression:")
        st.write(model_details.get('removed_features_stepwise', []), model_details.get('removed_features_backward', []))


        # Display R-squared comparison
        st.subheader("R² Value Comparison:")
        st.write(f"**After Stepwise Regression and Backward Elimination:** R² = {model_details['r_squared']:.6f}")

        # Reconstruct X_final from stored features
        available_features = [feature for feature in model_details['features'] if feature in X.columns]
        X_final = X[available_features]
        X_final_const = sm.add_constant(X_final)
        predictions = X_final_const.dot(
            [model_details['intercept']] + [model_details['coefficients'][col] for col in X_final.columns]
        )

        #mae = mean_absolute_error(y, predictions)
        #rmse = np.sqrt(mean_squared_error(y, predictions))
        #st.write(f"**Mean Absolute Error (MAE):** {mae:.6f}")
        #st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.6f}")

        st.subheader("Actual vs. Predicted Returns After Stepwise Regression")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Align actual and predicted values with the Date column
        if 'Date' in ml_data.columns:
            date_column = ml_data.loc[y.index, 'Date']  # Extract the corresponding Date column
            y_aligned = y.copy()
            predictions_aligned = predictions.copy()
            y_aligned.index = date_column
            predictions_aligned.index = date_column
        else:
            y_aligned = y
            predictions_aligned = predictions

        # Plot the actual and predicted returns
        y_aligned.plot(ax=ax, label='Actual Returns', color='blue')
        predictions_aligned.plot(ax=ax, label='Predicted Returns', linestyle='--', color='orange')

        # Add title, labels, and legend
        ax.set_title(f'{selected_ticker} Actual vs. Predicted Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.legend()

        # Render the plot in Streamlit
        st.pyplot(fig)

    else:
        st.write("Please select a ticker to proceed.")

    # Add a single expander for detailed explanation
    with st.expander("Detailed Explanation of Stepwise Regression"):
        st.markdown("""
        ### Stepwise Regression: What We Do
        - **Purpose**: Select the most significant features for predicting stock returns using stepwise regression followed by backward elimination.
        - **Process**:
          1. **Stepwise Selection**: Start with no features (forward selection) and iteratively add predictors based on minimizing MSE.
          2. **Backward Elimination**: After stepwise selection, iteratively remove the least significant predictors (highest p-values) until all remaining variables are statistically significant.
        - **Key Variables**:
          - **Macroeconomic Variables**: Economic indicators like GDP Growth, Unemployment Rate, etc.
          - **Stock-Specific Features**: Technical indicators (RSI, Moving Averages) and financial ratios (Debt-to-Equity, ROE, ROA).
          - **Sector Variables**: Performance of sector ETFs.

        ### Benefits:
        - Reduces overfitting by selecting only the most relevant variables.
        - Ensures statistical significance of predictors, enhancing model interpretability.
        - Simplifies the model without compromising much predictive power.

        ### Outputs:
        - Final selected features for each stock.
        - Updated models stored in the session state for further use.
        """)
