import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp

def investment_optimization():
    st.header("Portfolio Optimization with Short Selling and Risk Control")

    # Ensure necessary data is available
    if 'ml_data' not in st.session_state or 'ticker_models' not in st.session_state:
        st.warning("Please run the Data Acquisition and Stepwise Regression steps first.")
        return

    ml_data = st.session_state['ml_data']
    ticker_models = st.session_state['ticker_models']

    # Input investment constraints
    st.subheader("Input Investment Constraints")
    total_capital = st.number_input("Total Capital to Invest", min_value=1000, value=10000, step=100)
    max_loss = st.number_input("Maximum Allowable Loss (%)", min_value=0, max_value=100, value=10) / 100

    # Select the last row for predictions
    last_row = ml_data.iloc[-1:]

    selected_tickers = []
    predicted_returns = []
    lower_bounds = []
    upper_bounds = []

    st.subheader("Model Equations and Adjustments (Dropdowns)")

    for ticker, model_details in ticker_models.items():
        intercept = model_details['intercept']
        coefficients = model_details['coefficients']

        adjusted_coefficients = []
        adjusted_inputs = []

        with st.expander(f"Adjust Linear Regression Model for {ticker}", expanded=False):
            st.markdown(f"### {ticker} Model Equation")

            intercept_adjusted = st.number_input(
                f"Adjust Intercept ({ticker})", value=float(intercept), format="%.8f", key=f"{ticker}_intercept"
            )
            adjusted_coefficients.append(intercept_adjusted)
            adjusted_inputs.append(1.0)

            # Feature rows
            for feature, coeff in coefficients.items():
                if feature not in last_row.columns:
                    st.warning(f"Feature '{feature}' not found in the dataset for {ticker}. Skipping.")
                    continue

                feature_value = float(last_row[feature].values[0])

                adjusted_coeff = st.number_input(
                    f"Adjust Coefficient ({feature}, {ticker})",
                    value=float(coeff),
                    format="%.8f",
                    key=f"{ticker}_{feature}_coeff"
                )
                adjusted_input = st.number_input(
                    f"Adjust Input ({feature}, {ticker})",
                    value=feature_value,
                    format="%.8f",
                    key=f"{ticker}_{feature}_input"
                )

                adjusted_coefficients.append(adjusted_coeff)
                adjusted_inputs.append(adjusted_input)

            predicted_return = np.dot(adjusted_coefficients, adjusted_inputs)

            # Calculate risk metrics
            ticker_returns = ml_data[f'{ticker}_Returns'].dropna()
            std_dev = ticker_returns.std()

            lower_bound = predicted_return - 1.96 * std_dev
            upper_bound = predicted_return + 1.96 * std_dev

            predicted_returns.append(predicted_return)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            selected_tickers.append(ticker)

    if not selected_tickers:
        st.warning("No tickers with valid regression models available.")
        return

    predicted_returns = np.array(predicted_returns)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Create optimization variables for direct capital allocation (allowing short selling)
    capital_allocations = cp.Variable(len(selected_tickers))

    # Objective function: maximize total expected return (capital_allocations * predicted returns)
    objective = cp.Maximize(predicted_returns @ capital_allocations)

    # Constraints
    constraints = [
        cp.sum(capital_allocations) == total_capital,                     # Total capital constraint
        (lower_bounds @ capital_allocations) >= -(max_loss * total_capital),  # Loss constraint using lower bound
        capital_allocations >= -0.5 * total_capital,                      # Lower bound for short exposure
        capital_allocations <= 1.5 * total_capital                        # Upper bound for each allocation
    ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.SCS, verbose=True)
    except Exception as e:
        st.error(f"An error occurred while solving the optimization problem: {e}")
        return

    if problem.status == 'optimal':
        st.success("Portfolio Optimization Completed Successfully!")
        st.subheader("Optimized Portfolio Allocation")
        st.write("Negatives values allocation means we are shorting")

        optimized_allocations = capital_allocations.value
        allocation = {selected_tickers[i]: optimized_allocations[i] for i in range(len(selected_tickers))}
        allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Capital Allocation'])
        allocation_df.index.name = 'Ticker'
        st.table(allocation_df)

        # Calculate key metrics
        expected_portfolio_return = predicted_returns @ optimized_allocations
        portfolio_lower_bound = lower_bounds @ optimized_allocations
        portfolio_upper_bound = upper_bounds @ optimized_allocations

        st.subheader("Key Metrics")
        st.write(f"**Expected Portfolio Return:** ${expected_portfolio_return:.2f}")
        st.write(f"**Portfolio Return Range (95% Confidence Interval):** [${portfolio_lower_bound:.2f}, ${portfolio_upper_bound:.2f}]")
        st.write(f"**Total Risk-Based Loss (Lower Bound):** ${-portfolio_lower_bound:.2f} (Max allowed: ${max_loss * total_capital:.2f})")

    else:
        st.error(f"Optimization failed. Status: {problem.status}")
