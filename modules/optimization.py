# modules/optimization.py

import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp


def investment_optimization():
    st.header("Portfolio Optimization Using Linear Programming (95% Confidence Level)")

    # Ensure necessary data is available
    if 'ml_data' not in st.session_state or 'ticker_models' not in st.session_state:
        st.warning("Please run the Data Acquisition and Stepwise Regression steps first.")
        return

    ml_data = st.session_state['ml_data']
    ticker_models = st.session_state['ticker_models']

    # Input investment constraints
    st.subheader("Input Investment Constraints")
    total_capital = st.number_input("Total Capital to Invest", min_value=1000, value=10000, step=100)
    max_loss = st.number_input("Maximum Acceptable Loss (%)", min_value=0, max_value=100, value=10) / 100
    #max_weight = st.number_input("Maximum Weight per Stock (%)", min_value=0.01, max_value=1.0, value=0.25)
    max_weight=0.25
    # Select the last row for predictions
    last_row = ml_data.iloc[-1:]

    # Initialize containers for optimization
    selected_tickers = []
    predicted_returns = []
    risks = []
    lower_bounds = []
    upper_bounds = []

    st.subheader("Model Equations and Adjustments (Dropdowns)")

    # For each ticker, create a dropdown with its model details
    for ticker, model_details in ticker_models.items():
        intercept = model_details['intercept']
        coefficients = model_details['coefficients']  # Dictionary of feature: coefficient

        adjusted_coefficients = []
        adjusted_inputs = []

        with st.expander(f"Adjust Linear Regression Model for {ticker}", expanded=False):
            st.markdown(f"### {ticker} Model Equation")

            # Intercept row
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                intercept_adjusted = st.number_input(
                    f"Adjust Intercept ({ticker})",
                    value=float(intercept),
                    format="%.8f",
                    key=f"{ticker}_intercept"
                )
            with col2:
                st.write("Intercept")
            with col3:
                st.write(f"Coefficient: {intercept_adjusted:.8f}")

            adjusted_coefficients.append(intercept_adjusted)
            adjusted_inputs.append(1.0)  # Constant for the intercept

            # Feature rows
            for feature, coeff in coefficients.items():
                if feature not in last_row.columns:
                    st.warning(f"Feature '{feature}' not found in the dataset for {ticker}. Skipping.")
                    continue

                feature_value = float(last_row[feature].values[0])

                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    adjusted_coeff = st.number_input(
                        f"Adjust Coefficient ({feature}, {ticker})",
                        value=float(coeff),
                        format="%.8f",
                        key=f"{ticker}_{feature}_coeff"
                    )
                with col2:
                    adjusted_input = st.number_input(
                        f"Adjust Input ({feature}, {ticker})",
                        value=feature_value,
                        format="%.8f",
                        key=f"{ticker}_{feature}_input"
                    )
                with col3:
                    st.write(f"Coefficient: {adjusted_coeff:.8f}, Input: {adjusted_input:.8f}")

                adjusted_coefficients.append(adjusted_coeff)
                adjusted_inputs.append(adjusted_input)

            # Compute predicted return
            predicted_return = np.dot(adjusted_coefficients, adjusted_inputs)

            # Compute historical risk (std deviation)
            ticker_returns = ml_data[f'{ticker}_Returns'].dropna()
            std_dev = ticker_returns.std()

            # 95% Confidence Interval
            lower_bound = predicted_return - 1.96 * std_dev
            upper_bound = predicted_return + 1.96 * std_dev

            # Store results
            predicted_returns.append(predicted_return)
            risks.append(std_dev)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            selected_tickers.append(ticker)

            st.markdown(f"**Predicted Return for {ticker}:** {predicted_return:.8f}")
            st.markdown(f"**95% Confidence Interval:** [{lower_bound:.8f}, {upper_bound:.8f}]")
            st.markdown(f"**Risk (Standard Deviation):** {std_dev:.8f}")

    if not selected_tickers:
        st.warning("No tickers with valid regression models available.")
        return

    # Validate data for optimization
    st.write("Predicted Returns (Last Row):", predicted_returns)
    #st.write("Risks (Standard Deviation):", risks)
    st.write("Max Loss Constraint:", max_loss)

    # Ensure predicted returns and risk values are valid
    if not all(np.isfinite(predicted_returns)) or not all(np.isfinite(risks)):
        st.error("Invalid predicted returns or risk values. Please check your data.")
        return

    # Convert lists to numpy arrays
    predicted_returns = np.array(predicted_returns)
    risks = np.array(risks)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Create optimization variables
    weights = cp.Variable(len(selected_tickers))

    # Define the objective function: maximize the lower bound of the expected return (conservative approach)
    objective = cp.Maximize(lower_bounds @ weights)

    # Define constraints
    constraints = [
        cp.sum(weights) == 1,              # Fully invested portfolio
        weights >= 0,                      # No short selling
        weights <= max_weight,             # Diversification constraint
        cp.norm(risks * weights, 2) <= max_loss  # Risk constraint based on std deviation
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
        optimized_weights = weights.value
        allocation = {selected_tickers[i]: optimized_weights[i] * total_capital for i in range(len(selected_tickers))}
        allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Allocation'])
        allocation_df.index.name = 'Ticker'
        st.table(allocation_df)

        # Calculate key metrics
        expected_return = (predicted_returns @ optimized_weights) * total_capital
        portfolio_risk = np.sqrt(np.sum((risks * optimized_weights) ** 2))

        # Confidence interval for portfolio return
        portfolio_lower_bound = (lower_bounds @ optimized_weights) * total_capital
        portfolio_upper_bound = (upper_bounds @ optimized_weights) * total_capital

        st.subheader("Key Metrics")
        st.write(f"**Expected Portfolio Return:** ${expected_return:.2f}")
        st.write(f"**Portfolio Return Range (95% Confidence Interval):** [${portfolio_lower_bound:.2f}, ${portfolio_upper_bound:.2f}]")
        #st.write(f"**Portfolio Risk (Standard Deviation):** {portfolio_risk:.2%}")

    else:
        st.error(f"Optimization failed. Status: {problem.status}")

    # Add a dropdown to explain the optimization process
    # Add a dropdown to explain the optimization process
    """ 
    with st.expander("Explanation and Formulas"):
        st.markdown(
        ### Procedure Explanation
        1. **Linear Regression Models**: Each stock uses a linear regression model to predict returns based on selected features.
        2. **Expected Return**: Predicted returns are computed for the last row of data using the regression coefficients and inputs.
        3. **Risk (Standard Deviation)**: Historical standard deviation of returns is used to estimate risk.
        4. **Confidence Intervals**: A 95% confidence interval for each stock's return is calculated.
        5. **Optimization**: The portfolio is optimized using Linear Programming:
            - Maximize the lower bound of portfolio return.
            - Constrain risk based on the standard deviation of returns and user-defined maximum loss.
        )

        st.markdown("### Optimization Formula")
        st.markdown("**Objective Function:**")
        st.latex(r"Maximize \ \sum_{i=1}^{n} (\text{Lower Bound}_i \cdot \text{Weight}_i)")

        st.markdown("**Constraints:**")
        st.latex(r"\sum_{i=1}^{n} \text{Weight}_i = 1 \quad \text{(Fully Invested Portfolio)}")
        st.latex(r"\text{Weight}_i \geq 0 \quad \forall i \quad \text{(No Short Selling)}")
        st.latex(r"\text{Weight}_i \leq \text{Max Weight per Stock} \quad \forall i \quad \text{(Diversification Constraint)}")
        st.latex(r"\sqrt{\sum_{i=1}^{n} (\text{Risk}_i \cdot \text{Weight}_i)^2} \leq \text{Max Loss} \quad \text{(Risk Constraint)}")
    """ 