# modules/data_acquisition.py
import streamlit as st
import pandas as pd
import os
import warnings
import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

def data_acquisition():
    st.header("Data Acquisition and Preprocessing")
    
    # User inputs for date range
    start_date = st.date_input("Start Date", datetime.date(2013, 1, 1))
    end_date = st.date_input("End Date", datetime.date(2023, 10, 31))
    
    # Alpha Vantage API key input (simulated, not used in this case)
    api_key = st.text_input("Enter your Alpha Vantage API key (simulated input)", type="password")
    if not api_key:
        st.warning("Please enter your Alpha Vantage API key.")
        return
    
    # Simulated download folder and required files
    ml_data_path = "./data/ml_data.csv"
    sector_map_path = "./data/sector_map.csv"
    tickers_path = "./data/tickers.csv"
    macro_vars_path = "./data/macro_vars.txt"
    
    required_files = [ml_data_path, sector_map_path, tickers_path, macro_vars_path]
    
    # Simulated download button
    if st.button("Download Data (Simulated)"):
        with st.spinner("Simulating data download..."):
            # Check for required files
            for file_path in required_files:
                if not os.path.exists(file_path):
                    st.error(f"Missing required file: {file_path}")
                    return
            
            # Load data from pre-saved files
            ml_data = pd.read_csv(ml_data_path)
            sector_map = pd.read_csv(sector_map_path).set_index("Ticker").to_dict()["Sector_Returns"]
            tickers = pd.read_csv(tickers_path)["Tickers"].tolist()
            
            with open(macro_vars_path, "r") as f:
                macro_vars = f.read().splitlines()
            
            # Display the simulated data
            st.success("Data successfully loaded from pre-saved files.")
            st.write("Date Range Selected:")
            st.write(f"Start Date: {start_date}, End Date: {end_date}")
            st.write("Loaded ML Data (Sample):")
            st.write(ml_data.head())
            st.write("Available Tickers:")
            st.write(tickers)
            st.write("Sector Map:")
            st.write(sector_map)
            st.write("Macroeconomic Variables:")
            st.write(macro_vars)
            
            # Store data in session state for downstream tasks
            st.session_state['ml_data'] = ml_data
            st.session_state['tickers'] = tickers
            st.session_state['sector_map'] = sector_map
            st.session_state['macro_vars'] = macro_vars
    # Add explanation dropdown
# Streamlit dropdown explanation
    with st.expander("Explanation of Variables and Features"):
        st.markdown("""
        ### Stock-Specific Features
        - **Adjusted Close Prices**: Adjusted for corporate actions, used for accurate price analysis.
        - **Stock Returns**: Daily percentage price changes, essential for return calculations.
        - **Momentum**: Tracks long-term trends over 12 months.
        - **Moving Averages (50, 200 Days)**: Smoothens price data to highlight trends.
        - **Relative Strength Index (RSI)**: Identifies overbought/oversold conditions.

        ### Financial Ratios
        - **Debt-to-Equity Ratio**: Measures leverage. High values indicate higher debt.
        - **Return on Equity (ROE)**: Indicates profitability relative to equity.
        - **Return on Assets (ROA)**: Measures asset efficiency in generating profit.

        ### Macroeconomic Indicators
        - **GDP Growth**: Measures economic expansion or contraction.
        - **Unemployment Rate**: Reflects labor market conditions.
        - **Consumer Confidence Index (CCI)**: Indicates consumer sentiment.
        - **House Price Index (HPI)**: Tracks real estate trends.
        - **Industrial Production Index**: Represents manufacturing output.
        - **Consumer Price Index (CPI)**: Measures inflation trends.
        - **Interest Rates**: Indicates borrowing costs in the economy.
        - **VIX Index**: Reflects market volatility expectations.
        - **Economic Policy Uncertainty Index**: Captures policy-driven uncertainty.

        ### Sector-Level Features
        - **Sector Returns**: Aggregate performance for each sector.

        ### Calendar Effects
        - **Day of Week**: Captures daily return differences.
        - **Month of Year**: Captures seasonal return trends.
        """)



