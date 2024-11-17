# modules/data_acquisition.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from pandas_datareader import data as web
from alpha_vantage.fundamentaldata import FundamentalData
import warnings
# Suppress warnings
warnings.filterwarnings('ignore')
def data_acquisition():
    st.header("Data Acquisition and Preprocessing")
    # User inputs for date range
    start_date = st.date_input("Start Date", datetime.date(2013, 1, 1))
    end_date = st.date_input("End Date", datetime.date(2023, 10, 31))
    # Alpha Vantage API key input (secure input)
    api_key = st.text_input("Enter your Alpha Vantage API key", type="password")
    if not api_key:
        st.warning("Please enter your Alpha Vantage API key.")
        return
        
    fd = FundamentalData(key=api_key, output_format='pandas')
    # Define ticker symbols for each sector with additional stocks
    tech_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'INTC', 'ADBE', 'CSCO', 'ORCL']
    financial_stocks = ['JPM', 'GS', 'BAC', 'WFC', 'C', 'MS', 'BLK', 'AXP', 'SPGI', 'BK']
    healthcare_stocks = ['JNJ', 'PFE', 'MRK', 'ABT', 'TMO', 'UNH', 'MDT', 'CVS', 'LLY', 'DHR']
    energy_stocks = ['XOM', 'CVX', 'SLB', 'COP', 'EOG', 'PSX', 'MPC', 'KMI', 'VLO', 'HAL']
    consumer_discretionary_stocks = ['MCD', 'NKE', 'SBUX', 'HD', 'LOW', 'BKNG', 'TJX', 'DG', 'ROST', 'EBAY']
    industrial_stocks = ['CAT', 'BA', 'HON', 'LMT', 'GE', 'MMM', 'RTX', 'UPS', 'DE', 'FDX']
    utilities_stocks = ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED']
    # Combine all tickers into a single list
    sectors = {
        'Technology': tech_stocks,
        'Financials': financial_stocks,
        'Healthcare': healthcare_stocks,
        'Energy': energy_stocks,
        'Consumer Discretionary': consumer_discretionary_stocks,
        'Industrials': industrial_stocks,
        'Utilities': utilities_stocks
    }
    selected_sectors = st.multiselect("Select Sectors", list(sectors.keys()), default=list(sectors.keys()))
    # Update tickers based on selected sectors
    tickers = []
    for sector in selected_sectors:
        tickers.extend(sectors[sector])
    # Map each ticker to its sector
    sector_map = {}
    for sector in selected_sectors:
        for ticker in sectors[sector]:
            sector_map[ticker] = f'{sector}_Returns'
    if st.button("Download Data"):
        with st.spinner("Downloading data..."):
            # Download data for all tickers
            data = yf.download(tickers, start=start_date, end=end_date)
            # Flatten MultiIndex Columns
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
            # Calculate Returns for Each Stock
            for ticker in tickers:
                data[f'{ticker}_Returns'] = data[f'Adj Close_{ticker}'].pct_change()
            # Macroeconomic and Common Indicators
            # GDP Growth
            gdp = web.DataReader('GDP', 'fred', start_date, end_date)
            gdp_growth = gdp.pct_change().reindex(data.index, method='ffill').fillna(method='bfill')
            data['GDP_Growth'] = gdp_growth['GDP']
            # Unemployment Rate
            unemployment_rate = web.DataReader('UNRATE', 'fred', start_date, end_date)
            data['Unemployment_Rate'] = unemployment_rate.reindex(data.index, method='ffill').fillna(method='bfill')['UNRATE']
            # Consumer Confidence Index (CCI)
            consumer_confidence = web.DataReader('UMCSENT', 'fred', start_date, end_date)
            data['Consumer_Confidence'] = consumer_confidence.reindex(data.index, method='ffill').fillna(method='bfill')['UMCSENT']
            # House Price Index (HPI)
            hpi = web.DataReader('CSUSHPINSA', 'fred', start_date, end_date)
            data['House_Price_Index'] = hpi.reindex(data.index, method='ffill').fillna(method='bfill')['CSUSHPINSA']
            # Industrial Production Index
            industrial_production = web.DataReader('INDPRO', 'fred', start_date, end_date)
            data['Industrial_Production'] = industrial_production.reindex(data.index, method='ffill').fillna(method='bfill')['INDPRO']
            # CPI (Consumer Price Index)
            cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
            data['CPI'] = cpi.reindex(data.index, method='ffill').fillna(method='bfill')['CPIAUCSL']
            # Interest Rates (Daily Federal Funds Rate)
            interest_rate = web.DataReader('DFF', 'fred', start_date, end_date)
            data['Interest_Rate'] = interest_rate.reindex(data.index, method='ffill').fillna(method='bfill')['DFF'] / 100
            # VIX Index (Volatility Index)
            vix_data = yf.download('^VIX', start=start_date, end=end_date)['Close']
            data['VIX'] = vix_data.reindex(data.index, method='ffill').fillna(method='bfill')
            # Economic Policy Uncertainty Index
            epu = web.DataReader('USEPUINDXD', 'fred', start_date, end_date)
            data['Economic_Policy_Uncertainty'] = epu.reindex(data.index, method='ffill').fillna(method='bfill')['USEPUINDXD']
            # Sector ETF Performance (to capture sector-specific market trends)
            sector_etfs = {
                'Technology': 'XLK',
                'Financials': 'XLF',
                'Healthcare': 'XLV',
                'Energy': 'XLE',
                'Consumer Discretionary': 'XLY',
                'Industrials': 'XLI',
                'Utilities': 'XLU'
            }
            for sector, etf in sector_etfs.items():
                if sector in selected_sectors:
                    sector_data = yf.download(etf, start=start_date, end=end_date)['Adj Close']
                    data[f'{sector}_Returns'] = sector_data.pct_change().reindex(data.index, method='ffill').fillna(method='bfill')
            # Financial Ratios for Each Stock
            #for ticker in tickers:
            #    try:
            #        income_stmt, _ = fd.get_income_statement_annual(ticker)
            #        balance_sheet, _ = fd.get_balance_sheet_annual(ticker)
            #        # Debt to Equity Ratio
            #        total_liabilities = pd.to_numeric(balance_sheet['totalLiabilities'].iloc[0], errors='coerce')
            #        shareholders_equity = pd.to_numeric(balance_sheet['totalShareholderEquity'].iloc[0], errors='coerce')
            #        if pd.notna(total_liabilities) and pd.notna(shareholders_equity):
            #            data[f'{ticker}_Debt_Equity'] = total_liabilities / shareholders_equity
            #        # ROE and ROA
            #        net_income = pd.to_numeric(income_stmt['netIncome'].iloc[0], errors='coerce')
            #        total_assets = pd.to_numeric(balance_sheet['totalAssets'].iloc[0], errors='coerce')
            #        if pd.notna(net_income) and pd.notna(shareholders_equity):
            #            data[f'{ticker}_ROE'] = net_income / shareholders_equity
            #        if pd.notna(net_income) and pd.notna(total_assets):
            #            data[f'{ticker}_ROA'] = net_income / total_assets
            #    except Exception as e:
            #        st.write(f"Error with {ticker}: {e}")
            # Technical Indicators
            def rsi(series, period=14):
                delta = series.diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                gain = up.rolling(window=period).mean()
                loss = down.rolling(window=period).mean()
                RS = gain / loss
                return 100 - (100 / (1 + RS))
            for ticker in tickers:
                data[f'{ticker}_12M_Momentum'] = data[f'Adj Close_{ticker}'].pct_change(252).shift(1)
                data[f'{ticker}_MA50'] = data[f'Adj Close_{ticker}'].rolling(window=50).mean().shift(1)
                data[f'{ticker}_MA200'] = data[f'Adj Close_{ticker}'].rolling(window=200).mean().shift(1)
                data[f'{ticker}_RSI'] = rsi(data[f'Adj Close_{ticker}']).shift(1)
            # Calendar Effects
            data['Day_of_Week'] = data.index.dayofweek
            data['Month_of_Year'] = data.index.month
            data = pd.get_dummies(data, columns=['Day_of_Week', 'Month_of_Year'], drop_first=True)
            # Drop NaNs to finalize data for modeling
            ml_data = data.dropna().copy()
            st.success("Data acquisition and preprocessing completed.")
            st.write(ml_data.head())
            # Store data in session state
            st.session_state['ml_data'] = ml_data
            st.session_state['tickers'] = tickers
            st.session_state['sector_map'] = sector_map
            st.session_state['macro_vars'] = ['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence', 'House_Price_Index',
                                              'Industrial_Production', 'CPI', 'Interest_Rate', 'VIX', 'Economic_Policy_Uncertainty']