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
            # Download daily data for all tickers
            data_daily = yf.download(tickers, start=start_date, end=end_date)

            # Flatten MultiIndex Columns
            data_daily.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data_daily.columns.values]

        # Display the column names for verification

        # Extract 'Adj Close' columns
        adj_close_cols = [col for col in data_daily.columns if 'Adj Close' in col]

        adj_close = data_daily[adj_close_cols]


        # Standardize column names to '{ticker}_Adj Close'
        new_col_names = []
        for col in adj_close.columns:
            # Assuming the format is 'Adj Close_{ticker}'
            if 'Adj Close_' in col:
                ticker = col.split('_')[1]
            elif '_' in col:
                # Handle cases like '{ticker}_Adj Close'
                ticker = col.split('_')[0]
            else:
                st.write(f"Unexpected column format: {col}")
                continue
            new_col_names.append(f'{ticker}_Adj_Close')

        # Rename columns
        adj_close.columns = new_col_names

        # Display the renamed columns

        # Resample to quarterly frequency
        adj_close_qtr = adj_close.resample('Q').last()
        # Initialize data_qtr DataFrame
        data_qtr = pd.DataFrame(index=adj_close_qtr.index)
        st.write("pleaseeeee",adj_close_qtr.columns)

        # Calculate Quarterly Returns for Each Stock
        for ticker in tickers:
            adj_col = f'{ticker}_Adj_Close'
            if adj_col not in adj_close_qtr.columns:
                st.write(f"Adjusted close price for {ticker} not found.")
                continue  # Skip this ticker if data is missing
            data_qtr[f'{ticker}_Adj_Close'] = adj_close_qtr[adj_col]
            data_qtr[f'{ticker}_Returns'] = adj_close_qtr[adj_col].pct_change()

        # Macroeconomic and Common Indicators
        # GDP (already quarterly)
        gdp = web.DataReader('GDP', 'fred', start_date, end_date)
        gdp = gdp.resample('Q').last()
        # Shift GDP data by one quarter to account for data release lag
        gdp_shifted = gdp.shift(1)
        data_qtr['GDP'] = gdp_shifted.reindex(data_qtr.index).ffill()

        # GDP Growth Rate
        data_qtr['GDP_Growth'] = data_qtr['GDP'].pct_change()

        # Unemployment Rate (monthly data)
        unemployment_rate = web.DataReader('UNRATE', 'fred', start_date, end_date)
        unemployment_rate_qtr = unemployment_rate.resample('Q').mean()
        data_qtr['Unemployment_Rate'] = unemployment_rate_qtr.reindex(data_qtr.index).ffill()

        # Consumer Confidence Index (CCI) (monthly data)
        consumer_confidence = web.DataReader('UMCSENT', 'fred', start_date, end_date)
        consumer_confidence_qtr = consumer_confidence.resample('Q').mean()
        data_qtr['Consumer_Confidence'] = consumer_confidence_qtr.reindex(data_qtr.index).ffill()

        # House Price Index (HPI) (monthly data)
        hpi = web.DataReader('CSUSHPINSA', 'fred', start_date, end_date)
        hpi_qtr = hpi.resample('Q').last()
        data_qtr['House_Price_Index'] = hpi_qtr.reindex(data_qtr.index).ffill()

        # Industrial Production Index (monthly data)
        industrial_production = web.DataReader('INDPRO', 'fred', start_date, end_date)
        industrial_production_qtr = industrial_production.resample('Q').mean()
        data_qtr['Industrial_Production'] = industrial_production_qtr.reindex(data_qtr.index).ffill()

        # CPI (Consumer Price Index) (monthly data)
        cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
        cpi_qtr = cpi.resample('Q').mean()
        data_qtr['CPI'] = cpi_qtr.reindex(data_qtr.index).ffill()

        # Interest Rates (Daily Federal Funds Rate)
        interest_rate = web.DataReader('DFF', 'fred', start_date, end_date)
        interest_rate_qtr = interest_rate.resample('Q').mean()
        data_qtr['Interest_Rate'] = interest_rate_qtr.reindex(data_qtr.index).ffill() / 100

        # VIX Index (Volatility Index) (daily data)
        vix_data = yf.download('^VIX', start=start_date, end=end_date)['Close']
        vix_qtr = vix_data.resample('Q').mean()
        data_qtr['VIX'] = vix_qtr.reindex(data_qtr.index).ffill()

        # Economic Policy Uncertainty Index (monthly data)
        epu = web.DataReader('USEPUINDXD', 'fred', start_date, end_date)
        epu_qtr = epu.resample('Q').mean()
        data_qtr['Economic_Policy_Uncertainty'] = epu_qtr.reindex(data_qtr.index).ffill()

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
                sector_data_qtr = sector_data.resample('Q').last()
                data_qtr[f'{sector}_Returns'] = sector_data_qtr.pct_change().reindex(data_qtr.index).ffill()

        # Financial Ratios for Each Stock
        # Use quarterly financial statements
        """
        for ticker in tickers:
            try:
                income_stmt, _ = fd.get_income_statement_quarterly(ticker)
                balance_sheet, _ = fd.get_balance_sheet_quarterly(ticker)
                # Parse dates and set as index
                income_stmt['fiscalDateEnding'] = pd.to_datetime(income_stmt['fiscalDateEnding'])
                income_stmt.set_index('fiscalDateEnding', inplace=True)
                balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
                balance_sheet.set_index('fiscalDateEnding', inplace=True)
                # Debt to Equity Ratio
                total_liabilities = pd.to_numeric(balance_sheet['totalLiabilities'], errors='coerce')
                shareholders_equity = pd.to_numeric(balance_sheet['totalShareholderEquity'], errors='coerce')
                debt_equity = (total_liabilities / shareholders_equity).resample('Q').last()
                data_qtr[f'{ticker}_Debt_Equity'] = debt_equity.reindex(data_qtr.index).ffill()
                # ROE and ROA
                net_income = pd.to_numeric(income_stmt['netIncome'], errors='coerce')
                total_assets = pd.to_numeric(balance_sheet['totalAssets'], errors='coerce')
                roe = (net_income / shareholders_equity).resample('Q').last()
                roa = (net_income / total_assets).resample('Q').last()
                data_qtr[f'{ticker}_ROE'] = roe.reindex(data_qtr.index).ffill()
                data_qtr[f'{ticker}_ROA'] = roa.reindex(data_qtr.index).ffill()
            except Exception as e:
                st.write(f"Error with {ticker}: {e}")
        """ 
        
        # Technical Indicators
        for ticker in tickers:
            st.write("---------------")
            st.write(data_qtr.columns)
            adj_col = f'{ticker}_Adj_Close'
            if adj_col not in data_qtr.columns:
                st.write(f"Adjusted close column for {ticker} not found in data_qtr.")
                continue
            data_qtr[f'{ticker}_12Q_Momentum'] = data_qtr[adj_col].pct_change(12).shift(1)
            data_qtr[f'{ticker}_MA4'] = data_qtr[adj_col].rolling(window=4).mean().shift(1)
            data_qtr[f'{ticker}_MA8'] = data_qtr[adj_col].rolling(window=8).mean().shift(1)

        # Calendar Effects
        data_qtr['Quarter'] = data_qtr.index.quarter
        data_qtr['Year'] = data_qtr.index.year
        # One-hot encoding for Quarter
        data_qtr = pd.get_dummies(data_qtr, columns=['Quarter'], prefix='Q', drop_first=True)

        # Drop NaNs to finalize data for modeling
        st.write(data_qtr.head(100))
        ml_data = data_qtr.dropna().copy()
        st.success("Data acquisition and preprocessing completed.")
        st.write(ml_data.head())

        # Store data in session state
        st.session_state['ml_data'] = ml_data
        st.session_state['tickers'] = tickers
        st.session_state['sector_map'] = sector_map
        st.session_state['macro_vars'] = ['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence', 'House_Price_Index',
                                            'Industrial_Production', 'CPI', 'Interest_Rate', 'VIX', 'Economic_Policy_Uncertainty']
