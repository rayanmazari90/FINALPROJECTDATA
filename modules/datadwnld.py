# data_download.ipynb

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from pandas_datareader import data as web
import requests
import warnings
from alpha_vantage.fundamentaldata import FundamentalData
# Suppress warnings
warnings.filterwarnings('ignore')

def data_acquisition_and_save():
    # Define date range
    
    fd = FundamentalData(key="9MDBAA53ZS2CAXE2", output_format='pandas')
    start_date = datetime.date(2013, 1, 1)
    end_date = datetime.date(2023, 10, 31)

    # FMP API key
    

    # Define ticker symbols for each sector with additional stocks
    tech_stocks = ['AAPL', 'MSFT', 'NVDA']
    financial_stocks = ['JPM', 'GS', 'BAC']
    healthcare_stocks = ['JNJ', 'UNH', 'LLY']
    energy_stocks = ['XOM', 'CVX', 'EOG']
    consumer_discretionary_stocks = ['MCD', 'NKE', 'HD']
    industrial_stocks = ['CAT', 'LMT', 'UPS']
    utilities_stocks = ['NEE', 'DUK', 'SO']
    #tech_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'INTC', 'ADBE', 'CSCO', 'ORCL']
    #financial_stocks = ['JPM', 'GS', 'BAC', 'WFC', 'C', 'MS', 'BLK', 'AXP', 'SPGI', 'BK']
    #healthcare_stocks = ['JNJ', 'PFE', 'MRK', 'ABT', 'TMO', 'UNH', 'MDT', 'CVS', 'LLY', 'DHR']
    #energy_stocks = ['XOM', 'CVX', 'SLB', 'COP', 'EOG', 'PSX', 'MPC', 'KMI', 'VLO', 'HAL']
    #consumer_discretionary_stocks = ['MCD', 'NKE', 'SBUX', 'HD', 'LOW', 'BKNG', 'TJX', 'DG', 'ROST', 'EBAY']
    #industrial_stocks = ['CAT', 'BA', 'HON', 'LMT', 'GE', 'MMM', 'RTX', 'UPS', 'DE', 'FDX']
    #utilities_stocks = ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED']

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

    selected_sectors = list(sectors.keys())  # Select all sectors

    # Update tickers based on selected sectors
    tickers = []
    for sector in selected_sectors:
        tickers.extend(sectors[sector])

    # Map each ticker to its sector
    sector_map = {}
    for sector in selected_sectors:
        for ticker in sectors[sector]:
            sector_map[ticker] = f'{sector}_Returns'

    print("Downloading data...")

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
    gdp_growth = gdp.pct_change()
    data['GDP_Growth'] = gdp_growth.reindex(data.index, method='ffill').fillna(method='bfill')['GDP']

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

    # Sector ETF Performance
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

    # Financial Ratios for Each Stock using FMP API
    # Financial Ratios for Each Stock (Quarterly Adjusted to Monthly)
    """ 
    for ticker in tickers:
        try:
            # Fetch quarterly income statement and balance sheet using fd
            income_stmt, _ = fd.get_income_statement_quarterly(ticker)
            balance_sheet, _ = fd.get_balance_sheet_quarterly(ticker)

            # Ensure we have data
            if not income_stmt.empty and not balance_sheet.empty:
                # Convert 'fiscalDateEnding' to datetime and set as index
                income_stmt['fiscalDateEnding'] = pd.to_datetime(income_stmt['fiscalDateEnding'])
                balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])

                # Align data to the dataset's monthly frequency
                all_months = pd.date_range(start=start_date, end=end_date, freq='M')
                income_stmt = income_stmt.set_index('fiscalDateEnding').reindex(all_months, method='ffill')
                balance_sheet = balance_sheet.set_index('fiscalDateEnding').reindex(all_months, method='ffill')

                # Use the most recent monthly data
                latest_income = income_stmt.iloc[-1]
                latest_balance = balance_sheet.iloc[-1]

                # Debt to Equity Ratio
                total_liabilities = pd.to_numeric(latest_balance.get('totalLiabilities', np.nan), errors='coerce')
                shareholders_equity = pd.to_numeric(latest_balance.get('totalShareholderEquity', np.nan), errors='coerce')

                if pd.notna(total_liabilities) and pd.notna(shareholders_equity):
                    data[f'{ticker}_Debt_Equity'] = total_liabilities / shareholders_equity
                else:
                    data[f'{ticker}_Debt_Equity'] = np.nan  # Fill missing values with NaN

                # ROE and ROA
                net_income = pd.to_numeric(latest_income.get('netIncome', np.nan), errors='coerce')
                total_assets = pd.to_numeric(latest_balance.get('totalAssets', np.nan), errors='coerce')

                if pd.notna(net_income) and pd.notna(shareholders_equity):
                    data[f'{ticker}_ROE'] = net_income / shareholders_equity
                else:
                    data[f'{ticker}_ROE'] = np.nan

                if pd.notna(net_income) and pd.notna(total_assets):
                    data[f'{ticker}_ROA'] = net_income / total_assets
                else:
                    data[f'{ticker}_ROA'] = np.nan

            else:
                print(f"No data available for {ticker}")
                data[f'{ticker}_Debt_Equity'] = np.nan
                data[f'{ticker}_ROE'] = np.nan
                data[f'{ticker}_ROA'] = np.nan

        except Exception as e:
            print(f"Error processing data for {ticker}: {e}")
            data[f'{ticker}_Debt_Equity'] = np.nan
            data[f'{ticker}_ROE'] = np.nan
            data[f'{ticker}_ROA'] = np.nan
    """
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

    print("Data acquisition and preprocessing completed.")
    print(ml_data.head())

    # Save the data locally
    ml_data.to_csv('../data/ml_data.csv')
    print("Data saved to 'ml_data.csv'")

    # Save additional necessary data
    sector_map_df = pd.DataFrame(list(sector_map.items()), columns=['Ticker', 'Sector_Returns'])
    sector_map_df.to_csv('../data/sector_map.csv', index=False)

    # Save tickers list
    tickers_df = pd.DataFrame(tickers, columns=['Tickers'])
    tickers_df.to_csv('../data/tickers.csv', index=False)

    # Save macro variables
    macro_vars = ['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence', 'House_Price_Index',
                  'Industrial_Production', 'CPI', 'Interest_Rate', 'VIX', 'Economic_Policy_Uncertainty']
    with open('../data/macro_vars.txt', 'w') as f:
        for var in macro_vars:
            f.write(f"{var}\n")

if __name__ == "__main__":
    data_acquisition_and_save()
