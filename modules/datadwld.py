# data_acquisition.py

import argparse
import os
import sys
import datetime
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given pandas Series.

    Parameters:
        series (pd.Series): The price series.
        period (int): The number of periods to use for RSI calculation.

    Returns:
        pd.Series: The RSI values.
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    gain = up.rolling(window=period, min_periods=period).mean()
    loss = down.rolling(window=period, min_periods=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))


def download_stock_data(tickers, start_date, end_date):
    """
    Download historical stock data for given tickers.

    Parameters:
        tickers (list): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Combined stock data with adjusted close prices.
    """
    print("Downloading stock data...")
    data = yf.download(tickers, start=start_date, end=end_date)
    if data.empty:
        print("No stock data downloaded. Please check ticker symbols and date range.")
        sys.exit(1)
    # Flatten MultiIndex Columns
    data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
    # Calculate Returns for Each Stock
    for ticker in tickers:
        data[f'{ticker}_Returns'] = data.get(f'Adj Close_{ticker}', pd.Series()).pct_change()
    return data


def download_macro_data(start_date, end_date):
    """
    Download macroeconomic data from FRED.

    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Macro data merged into a single DataFrame.
    """
    print("Downloading macroeconomic data...")
    macro_vars = {}
    macro_var_names = {
        'GDP': 'GDP_Growth',
        'UNRATE': 'Unemployment_Rate',
        'UMCSENT': 'Consumer_Confidence',
        'CSUSHPINSA': 'House_Price_Index',
        'INDPRO': 'Industrial_Production',
        'CPIAUCSL': 'CPI',
        'DFF': 'Interest_Rate',
        'USEPUINDXD': 'Economic_Policy_Uncertainty'
    }

    for fred_code, var_name in macro_var_names.items():
        try:
            data = web.DataReader(fred_code, 'fred', start_date, end_date)
            if fred_code == 'DFF':
                # Convert Interest Rate from percentage to decimal
                data[var_name] = data[fred_code] / 100
            else:
                # Calculate percentage change for other macro variables
                data[var_name] = data[fred_code].pct_change()
            macro_vars[var_name] = data[var_name]
        except Exception as e:
            print(f"Error downloading {var_name}: {e}")

    macro_df = pd.DataFrame(macro_vars)
    return macro_df


def download_vix(start_date, end_date):
    """
    Download VIX index data.

    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.Series: VIX Close prices.
    """
    print("Downloading VIX data...")
    try:
        vix_data = yf.download('^VIX', start=start_date, end=end_date)['Close']
        return vix_data
    except Exception as e:
        print(f"Error downloading VIX data: {e}")
        return pd.Series(dtype='float64')


def download_sector_etfs(selected_sectors, start_date, end_date):
    """
    Download sector ETF performance data.

    Parameters:
        selected_sectors (list): List of selected sector names.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Sector ETF returns.
    """
    print("Downloading sector ETF data...")
    sector_etfs = {
        'Technology': 'XLK',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Energy': 'XLE',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Utilities': 'XLU'
    }
    etf_returns = {}
    for sector in selected_sectors:
        etf = sector_etfs.get(sector)
        if etf:
            try:
                etf_data = yf.download(etf, start=start_date, end=end_date)['Adj Close']
                etf_returns[f'{sector}_Returns'] = etf_data.pct_change()
            except Exception as e:
                print(f"Error downloading ETF for {sector}: {e}")
    return pd.DataFrame(etf_returns)


def calculate_technical_indicators(tickers, data):
    """
    Calculate technical indicators for each stock.

    Parameters:
        tickers (list): List of ticker symbols.
        data (pd.DataFrame): The main data DataFrame to which indicators will be added.

    Returns:
        pd.DataFrame: Updated data with technical indicators.
    """
    print("Calculating technical indicators...")
    for ticker in tickers:
        try:
            adj_close = data.get(f'Adj Close_{ticker}', pd.Series())
            if adj_close.empty:
                print(f"No adjusted close data for {ticker}. Skipping technical indicators.")
                continue
            data[f'{ticker}_12M_Momentum'] = adj_close.pct_change(252).shift(1)
            data[f'{ticker}_MA50'] = adj_close.rolling(window=50).mean().shift(1)
            data[f'{ticker}_MA200'] = adj_close.rolling(window=200).mean().shift(1)
            data[f'{ticker}_RSI'] = rsi(adj_close).shift(1)
        except Exception as e:
            print(f"Error calculating indicators for {ticker}: {e}")
    return data


def add_calendar_effects(data):
    """
    Add calendar effect features to the data.

    Parameters:
        data (pd.DataFrame): The main data DataFrame.

    Returns:
        pd.DataFrame: Updated data with calendar effects.
    """
    print("Adding calendar effects...")
    data['Day_of_Week'] = data.index.dayofweek
    data['Month_of_Year'] = data.index.month
    data = pd.get_dummies(data, columns=['Day_of_Week', 'Month_of_Year'], drop_first=True)
    return data


def main(args):
    # Define sector stock tickers
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'INTC', 'ADBE', 'CSCO', 'ORCL'],
        'Financials': ['JPM', 'GS', 'BAC', 'WFC', 'C', 'MS', 'BLK', 'AXP', 'SPGI', 'BK'],
        'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'TMO', 'UNH', 'MDT', 'CVS', 'LLY', 'DHR'],
        'Energy': ['XOM', 'CVX', 'SLB', 'COP', 'EOG', 'PSX', 'MPC', 'KMI', 'VLO', 'HAL'],
        'Consumer Discretionary': ['MCD', 'NKE', 'SBUX', 'HD', 'LOW', 'BKNG', 'TJX', 'DG', 'ROST', 'EBAY'],
        'Industrials': ['CAT', 'BA', 'HON', 'LMT', 'GE', 'MMM', 'RTX', 'UPS', 'DE', 'FDX'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'ED']
    }

    # Combine tickers based on selected sectors
    selected_sectors = args.sectors
    if not selected_sectors:
        selected_sectors = list(sectors.keys())
    tickers = []
    for sector in selected_sectors:
        tickers.extend(sectors.get(sector, []))

    # Map each ticker to its sector
    sector_map = {}
    for sector in selected_sectors:
        for ticker in sectors.get(sector, []):
            sector_map[ticker] = sector

    # Download data
    stock_data = download_stock_data(tickers, args.start_date, args.end_date)
    macro_data = download_macro_data(args.start_date, args.end_date)
    vix_data = download_vix(args.start_date, args.end_date)
    sector_etf_data = download_sector_etfs(selected_sectors, args.start_date, args.end_date)

    # Merge all data
    print("Merging all data...")
    data = stock_data.join(macro_data, how='left')
    data = data.join(vix_data.rename('VIX'), how='left')
    data = data.join(sector_etf_data, how='left')

    # Forward fill and backward fill missing data
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    # Calculate technical indicators
    data = calculate_technical_indicators(tickers, data)

    # Add calendar effects
    data = add_calendar_effects(data)

    # Drop rows with any remaining NaNs
    ml_data = data.dropna().copy()

    # Display completion message
    print("Data acquisition and preprocessing completed.")

    # Define output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save the final data
    ml_data_path = os.path.join(output_dir, 'ml_data.csv')
    ml_data.to_csv(ml_data_path)
    print(f"ML data saved to {ml_data_path}")

    # Save auxiliary information
    tickers_path = os.path.join(output_dir, 'tickers.csv')
    pd.DataFrame(tickers, columns=['Ticker']).to_csv(tickers_path, index=False)
    print(f"Tickers saved to {tickers_path}")

    sector_map_path = os.path.join(output_dir, 'sector_map.csv')
    pd.DataFrame(list(sector_map.items()), columns=['Ticker', 'Sector']).to_csv(sector_map_path, index=False)
    print(f"Sector map saved to {sector_map_path}")

    macro_vars = ['GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence', 'House_Price_Index',
                  'Industrial_Production', 'CPI', 'Interest_Rate', 'VIX', 'Economic_Policy_Uncertainty']
    macro_vars_path = os.path.join(output_dir, 'macro_vars.txt')
    with open(macro_vars_path, 'w') as f:
        for var in macro_vars:
            f.write(f"{var}\n")
    print(f"Macro variables saved to {macro_vars_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and store financial and macroeconomic data.")
    parser.add_argument('--start_date', type=str, default='2013-01-01',
                        help='Start date in YYYY-MM-DD format.')
    parser.add_argument('--end_date', type=str, default=datetime.date.today().strftime('%Y-%m-%d'),
                        help='End date in YYYY-MM-DD format.')
    parser.add_argument('--sectors', type=str, nargs='*',
                        choices=['Technology', 'Financials', 'Healthcare', 'Energy',
                                 'Consumer Discretionary', 'Industrials', 'Utilities'],
                        default=['Technology', 'Financials', 'Healthcare', 'Energy',
                                 'Consumer Discretionary', 'Industrials', 'Utilities'],
                        help='Sectors to include.')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the downloaded data.')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
