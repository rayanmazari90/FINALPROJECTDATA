# modules/data_acquisition.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from pandas_datareader import data as web
import warnings
import requests
import time
from functools import lru_cache

# Suppress warnings
warnings.filterwarnings('ignore')

def data_acquisition():
    st.header("Data Acquisition and Preprocessing")
    
    # User inputs for date range
    start_date = st.date_input("Start Date", datetime.date(2013, 1, 1))
    end_date = st.date_input("End Date", datetime.date(2023, 10, 31))
    
    # Financial Modeling Prep API key input (secure input)
    # It's recommended to use Streamlit's secrets management for better security
    # Uncomment the next line if using Streamlit secrets
    # api_key = st.secrets["FMP_API_KEY"]
    
    # For demonstration, using text input (ensure to replace with secrets in production)
    api_key = st.text_input("Enter your Financial Modeling Prep API key", type="password")
    if not api_key:
        st.warning("Please enter your Financial Modeling Prep API key.")
        return
    
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
            # Download stock data for all tickers
            data = yf.download(tickers, start=start_date, end=end_date)
            
            # Flatten MultiIndex Columns if necessary
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
            else:
                data.columns = [col.strip() for col in data.columns]
            
            # Calculate Returns for Each Stock
            for ticker in tickers:
                adj_close_col = f'Adj Close_{ticker}'
                if adj_close_col in data.columns:
                    data[f'{ticker}_Returns'] = data[adj_close_col].pct_change()
                else:
                    st.warning(f"Adjusted Close data for {ticker} not found.")
            
            # Macroeconomic and Common Indicators
            try:
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
            except Exception as macro_err:
                st.error(f"Error fetching macroeconomic indicators: {macro_err}")
            
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
                    try:
                        sector_data = yf.download(etf, start=start_date, end=end_date)['Adj Close']
                        if sector_data.empty:
                            st.warning(f"Adjusted Close data for {etf} not found.")
                        else:
                            data[f'{sector}_Returns'] = sector_data.pct_change().reindex(data.index, method='ffill').fillna(method='bfill')
                    except Exception as etf_err:
                        st.error(f"Error fetching data for {etf}: {etf_err}")
            
            # Initialize a separate DataFrame for aggregated financial ratios
            aggregated_ratios = pd.DataFrame()
            
            # Define the expected ratio fields and their corresponding new column names
            ratio_fields = {
                'debtEquity': 'Debt_Equity',
                'roe': 'ROE',
                'roa': 'ROA'
            }
            
            # Caching the API calls to optimize performance
            @st.cache_data(ttl=3600)
            def fetch_financial_ratios(ticker, api_key):
                url = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?limit=10&apikey={api_key}'
                response = requests.get(url)
                response.raise_for_status()
                return response.json()
            
            for ticker in tickers:
                try:
                    # Fetch ratios using the cached function
                    ratios = fetch_financial_ratios(ticker, api_key)
                    
                    if ratios:
                        # Convert the list of ratios to a DataFrame
                        df = pd.DataFrame(ratios)
                        
                        # Debug: Display available columns for the current ticker
                        st.write(f"### Available Columns for {ticker}: {df.columns.tolist()}")
                        
                        # Ensure the 'date' column is in datetime format
                        if 'date' not in df.columns:
                            st.warning(f"'date' column missing for {ticker}. Skipping.")
                            continue
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Check for the presence of expected ratio fields
                        available_fields = [field for field in ratio_fields.keys() if field in df.columns]
                        missing_fields = [field for field in ratio_fields.keys() if field not in df.columns]
                        
                        if missing_fields:
                            st.warning(f"**Warning:** Missing columns for {ticker}: {missing_fields}")
                            # Add missing fields with NaN values
                            for field in missing_fields:
                                df[field] = np.nan
                        
                        # Select relevant columns
                        selected_columns = ['date'] + list(ratio_fields.keys())
                        df = df[selected_columns]
                        
                        # Rename columns to include the ticker symbol for clarity
                        rename_dict = {'date': 'date'}
                        for field in ratio_fields.keys():
                            rename_dict[field] = f'{ticker}_{ratio_fields[field]}'
                        df = df.rename(columns=rename_dict)
                        
                        # Merge with the aggregated_ratios DataFrame
                        if aggregated_ratios.empty:
                            aggregated_ratios = df
                        else:
                            aggregated_ratios = pd.merge(aggregated_ratios, df, on='date', how='outer')
                    else:
                        st.write(f"No ratio data available for {ticker}")
                    
                    # To respect API rate limits, introduce a short delay
                    time.sleep(1)  # Sleep for 1 second between API calls
                    
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"HTTP error occurred for {ticker}: {http_err}")
                except Exception as err:
                    st.error(f"An error occurred for {ticker}: {err}")
            
            # Aligning Yearly Ratios with Daily Data
            if not aggregated_ratios.empty:
                # Sort the ratios by date
                aggregated_ratios = aggregated_ratios.sort_values('date')
                
                # Set 'date' as the DataFrame index
                aggregated_ratios.set_index('date', inplace=True)
                
                # Resample to daily frequency using forward-fill to propagate the latest ratio
                aggregated_ratios_daily = aggregated_ratios.resample('D').ffill()
                
                # Merge the ratios with the main data
                data = data.merge(aggregated_ratios_daily, left_index=True, right_index=True, how='left')
                
                # Handle any remaining missing ratio data by forward-filling
                data.fillna(method='ffill', inplace=True)
                
                # Optionally, you can drop rows where ratio data is still missing
                # data.dropna(subset=[f'{ticker}_Debt_Equity' for ticker in tickers], inplace=True)
            else:
                st.warning("No financial ratios were fetched. Proceeding without ratio data.")
            
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
                adj_close_col = f'Adj Close_{ticker}'
                if adj_close_col in data.columns:
                    data[f'{ticker}_12M_Momentum'] = data[adj_close_col].pct_change(252).shift(1)
                    data[f'{ticker}_MA50'] = data[adj_close_col].rolling(window=50).mean().shift(1)
                    data[f'{ticker}_MA200'] = data[adj_close_col].rolling(window=200).mean().shift(1)
                    data[f'{ticker}_RSI'] = rsi(data[adj_close_col]).shift(1)
                else:
                    st.warning(f"Adjusted Close data for {ticker} not found. Skipping technical indicators.")
            
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
            st.session_state['macro_vars'] = [
                'GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence', 'House_Price_Index',
                'Industrial_Production', 'CPI', 'Interest_Rate', 'VIX', 'Economic_Policy_Uncertainty'
            ]

