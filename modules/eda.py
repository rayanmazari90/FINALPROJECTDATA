# modules/eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_feature_selection():
    st.header("Data Visualiation - Exploratory Data Analysis (EDA)")

    if 'ml_data' not in st.session_state:
        st.warning("Please run the Data Acquisition step first.")
        return

    ml_data = st.session_state['ml_data']
    tickers = st.session_state['tickers']
    sector_map = st.session_state['sector_map']
    macro_vars = st.session_state['macro_vars']

    # Provide options for the user to select variables to visualize
    st.subheader("Select Variables for Visualization")

    # Create a list of available variables
    stock_specific_vars = []
    for ticker in tickers:
        stock_specific_vars.extend([
            f'{ticker}_Returns',
            f'{ticker}_Debt_Equity',
            f'{ticker}_ROE',
            f'{ticker}_ROA',
            f'{ticker}_12M_Momentum',
            f'{ticker}_MA50',
            f'{ticker}_MA200',
            f'{ticker}_RSI'
        ])

    available_vars = macro_vars + stock_specific_vars

    selected_vars = st.multiselect("Select variables to visualize", available_vars)

    if selected_vars:
        # Show distribution plots
        st.subheader("Distribution Plots")
        for var in selected_vars:
            if var in ml_data.columns:
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(ml_data[var].dropna(), kde=True, ax=ax[0])
                ax[0].set_title(f'Histogram of {var}')
                sns.boxplot(x=ml_data[var].dropna(), ax=ax[1])
                ax[1].set_title(f'Boxplot of {var}')
                st.pyplot(fig)
            else:
                st.write(f"Variable {var} not found in the data.")

        # Show time series plots
        st.subheader("Time Series Plots")
        for var in selected_vars:
            if var in ml_data.columns and 'Date' in ml_data.columns:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(pd.to_datetime(ml_data['Date']), ml_data[var], label=var)
                ax.set_title(f'Time Series of {var}')
                ax.set_xlabel('Date')
                ax.set_ylabel(var)
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
                st.pyplot(fig)

        # Show correlation matrix
        st.subheader("Correlation Matrix")
        corr_vars = [var for var in selected_vars if var in ml_data.columns]
        corr_data = ml_data[corr_vars].dropna()
        if not corr_data.empty:
            corr_matrix = corr_data.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough data to compute correlation matrix.")

        # Provide insights
        

    else:
        st.write("Please select at least one variable to visualize.")
    # Explanation Dropdown
    with st.expander("Explanation of Graphs and Multicollinearity"):
        st.subheader("Insights")
        st.write("Based on the visualizations, you can observe the following:")
        st.write("- **Distributions:** Skewness or kurtosis in the distributions may indicate outliers or non-normality.")
        st.write("- **Time Series Trends:** Upward or downward trends over time may suggest the presence of trends or seasonality.")
        st.write("- **Correlations:** High correlation between variables may indicate multicollinearity, which can affect regression models.")
        st.markdown("""
        ### Types of Graphs
        - **Histogram**: Displays the distribution of values for a variable, helping to identify skewness or outliers.
        - **Boxplot**: Highlights the spread and potential outliers for a variable.
        - **Time Series Plot**: Shows how a variable changes over time, identifying trends or seasonality.
        - **Correlation Matrix (Heatmap)**: Displays pairwise correlations between variables, with high correlations (> 0.8) potentially indicating multicollinearity.

        ### Multicollinearity and Its Impact
        - **Definition**: Multicollinearity occurs when two or more independent variables are highly correlated.
        - **Problem**: It can distort the coefficients in regression models, making them unreliable.
        - **Solution**: Considering removing one of the correlated variables or using techniques like stepwise regression for valid models.
        """)
