# app.py

import streamlit as st
from modules.data_acquisition import data_acquisition
from modules.modeling import linear_regression_models
from modules.eda import eda_feature_selection
from modules.stepwise_regression import stepwise_regression
from modules.decision_theory import decision_theory
from modules.optimization import investment_optimization

def main():
    st.sidebar.title("Navigation")
    options = [
        "Home",
        "Data Acquisition",
        "EDA",
        "Linear Regression Models",
        "Stepwise Regression",
        "Decision Theory",
        "Investment Optimization"
    ]
    choice = st.sidebar.radio("Go to", options)

    if choice == "Home":
        st.title("Stock Return Prediction Model")
        st.markdown("""
        Welcome to the Stock Return Prediction Model app. This application allows you to:

        - Acquire and preprocess stock and macroeconomic data.
        - Perform Exploratory Data Analysis (EDA) for feature selection.
        - Build linear regression models for stock returns.
        - Filter models based on performance metrics.
        - Visualize model predictions.

        More features like stepwise regression and investment optimization will be added soon.
        """)
    elif choice == "Data Acquisition":
        data_acquisition()
    elif choice == "EDA":
        eda_feature_selection()
    elif choice == "Linear Regression Models":
        linear_regression_models()
    elif choice == "Stepwise Regression":
        stepwise_regression()
    elif choice == "Decision Theory":
        decision_theory()
    elif choice == "Investment Optimization":
        investment_optimization()

if __name__ == "__main__":
    main()
