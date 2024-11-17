import streamlit as st
from modules.data_acquisition import data_acquisition
from modules.modeling import linear_regression_models
from modules.eda import eda_feature_selection
from modules.stepwise_regression import stepwise_regression
from modules.optimization import investment_optimization
from PIL import Image
from pathlib import Path

def display_images_in_rows():
    st.header("Meet the Team")
    # Define the folder path using pathlib
    image_folder = Path("images")

    if not image_folder.exists():
        st.error("Image folder not found.")
        return

    # Collect image paths
    image_files = [img for img in image_folder.glob("*.*") if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif"]]

    if not image_files:
        st.warning("No images found in the folder.")
        return

    # Set a consistent size for all images
    image_size = (150, 150)  # Width, Height in pixels

    # Display images in rows of 5 with spacing
    cols_per_row = 3
    for i in range(0, len(image_files), cols_per_row):
        # Create a row with up to `cols_per_row` columns
        cols = st.columns(min(cols_per_row, len(image_files) - i))
        for col, img_path in zip(cols, image_files[i:i + cols_per_row]):
            with col:
                img = Image.open(img_path)
                img_resized = img.resize(image_size)  # Resize image
                st.image(img_resized, caption=img_path.stem, use_column_width=False)
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)  # Add vertical spacing

        # Add spacing between rows
        st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

def main():
    st.sidebar.title("Navigation")
    options = [
        "Home",
        "Data Acquisition",
        "EDA",
        "Linear Regression Models",
        "Stepwise Regression",
        "Portfolio Optimization",
        "Future Improvements and Conclusion"
    ]
    choice = st.sidebar.radio("Go to", options)

    if choice == "Home":
        st.title("Group E - 5")
        st.title("Stock Return Prediction and Portfolio Optimization")
        st.markdown("""
        Welcome to the Stock Return Prediction and Portfolio Optimization app. This application allows you to:

        - Acquire and preprocess stock and macroeconomic data.
        - Perform Exploratory Data Analysis (EDA) for feature selection.
        - Build linear regression models for stock returns.
        - Apply stepwise regression for feature selection.
        - Optimize your investment portfolio using linear programming.
        """)
        display_images_in_rows()

    elif choice == "Data Acquisition":
        data_acquisition()

    elif choice == "EDA":
        eda_feature_selection()

    elif choice == "Linear Regression Models":
        linear_regression_models()

    elif choice == "Stepwise Regression":
        stepwise_regression()

    elif choice == "Portfolio Optimization":
        investment_optimization()

    elif choice == "Future Improvements and Conclusion":
        st.title("Future Improvements and Conclusion")
        st.markdown("""
        ## Future Improvements
        Here are the potential areas for improving the application:
        
        1. **Expand Dataset**:
           - Add new stocks and macroeconomic variables for prediction.
           - Include more time-series variables to capture market dynamics.

        2. **Enhance Modeling Techniques**:
           - Introduce advanced machine learning models such as:
             - **XGBoost** for gradient boosting.
             - **Neural Networks (Deep Learning)** for nonlinear patterns.
           - Shift from simple linear regression to **quarterly predictions** using **time-series models** like ARIMA or LSTM.

        3. **Feature Engineering**:
           - Incorporate more features, such as sentiment analysis or alternative data sources.
           - Transform variables (e.g., **log**, **sqrt**, or **inverse**) to improve normality and predictive power.
           - Scale input variables for more robust model performance.

        4. **Optimization Enhancements**:
           - Add more realistic constraints in the optimization process, such as sector allocation limits or minimum stock holdings.

        5. **User Experience Improvements**:
           - Add interactive data uploads for personalized analysis.
           - Provide a dashboard for visualizing portfolio performance over time.

        ## Conclusion
        This application demonstrates the potential to combine financial data analysis with advanced machine learning techniques for portfolio optimization. By continuously enhancing the dataset, models, and optimization strategies, the app can become a powerful tool for investors looking to make data-driven decisions.

        Thank you for using the app. Stay tuned for future updates!
        """)


if __name__ == "__main__":
    main()