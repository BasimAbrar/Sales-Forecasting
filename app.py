import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


model = joblib.load("walmart_sales_model.pkl")
features = joblib.load("model_features.pkl")

st.title("Walmart Sales Forecasting Website")
st.write("Upload your dataset and get predictions using the pretrained XGBoost model.")


file = st.file_uploader("Upload your CSV file", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())


    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)


    data.fillna(0, inplace=True)


    input_data = data[features]


    predictions = model.predict(input_data)
    data['Predicted_Sales'] = predictions

    st.write("### Predictions")
    st.dataframe(data[['Store','Dept','Date','Predicted_Sales']].head(20))


    st.write("### Sales Forecast Visualization")
    fig, ax = plt.subplots(figsize=(10,5))
    if 'Date' in data.columns:
        data.groupby('Date')['Predicted_Sales'].sum().plot(ax=ax)
        ax.set_title("Predicted Sales Over Time")
        ax.set_ylabel("Sales")
        ax.set_xlabel("Date")
        st.pyplot(fig)

    st.download_button(
        label="Download Predictions as CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="walmart_predictions.csv",
        mime="text/csv"
    )
