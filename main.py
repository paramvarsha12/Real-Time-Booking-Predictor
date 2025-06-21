import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

model = joblib.load("../model/flight_price_model.pkl")

st.title("Flight Price Predictor")
st.markdown("Enter flight details below to predict the price.")

airline = st.selectbox("Airline", ['Indigo', 'Air India', 'SpiceJet', 'Vistara', 'GoAir'])
ch_code = st.text_input("Airline Code (e.g., AI, SG, 6E)", "6E")
num_code = st.number_input("Flight Number", min_value=1000, max_value=9999, value=1234)
dep_time = st.time_input("Departure Time", datetime.now().time())
arr_time = st.time_input("Arrival Time", datetime.now().time())
from_city = st.selectbox("From", ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'])
to_city = st.selectbox("To", ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'])
duration = st.text_input("Duration (e.g., 2h 50m)", "2h 30m")
stops = st.selectbox("Stops", ['non-stop', '1 stop', '2 stops'])
flight_class = st.selectbox("Class", ['Economy', 'Business'])
journey_date = st.date_input("Journey Date")

if st.button("Predict Price"):
    try:
        journey_date = pd.to_datetime(journey_date)
        dep_hour = dep_time.hour
        dep_min = dep_time.minute
        arr_hour = arr_time.hour
        arr_min = arr_time.minute

        h, m = 0, 0
        if 'h' in duration:
            h = int(duration.split('h')[0].strip())
        if 'm' in duration:
            m = int(duration.split('h')[1].replace('m', '').strip()) if 'h' in duration else int(duration.replace('m', '').strip())
        total_mins = h * 60 + m

        stop_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2}
        stop_value = stop_mapping.get(stops, 0)

        class_mapping = {'Economy': 0, 'Business': 1}
        class_value = class_mapping[flight_class]

        input_df = pd.DataFrame({
            'date': [journey_date.strftime('%d-%m-%Y')],
            'airline': [airline],
            'ch_code': [ch_code],
            'num_code': [num_code],
            'dep_time': [dep_time.strftime('%H:%M')],
            'from': [from_city],
            'time_taken': [duration],
            'stop': [stops],
            'arr_time': [arr_time.strftime('%H:%M')],
            'to': [to_city],
            'Class': [flight_class],
            'price': [0]
        })

        from data_preprocessing_flight import preprocess_data
        processed_input = preprocess_data(input_df.drop('price', axis=1))

        training_data = pd.read_csv("../data/business.csv")
        reference_cols = preprocess_data(training_data.drop('price', axis=1)).columns

        for col in reference_cols:
            if col not in processed_input.columns:
                processed_input[col] = 0

        processed_input = processed_input[reference_cols]

        prediction = model.predict(processed_input)[0]
        st.success(f"Predicted Flight Price: ₹{int(prediction):,}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
