import pandas as pd
import joblib
from data_preprocessing_flight import preprocess_data

model = joblib.load("../model/flight_price_model.pkl")

sample_input = {
    'date': ['18-06-2025'],
    'airline': ['Air India'],
    'ch_code': ['AI-202'],
    'num_code': [202],
    'dep_time': ['10:30'],
    'from': ['Delhi'],
    'time_taken': ['2h 30m'],
    'stop': ['non-stop'],
    'arr_time': ['13:00'],
    'to': ['Mumbai'],
    'Class': ['Economy']
}

input_df = pd.DataFrame(sample_input)

processed_input = preprocess_data(input_df)

prediction = model.predict(processed_input)

print(f"\n Predicted Flight Price: ₹{int(prediction[0])}")