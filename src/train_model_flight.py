import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from data_loader import load_and_merge_data
from data_preprocessing_flight import preprocess_data

df = load_and_merge_data()


df['price'] = df['price'].replace(',', '', regex=True).astype(float)

X_raw = df.drop('price', axis=1)
y_raw = df['price']

X = preprocess_data(X_raw)
y = y_raw[X.index]  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(" Model trained successfully!")
print(" Mean Absolute Error:", round(mae, 2))
print(" R² Score:", round(r2, 2))

joblib.dump(model, "../model/flight_price_model.pkl")
