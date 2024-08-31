from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv("sales.csv")
df['date'] = pd.to_datetime(df['date'])

# Clean and preprocess the data
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
df.drop_duplicates(inplace=True)
df['price'] = df['price'].replace(0, np.nan)
df['price'].fillna(method='ffill', inplace=True)
df = df[(df['sales'] != 0) & (df['stock'] != 0)]
df.reset_index(drop=True, inplace=True)

# Train Linear Regression model
X = df[['stock', 'price']]
y = df['sales']
lr_model = LinearRegression()
lr_model.fit(X, y)

# Train ARIMA model on the 'sales' column
df.sort_values(by='date', inplace=True)
arima_model = ARIMA(df['sales'], order=(1, 1, 1))
arima_fit = arima_model.fit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Convert the input data into a DataFrame
        input_data = pd.DataFrame([data])

        # Ensure input_data has the same column names as the training data
        expected_columns = ['stock', 'price']
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)

        # Linear Regression prediction
        lr_prediction = lr_model.predict(input_data)

        # ARIMA prediction (predicting the next time step)
        # Note: ARIMA prediction doesn't depend on input_data; it predicts based on past sales data
        arima_prediction = arima_fit.forecast(steps=1)

        return jsonify({
            'linear_regression_prediction': lr_prediction.tolist(),
            'arima_prediction': arima_prediction.tolist()
        })
    
    except Exception as e:
        print("error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
