from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime as dt

app = Flask(__name__)

# Load the stock data (for simplicity, we'll use synthetic data here)
dates = pd.date_range(start='2022-01-01', periods=100, freq='B')
prices = np.sin(np.linspace(-3, 3, 100)) * 10 + 150
data = pd.DataFrame({'Date': dates, 'Price': prices})

# Feature engineering: Convert dates to numerical values
data['Date'] = data['Date'].map(dt.datetime.toordinal)

# Train a simple linear regression model
model = LinearRegression()
model.fit(data[['Date']], data['Price'])

@app.route('/predict', methods=['GET'])
def predict():
    # Get the date parameter from the request
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'No date provided'}), 400
    
    # Convert the date string to an ordinal number
    try:
        date = dt.datetime.strptime(date_str, '%Y-%m-%d').toordinal()
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    
    # Make a prediction
    predicted_price = model.predict([[date]])[0]
    
    return jsonify({'date': date_str, 'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
