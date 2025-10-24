from flask import Flask, request, jsonify
from prophet import Prophet
from river import time_series
import pandas as pd

app = Flask(__name__)

@app.route('/forecast/prophet', methods=['POST'])
def forecast_prophet():
    data = request.get_json()
    df = pd.DataFrame(data['history'])
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=data['periods'])
    forecast = model.predict(future)
    result = [{"date": str(row["ds"]), "prophet": row["yhat"]} for index, row in forecast.iterrows()]
    return jsonify(result)

@app.route('/forecast/holt', methods=['POST'])
def forecast_holt():
    data = request.get_json()
    series = [row["y"] for row in data['history']]
    model = time_series.HoltWinters()
    forecast = []
    for value in series:
        model = model.learn_one(value)
    for _ in range(data['periods']):
        forecast.append(model.forecast(steps=1)[0])
    result = [{"holt": f} for f in forecast]
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
