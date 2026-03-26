from data import get_stock_data
from model import preprocess_data, create_dataset, build_model

import numpy as np
from dash import Dash, dcc, html
import plotly.graph_objs as go

# Step 1: Get data
data = get_stock_data("AAPL")

# Step 2: Preprocess
scaled_data, scaler = preprocess_data(data)

# Step 3: Create dataset
X, y = create_dataset(scaled_data)

X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Train model
model = build_model()
model.fit(X, y, epochs=3, batch_size=32)

# Step 5: Predict
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Step 6: Dashboard
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Eye 📈"),
    
    dcc.Graph(
        figure={
            'data': [
                go.Scatter(y=data['Close'], name='Actual'),
                go.Scatter(y=predictions.flatten(), name='Predicted')
            ],
            'layout': go.Layout(title='Stock Price Prediction')
        }
    )
])

if __name__ == '__main__':
    app.run(debug=True)
