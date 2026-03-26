# stock-eye
Real-time stock price prediction system using LSTM with interactive dashboard built using Plotly Dash.
# Stock Eye: LSTM-Based Stock Price Prediction System

## Overview
Stock Eye is a time-series forecasting system that predicts stock prices using Long Short-Term Memory (LSTM) neural networks. The project integrates data acquisition, preprocessing, model training, and visualization into a single pipeline with an interactive dashboard.

## Objectives
- To implement a deep learning model for stock price prediction
- To analyze financial time-series data
- To visualize model outputs using an interactive interface

## Methodology
1. Data Collection: Historical stock data is fetched using the Yahoo Finance API.
2. Data Preprocessing: Closing prices are normalized using MinMax scaling.
3. Sequence Generation: Time-series sequences are created for supervised learning.
4. Model Training: An LSTM network is trained to capture temporal dependencies.
5. Prediction: The model forecasts future price movements.
6. Visualization: Results are displayed using Plotly Dash.

## Tech Stack
- Python
- NumPy, Pandas
- TensorFlow / Keras (LSTM)
- Scikit-learn
- Plotly Dash

## Project Structure
