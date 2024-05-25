import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Feedstuff Prediction 📈", page_icon="📈")

st.title("Feedstuff Price Prediction and Forecast")

# Load the cleaned dataset
file_path = 'cleaned_feeder.csv'
data = pd.read_csv(file_path)

# Convert the 'TLIST(M1)' column to a datetime format
data['Date'] = pd.to_datetime(data['TLIST(M1)'], format='%Y%m')
data.set_index('Date', inplace=True)

# Sidebar options
st.sidebar.header("Options")
feedstuff = st.sidebar.selectbox('Select Feedstuff:', data['Type of Feedstuff'].unique())
chart_type = st.sidebar.selectbox('Select Chart Type:', ['Line Chart', 'Bar Chart'])
show_forecast = st.sidebar.checkbox('Show Forecast')
show_best_time_to_buy = st.sidebar.checkbox('Click here to find out the best time to buy')

# Filter data for the selected feedstuff
feedstuff_data = data[data['Type of Feedstuff'] == feedstuff].sort_index()

# Handle missing values by filling them with the mean of the column
feedstuff_data['VALUE'].fillna(feedstuff_data['VALUE'].mean(), inplace=True)

# Ensure no NaN values are present
if feedstuff_data['VALUE'].isnull().sum() > 0:
    st.error("NaN values found in the dataset after filling. Please check the data.")
    st.stop()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(feedstuff_data['VALUE'].values.reshape(-1, 1))

# Create sequences for the LSTM model
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12  # Use 12 months of data to predict the next month
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model with Dropout layers and L2 regularization
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(LSTM(50, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with reduced verbosity
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=2)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Invert the normalization for the actual values
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the mean squared error
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')

# Plot the predictions against the actual values
if chart_type == 'Line Chart':
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feedstuff_data.index[-len(y_test):], y=y_test.flatten(), mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=feedstuff_data.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicted', line=dict(color='red')))
else:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=feedstuff_data.index[-len(y_test):], y=y_test.flatten(), name='Actual'))
    fig.add_trace(go.Bar(x=feedstuff_data.index[-len(predictions):], y=predictions.flatten(), name='Predicted', marker_color='red'))

fig.update_layout(title=f'{feedstuff} Price Prediction using LSTM', xaxis_title='Date', yaxis_title='Price (Euro per Tonne)')
st.plotly_chart(fig)

# Forecast future values if the checkbox is selected
if show_forecast:
    last_sequence = scaled_data[-seq_length:]

    # Forecast function
    def forecast_lstm(model, last_sequence, steps):
        forecast = []
        current_sequence = last_sequence
        for _ in range(steps):
            current_sequence = current_sequence.reshape((1, seq_length, 1))
            next_value = model.predict(current_sequence)
            forecast.append(next_value[0, 0])
            current_sequence = np.append(current_sequence[:, 1:, :], next_value.reshape((1, 1, 1)), axis=1)
        return np.array(forecast)

    # Calculate the number of steps to forecast
    last_date = feedstuff_data.index[-1]
    forecast_end_date = pd.to_datetime('2024-12-01')
    steps_to_forecast = (forecast_end_date.year - last_date.year) * 12 + (forecast_end_date.month - last_date.month)

    # Forecast future values
    future_forecast = forecast_lstm(model, last_sequence, steps_to_forecast)
    future_forecast = scaler.inverse_transform(future_forecast.reshape(-1, 1))

    # Create a date range for the forecast
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps_to_forecast, freq='M')

    # Add a column with cash emojis representing the prices
    def price_to_emoji(price):
        if price < np.percentile(future_forecast, 33):
            return '💵'
        elif price < np.percentile(future_forecast, 66):
            return '💵💵'
        else:
            return '💵💵💵'

    forecast_df = pd.DataFrame(future_forecast, index=forecast_dates, columns=['Forecasted Price'])
    forecast_df['Price Category'] = forecast_df['Forecasted Price'].apply(price_to_emoji)

    # Plot the forecast values
    if chart_type == 'Line Chart':
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=feedstuff_data.index, y=feedstuff_data['VALUE'], mode='lines', name='Actual'))
        fig_forecast.add_trace(go.Scatter(x=feedstuff_data.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicted', line=dict(color='red')))
        fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=future_forecast.flatten(), mode='lines', name='Forecast', line=dict(color='green')))
    else:
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Bar(x=feedstuff_data.index, y=feedstuff_data['VALUE'], name='Actual'))
        fig_forecast.add_trace(go.Bar(x=feedstuff_data.index[-len(predictions):], y=predictions.flatten(), name='Predicted', marker_color='red'))
        fig_forecast.add_trace(go.Bar(x=forecast_dates, y=future_forecast.flatten(), name='Forecast', marker_color='green'))

    fig_forecast.update_layout(title=f'{feedstuff} Price Prediction and Forecast using LSTM', xaxis_title='Date', yaxis_title='Price (Euro per Tonne)')
    st.plotly_chart(fig_forecast)

    # Display the forecasted values with emojis
    st.write(forecast_df)

    # Determine the best time to buy if checkbox is selected
    if show_best_time_to_buy:
        best_time_to_buy = forecast_df['Forecasted Price'].idxmin()
        st.markdown(f"""
            <div style='background-color: rgba(255, 0, 0, 0.15); padding: 10px; border-radius: 5px;'>
                <h3 style='color: black;'>🎉 The best time to buy {feedstuff} is in <b>{best_time_to_buy.strftime('%B %Y')}</b>! 🎉</h3>
            </div>
            """, unsafe_allow_html=True)
        st.balloons()
