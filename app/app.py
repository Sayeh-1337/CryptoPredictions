import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import requests
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go

# Function to fetch and process Binance Klines data
def get_bars(symbol, interval='30m'):
    root_url = 'https://api.binance.com/api/v3/klines'
    url = f"{root_url}?symbol={symbol}&interval={interval}"
    response = requests.get(url)
    
    if response.ok:
        data = response.json()
        df = pd.DataFrame(
            data,
            columns=[
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume',
                'Ignore'
            ]
        )
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        return df
    else:
        st.error("Error fetching data from the API")
        return None

# Function to fit a Prophet model and forecast future mean prices
def forecast_token(symbol, interval='30m', periods=10, target_datetime=None):
    # Fetch the data
     # Create a spinner while waiting for data retrieval
    with st.spinner("Fetching data..."):
        # Fetch the data
        df = get_bars(symbol, interval)
    # If data is successfully fetched, proceed with forecasting
    if df is not None:
        # Prepare the data for Prophet
        df['Mean'] = (df['Low'] + df['High']) / 2
        prophet_data = df.reset_index()[['Open time', 'Mean']].rename(columns={'Open time': 'ds', 'Mean': 'y'})
        prophet_data.dropna()
        
        # Create and fit the Prophet model
        model = Prophet()
        model.fit(prophet_data)
        
        # Ensure the target datetime is a datetime object
        if isinstance(target_datetime, str):
            target_datetime = pd.to_datetime(target_datetime)
        
        # Create a future dataframe starting from the user-specified datetime
        future_dates = pd.date_range(start=target_datetime, periods=periods + 1, freq='30T')  # +1 because the start date is inclusive
        future = pd.DataFrame(future_dates, columns=['ds'])
        
        # Make predictions
        forecast = model.predict(future)
        
        # Plot the historical data and forecast
        fig1 = model.plot(forecast)
        plt.title(f"Mean Price Forecast ({symbol})")
        plt.xlabel("Date")
        plt.ylabel("Mean Price")
        plt.grid(True)
        st.pyplot(fig1)

        # Plotting the results
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot historical data
        ax.plot(prophet_data['ds'], prophet_data['y'], 'ko-', label='Historical Mean Prices')
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecasted Mean Prices')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)
        ax.set_title(f"Mean Price Forecast for {symbol}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Price")
        ax.grid(True)
        ax.legend()
        
        st.pyplot(fig)
        # Plot the forecast components (trend, weekly, yearly patterns)
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # Plot the forecasted mean prices with trend arrows
        fig3, ax = plt.subplots(figsize=(10, 6))
        # Plot historical data
        ax.plot(prophet_data['ds'], prophet_data['y'], 'ko-', label='Historical Mean Prices')
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecasted Mean Prices')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)

        # Add trend arrows
        trend_arrows = forecast['yhat'].diff().apply(lambda x: 'up' if x >= 0 else 'down').map({'up': 'green', 'down': 'red'}).tolist()
        for i, arrow_color in enumerate(trend_arrows):
            if arrow_color == 'green':
                ax.annotate('↑', (forecast.loc[i, 'ds'], forecast.loc[i, 'yhat']), color=arrow_color, fontsize=12)
            else:
                ax.annotate('↓', (forecast.loc[i, 'ds'], forecast.loc[i, 'yhat']), color=arrow_color, fontsize=12)

        ax.set_title(f"Mean Price Forecast for {symbol}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Price")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig3)
        # # Plot the historical data and forecast
        # fig1 = model.plot(forecast)
        # st.plotly_chart(fig1)

        # # Plot forecasted mean prices
        # fig2 = go.Figure()
        # fig2.add_trace(go.Scatter(x=prophet_data['ds'], y=prophet_data['y'], mode='lines', name='Historical Mean Prices'))
        # fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Mean Prices'))
        # fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='Confidence Interval', fillcolor='rgba(0, 176, 246, 0.2)'))
        # fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='Confidence Interval', fillcolor='rgba(0, 176, 246, 0.2)'))
        # fig2.update_layout(title=f"Mean Price Forecast ({symbol})", xaxis_title="Date", yaxis_title="Mean Price")
        # st.plotly_chart(fig2)

        # # Add trend arrows
        # trend_arrows = forecast['yhat'].diff().apply(lambda x: 'up' if x >= 0 else 'down').map({'up': 'green', 'down': 'red'}).tolist()
        # for i, arrow_color in enumerate(trend_arrows):
        #     if arrow_color == 'green':
        #         fig2.add_annotation(x=forecast.loc[i, 'ds'], y=forecast.loc[i, 'yhat'], text='↑', showarrow=False, font=dict(color='green', size=12))
        #     else:
        #         fig2.add_annotation(x=forecast.loc[i, 'ds'], y=forecast.loc[i, 'yhat'], text='↓', showarrow=False, font=dict(color='red', size=12))

        # st.plotly_chart(fig2)

        # Return and display forecast data
        predicted_mean_prices = forecast[['ds', 'yhat']]
        predicted_mean_prices = predicted_mean_prices.rename(columns={'ds': 'Date Time', 'yhat': 'Predicted Mean Price'})
        st.write(f"**Predicted Mean Price for {symbol} on {target_datetime}:**")
        if not predicted_mean_prices.empty:
            #st.write(predicted_mean_prices)
            st.dataframe(predicted_mean_prices)
        else:
            st.write("No prediction available for the specified datetime.")


# Streamlit App
def main():
    # List of tokens to forecast
    tokens = [
        "ADAUSDT", "ALGOUSDT", "ARBUSDT", "AVAXUSDT", "BNBUSDT",
        "BTCUSDT", "DOGEUSDT", "ETHUSDT", "FILUSDT",
        "LTCUSDT", "MATICUSDT", "SOLUSDT", "XLMUSDT"
    ]

    # Streamlit Sidebar
    st.sidebar.title("Cryptocurrency Forecast")
    selected_token = st.sidebar.selectbox("Select a Token", tokens)

    # # Date selection
    # target_date = st.sidebar.date_input("Select Prediction Date", value=pd.to_datetime("today"))
    # # Time selection
    # target_time = st.sidebar.time_input("Select Prediction Time", value=pd.to_datetime("now").time())
    # Date and time selection
    target_date = st.sidebar.date_input("Select Prediction Date")
    target_time = st.sidebar.time_input("Select Prediction Time")


    # Combine date and time into a single datetime object
    target_datetime = datetime.combine(target_date, target_time)

    # Main Content
    st.title("Cryptocurrency Price Forecast using Prophet")
    st.write(f"### Forecast for {selected_token}")
    forecast_token(selected_token, target_datetime=target_datetime)

if __name__ == "__main__":
    main()