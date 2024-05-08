import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from .creator import create_dataset

logger = logging.getLogger(__name__)


class BinanceDataset:
    dataset = []

    def __init__(self, main_features, start_date=None, end_date=None, window_size=10, args=None):
        # Fetching data from the folder
        self.folder_path = args.dataset_path     #"data/binance-data/data/spot/monthly/klines/"
        self.coins = args.crypto_symbols#["ADAUSDT", "ALGOUSDT", "ARBUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "FILUSDT", "LTCUSDT", "MATICUSDT", "SOLUSDT", "XLMUSDT"]

        dataframes = []
        for coin in self.coins:
            coin_folder_path = os.path.join(self.folder_path, coin, "30m")
            for file_name in os.listdir(coin_folder_path):
                if file_name.endswith(".zip"):
                    csv_file_name = os.path.splitext(file_name)[0]
                    csv_file_path = os.path.join(coin_folder_path, csv_file_name)
                    df = pd.read_csv(csv_file_path)
                    df = df.iloc[:, :6]  # Extracting the first six columns
                    dataframes.append(df)

        # Concatenating all dataframes
        df = pd.concat(dataframes)

        # Renaming the columns
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

        # Parsing timestamp as datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
      
        # Reordering the columns
        df = df[["timestamp", "low", "high", "open", "close", "volume"]]

        # Renaming the columns
        df.rename(columns={"timestamp": "Date"}, inplace=True)
        df.rename(columns={"low": "Low"}, inplace=True)
        df.rename(columns={"high": "High"}, inplace=True)
        df.rename(columns={"open": "Open"}, inplace=True)
        df.rename(columns={"close": "Close"}, inplace=True)
        df.rename(columns={"volume": "Volume"}, inplace=True)
        
        # Cleaning the data for any NaN or Null fields
        df = df.dropna()

        # Creating a new feature for better representing day-wise values
        df["Mean"] = (df["Low"] + df["High"]) / 2

        # Creating a copy for making small changes
        dataset_for_prediction = df.copy()
        dataset_for_prediction["Actual"] = dataset_for_prediction["Mean"].shift()
        dataset_for_prediction = dataset_for_prediction.dropna()

        # Setting timestamp as index
        dataset_for_prediction.set_index("Date", inplace=True)

        drop_cols = ['High', 'Low', 'Close', 'Open', 'Volume', 'Mean']
        for item in main_features:
            if item in drop_cols:
                drop_cols.remove(item)
        df = df.drop(drop_cols, axis=1)

        if start_date == '-1':
            start_date = df.iloc[0].Date
        else:
            start_date = datetime.strptime(str(start_date), '%Y-%m-%d %H:%M:%S')

        if end_date == '-1':
            end_date = df.iloc[-1].Date
        else:
            end_date = datetime.strptime(str(end_date), '%Y-%m-%d %H:%M:%S')

        start_index = 0
        end_index = df.shape[0] - 1
        for i in range(df.shape[0]):
            if df.Date[i] <= start_date:
                start_index = i

        for i in range(df.shape[0] - 1, -1, -1):
            if df.Date[i] >= end_date:
                end_index = i

        # prediction mean based upon open
        dates = df.Date[start_index:end_index]
        df = df.drop('Date', axis=1)
        arr = np.array(df)
        arr = arr[start_index:end_index]
        features = df.columns

        self.dataset, self.profit_calculator = create_dataset(arr, list(dates), look_back=window_size, features=features)

    def get_dataset(self):
        return self.dataset, self.profit_calculator