#!/bin/bash

# This is a simple script to download klines hsitorical dataset by given parameters.

mkdir -p ./data/binance-data

python data_loader/binance_public_data_downloader/download-kline.py -t spot -i 30m -skip-daily 1 -startDate 2020-01-01 -s ADAUSDT ALGOUSDT ARBUSDT AVAXUSDT BNBUSDT BTCUSDT DOGEUSDT ETHUSDT FILUSDT LTCUSDT MATICUSDT SOLUSDT XLMUSDT

mv  data_loader/binance_public_data_downloader/data/ ./data/binance-data/.