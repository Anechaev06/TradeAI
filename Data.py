import datetime
import requests
import json
from pybit import spot
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from pybit.exceptions import InvalidRequestError
import numpy as np
from settings import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#параметры запроса
symbol = 'BTCUSD'
interval = '15'
fields = "open_time,open,high,low,close,volume"

#создаем пустой DataFrame
df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

while True:
# получаем текущее время на сервере Bybit
    url = 'https://api.bybit.com/v2/public/time'
    response = requests.get(url)
    server_time = json.loads(response.text)['time_now']

    # преобразуем server_time в целочисленный тип
    server_time = int(float(server_time))

    # вычисляем время начала текущего бара
    start_time = int(time.time()) - 60

    # ожидаем ближайшей секунды текущей минуты
    current_time = datetime.datetime.fromtimestamp(server_time).strftime('%S.%f')
    wait_time = 59.0 - float(current_time)
    time.sleep(wait_time)

    # Запрос первой порции данных
    url = f"https://api.bybit.com/v2/public/kline/list?symbol={symbol}&interval={interval}&from={start_time}&limit=200&fields={fields}"
    response = requests.get(url, headers={"api-key": API, "sign": secret})
    data = json.loads(response.text)
    new_df = pd.DataFrame(data['result'])

    # преобразование времени в удобный для чтения формат
    new_df['open_time'] = pd.to_datetime(new_df['open_time'], unit='s')

    # добавляем новые данные в DataFrame
    new_data = new_df[~new_df['open_time'].isin(df['open_time'])]
    df = pd.concat([df, new_data], ignore_index=True)

    # удаляем колонку turnover
    df = df.drop(columns=["turnover","interval"])

    # Применение страты
    if not new_data.empty:
        print(df)
        # сохраняем данные в файл в формате CSV
        df.to_csv('data.csv', index=False)

