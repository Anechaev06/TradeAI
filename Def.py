# Импорт библиотек
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
import signal

#ADX
def adx(high, low, close, n=14):
    """
    Расчет индикатора ADX и DI+ и DI-

    Аргументы:
    high -- столбец с максимальными ценами
    low -- столбец с минимальными ценами
    close -- столбец с ценами закрытия
    n -- период

    Возвращает:
    adx -- столбец со значениями индикатора ADX
    di_pos -- столбец со значениями индикатора DI+
    di_neg -- столбец со значениями индикатора DI-
    """
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_pos = high - high.shift()
    dm_neg = low.shift() - low
    dm_pos[dm_pos < 0] = 0
    dm_neg[dm_neg < 0] = 0
    
    # Directional Indicator
    di_pos = 100 * dm_pos.ewm(alpha=1/n, min_periods=n).mean() / tr.ewm(alpha=1/n, min_periods=n).mean()
    di_neg = 100 * dm_neg.ewm(alpha=1/n, min_periods=n).mean() / tr.ewm(alpha=1/n, min_periods=n).mean()
    
    # ADX
    dx = 100 * abs(di_pos - di_neg) / (di_pos + di_neg)
    adx = dx.ewm(alpha=1/n, min_periods=n).mean()
    
    return adx, di_pos, di_neg

def get_prev_data15():
    # Получаем данные с интервалом в 15 минут
    symbol = "BTCUSD"
    interval = "15"
    fields = "open_time,open,high,low,close,volume"

    # Время и запрос
    start_time = int((datetime.datetime.now() - datetime.timedelta(minutes=27)).timestamp())
    end_time = datetime.datetime.now()
    url = f"https://api.bybit.com/v2/public/kline/list?symbol={symbol}&interval={interval}&from={start_time}&to={end_time}&limit=200&fields={fields}"
    response = requests.get(url, headers={"api-key": API, "sign": secret})
    data = json.loads(response.text)
    dfn = pd.DataFrame(data['result'])

    # Преобразуем время в формат datetime
    dfn['open_time'] = pd.to_datetime(dfn['open_time'], unit='s')

    # Преобразуем значения столбцов в числа с плавающей точкой и удаляем дубликаты
    return(dfn)

def macd(df, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = df['close'].ewm(span=fast_period, min_periods=fast_period).mean()
    ema_slow = df['close'].ewm(span=slow_period, min_periods=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def get_historical_data(symbol, interval, days):
    fields = "open_time,open,high,low,close,volume"

    # Время
    now = datetime.datetime.now()
    start_of_today = datetime.datetime(now.year, now.month, now.day)
    start_of_yesterday = start_of_today - datetime.timedelta(days=days)

    # Начальное и конечное время для запроса данных
    start_time = str(int(start_of_yesterday.timestamp()))
    end_time = str(int(start_of_today.timestamp()))

    # Запрос первой порции данных
    url = f"https://api.bybit.com/v2/public/kline/list?symbol={symbol}&interval={interval}&from={start_time}&to={end_time}&limit=200&fields={fields}"
    response = requests.get(url, headers={"api-key": API, "sign": secret})
    data = json.loads(response.text)
    df = pd.DataFrame(data['result'])

    # Запрос оставшихся данных
    while len(data['result']) == 200:
        start_time = str(int(data['result'][-1]['open_time']) + 60)  # Новое начальное время
        url = f"https://api.bybit.com/v2/public/kline/list?symbol={symbol}&interval={interval}&from={start_time}&to={end_time}&limit=200&fields={fields}"
        response = requests.get(url, headers={"api-key": API, "sign": secret})
        data = json.loads(response.text)
        df = pd.concat([df, pd.DataFrame(data['result'])], axis=0, ignore_index=True)

    # Преобразуем время в формат datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='s')

    # Устанавливаем индекс по времени
    df = df.set_index('open_time')

    # Удаляем столбцы "symbol" и "interval"
    df = df.drop(['symbol', 'interval'], axis=1)

    # Преобразуем значения столбцов в числа с плавающей точкой и удаляем дубликаты
    df = df.astype(float)
    df.drop_duplicates(inplace=True)

    return df

def calculate_profit(df,initial_cash, commission, days):
    position = 'out'
    # Устанавливаем начальную сумму денег и количество криптовалют, которые у нас есть
    initial_cash2 = initial_cash
    initial_coins = 0

    # Создаем список для хранения изменений стоимости криптовалют
    coin_prices = []

    # Инициализируем счетчики для статистики
    total_trades = 0
    successful_trades = 0
    unsuccessful_trades = 0

    # Проходим по строкам таблицы
    for index, row in df.iterrows():
        # Если есть позиция на покупку
        if position == 'long':
            # Проверяем, если есть сигнал на продажу
            if row['signal'] == 'sell':
                # Продаем криптовалюты
                coins_sold = initial_coins
                coin_prices.append(row['close'])
                initial_cash += coins_sold * row['close'] * (1 - commission)
                initial_coins = 0
                position = 'out'

                # Обновляем счетчики статистики
                total_trades += 1
                if len(coin_prices) > 1 and coin_prices[-1] > coin_prices[-2]:
                    successful_trades += 1
                else:
                    unsuccessful_trades += 1

        # Если нет позиции на покупку, но есть сигнал на покупку
        elif position == 'out' and row['signal'] == 'buy':
            # Покупаем криптовалюты
            coins_bought = (initial_cash * (1 - commission)) / row['close']
            coin_prices.append(row['close'])
            initial_coins += coins_bought
            initial_cash = 0
            position = 'long'

    # Если остались криптовалюты в конце, продаем их по последней известной цене
    if initial_coins > 0:
        coins_sold = initial_coins
        coin_prices.append(df.iloc[-1]['close'])
        initial_cash += coins_sold * df.iloc[-1]['close'] * (1 - commission)
        initial_coins = 0
        position = 'out'

    # Вычисляем изменение суммы денег
    profit = initial_cash - initial_cash2

    # Вычисляем win rate
    win_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0

    # Выводим результаты
    print("initial_cash2:", initial_cash2)
    print("Profit: $", profit)
    print("Total Trades:", total_trades)
    print("Successful Trades:", successful_trades)
    print("Unsuccessful Trades:", unsuccessful_trades)
    print("Win Rate: {:.2f}%".format(win_rate))
    print("commission:",commission)
    print("days:",days)

def make_graf(df):
    # Создаем объекты go.Scatter для линии MACD, сигнальной линии и гистограммы
    macd_line = go.Scatter(x=df.index, y=df['macd_line'], name='MACD', line=dict(color='blue', width=1), opacity=0.8)
    signal_line = go.Scatter(x=df.index, y=df['signal_line'], name='Signal', line=dict(color='red', width=1), opacity=0.8)
    histogram = go.Bar(x=df.index, y=df['histogram'], name='Histogram', marker=dict(color=df['histogram'], colorscale='RdYlGn', line=dict(color='gray', width=0.5)), opacity=0.8)

    # Создаем объект субграфика
    fig = make_subplots(rows=11, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Строим график свечей
    candlestick = go.Candlestick(x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='candlestick')
    fig.add_trace(candlestick, row=1, col=1)

    # Строим график объема
    bar = go.Bar(x=df.index,
                y=df['volume'],
                name='volume',
                marker=dict(color=df['close'] - df['open'],
                            colorscale='RdYlGn',
                            line=dict(color='gray', width=0.5)),
                opacity=0.8)
    fig.add_trace(bar, row=4, col=1)

    # Строим график ADX
    adx = go.Scatter(x=df.index,
                    y=df['ADX'],
                    name='ADX',
                    line=dict(color='blue', width=1),
                    opacity=0.8)
    fig.add_trace(adx, row=5, col=1)

    # Строим график DI+
    dip = go.Scatter(x=df.index,
                    y=df['DI+'],
                    name='DI+',
                    line=dict(color='green', width=1),
                    opacity=0.8)
    fig.add_trace(dip, row=5, col=1)

    # Строим график DI-
    din = go.Scatter(x=df.index,
                    y=df['DI-'],
                    name='DI-',
                    line=dict(color='red', width=1),
                    opacity=0.8)
    fig.add_trace(din, row=5, col=1)

    # Добавляем график MACD
    fig.add_trace(macd_line, row=6, col=1)
    fig.add_trace(signal_line, row=6, col=1)
    fig.add_trace(histogram, row=6, col=1)

    # Добавляем график стохастиков
    stochastics_k = go.Scatter(x=df.index, y=df['%K'], name='%K', line=dict(color='purple', width=1), opacity=0.8)
    stochastics_d = go.Scatter(x=df.index, y=df['%D'], name='%D', line=dict(color='orange', width=1), opacity=0.8)
    fig.add_trace(stochastics_k, row=7, col=1)
    fig.add_trace(stochastics_d, row=7, col=1)

    # Строим график RSI
    rsi_line = go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange', width=1), opacity=0.8)
    fig.add_trace(rsi_line, row=8, col=1)

    # Строим график STC
    stc_line = go.Scatter(x=df.index, y=df['stc'], name='STC', line=dict(color='purple', width=1), opacity=0.8)
    fig.add_trace(stc_line, row=9, col=1)

    # Строим график STC2
    stc_line2 = go.Scatter(x=df.index, y=df['STC'], name='STC', line=dict(color='purple', width=1))
    fig.add_trace(stc_line2, row=10, col=1)

    # Создаем списки с координатами сигналов покупки и продажи
    buy_signals = df.loc[df['signal'] == 'buy']
    sell_signals = df.loc[df['signal'] == 'sell']

    # Строим график сигналов покупки и продажи
    if not buy_signals.empty:
        buy_scatter = go.Scatter(x=buy_signals.index, y=buy_signals['close'],
                            name='buy', mode='markers', 
                            marker=dict(color='green', size=10),
                            hovertext=buy_signals['score'])
        fig.add_trace(buy_scatter, row=1, col=1)
    if not sell_signals.empty:
        sell_scatter = go.Scatter(x=sell_signals.index, y=sell_signals['close'],
                            name='sell', mode='markers', 
                            marker=dict(color='red', size=10),
                            hovertext=sell_signals['score'])
        fig.add_trace(sell_scatter, row=1, col=1)

    # Настраиваем макет графика
    fig.update_layout(title='BTCUSD',
                    yaxis_title='Price, USD',
                    height=1000,
                    xaxis_rangeslider_visible=True)

    # Отображаем график
    fig.show()

def calculate_stochastic(df, k_period, d_period, smoothing_period):
    """
    Рассчитывает значения %K и %D для индикатора двойного стохастика.
    :param df: DataFrame с историческими данными.
    :param k_period: Период для расчета %K.
    :param d_period: Период для расчета %D.
    :param smoothing_period: Период сглаживания для расчета %D.
    :return: DataFrame с добавленными столбцами %K и %D.
    """
    # Рассчитываем значения %K
    df['lowest_low'] = df['low'].rolling(k_period).min()
    df['highest_high'] = df['high'].rolling(k_period).max()
    df['%K'] = (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']) * 100

    # Рассчитываем значения %D
    df['%D'] = df['%K'].rolling(d_period).mean()

    # Применяем сглаживание к %D
    df['%DS'] = df['%D'].rolling(smoothing_period).mean()

    df.drop(['lowest_low', 'highest_high'], axis=1, inplace=True)

    return df

def calculate_rsi(df, period=14):
    """
    Рассчитывает значения RSI (Relative Strength Index) для заданного DataFrame с 5-минутными данными.
    :param df: DataFrame с историческими данными.
    :param period: Период для расчета RSI (по умолчанию 14).
    :return: DataFrame с добавленным столбцом RSI.
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(period).mean()
    average_loss = loss.rolling(period).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    df['RSI'] = rsi
    return df

def remove_recent_days(df, days):
    # Определяем дату, начиная с которой нужно удалить данные
    end_date = df.index[-1] - pd.DateOffset(days=days)

    # Фильтруем датафрейм по дате
    df = df[df.index <= end_date]

    return df

def calculate_stc(df, fast_period=23, slow_period=50, smooth_period=10):
    df['h'] = df['high'].rolling(fast_period).max()
    df['l'] = df['low'].rolling(fast_period).min()
    df['c'] = df['close']
    df['roc'] = ((df['c'] - df['l']) / (df['h'] - df['l'])) * 100
    df['stc'] = df['roc'].rolling(slow_period).mean().rolling(smooth_period).mean()
    df.drop(['h', 'l', 'c', 'roc'], axis=1, inplace=True)
    return df

def calculate_stc2(df, length, fast_length, slow_length, aaa):
    """
    Рассчитывает значения индикатора STC.
    :param df: DataFrame с историческими данными.
    :param length: Длина для расчета.
    :param fast_length: Длина быстрой скользящей средней.
    :param slow_length: Длина медленной скользящей средней.
    :param aaa: Значение AAA.
    :return: DataFrame с добавленным столбцом STC.
    """
    def aaaa(bb, bbb, bbbb):
        fast_ma = df['close'].ewm(span=bbb).mean()
        slow_ma = df['close'].ewm(span=bbbb).mean()
        aaaa = fast_ma - slow_ma
        return aaaa

    def aaaaa(eeeee, bb, bbb):
        aaa = 0.5
        ccccc = 0.0
        ddd = 0.0
        dddddd = 0.0
        eeee = 0.0

        bbbbbb = aaaa(df['close'], bb, bbb)
        ccc = bbbbbb.rolling(window=eeeee).min()
        cccc = bbbbbb.rolling(window=eeeee).max() - ccc
        ccccc = (bbbbbb - ccc) / cccc * 100
        ddd = ccccc.ewm(span=eeeee).mean()
        dddd = ddd.rolling(window=eeeee).min()
        dddddd = ddd.rolling(window=eeeee).max() - dddd
        eeee = (ddd - dddd) / dddddd * 100

        return eeee

    stc = aaaaa(length, fast_length, slow_length)

    df['STC'] = stc

    return df

def calculate_ema(df, period):
    # Расчет значения EMA
    ema = df.ewm(span=period, adjust=False).mean()
    return ema




