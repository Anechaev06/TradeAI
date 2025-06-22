import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_bybit_rates(coin, method, trade_type):
    url = f"https://www.bybit.com/fiat/trade/otc/?actionType={trade_type}&token={coin}&fiat=RUB&paymentMethod={method}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Здесь вы можете использовать селекторы CSS или методы BeautifulSoup для извлечения данных о 2p2 сделках
        # Например, вы можете найти элементы таблицы или другие элементы, содержащие информацию о сделках
        # Верните список цен, как в вашем исходном коде
        print (prices)
        return [prices]
    else:
        print(f"Ошибка: {response.status_code}")
        return []

#Arrays
coins = ['ETH', 'BTC', 'USDT']
payment_method = ["TinkoffNew", "RosBankNew"]
prices = []

for method in payment_method:
    for coin in coins:
        #Append Buy Price
        for price in get_bybit_rates(coin, method, 1):  # 1 for BUY
            prices.append(price)

        #Append Sell Price
        for price in get_bybit_rates(coin, method, 2):  # 2 for SELL
            prices.append(price)

#Dataframe Organization
data = {'Platform': ['Bybit'] * 36, 
        'Payment Method': ['TinkoffNew']*18 + ['RosBankNew']*18,
        'Coin': ['ETH']*12 + ['BTC']*12 + ['USDT']*12, 
        'Operation Type': ['BUY']*6 + ['SELL']*6 +
                          ['BUY']*6 + ['SELL']*6 +
                          ['BUY']*6 + ['SELL']*6,
        'Prices': [price for price in prices]}
df = pd.DataFrame(data)

#Print
print(df)
