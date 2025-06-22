def open_order(symbol,amount):
    try:
        sesion = spot.HTTP(endpoints='https://api.bybit.com', api_key=API, api_secret=secret)
        order = sesion.place_active_order(
            symbol=symbol,
            side="Buy",
            type="Market",
            qty=amount
            timeInForce="GTC"
        )
    except:
        return(print('Сделка не удалась'))
    
def conditional_purchase(symbol,amount):
    # параметры запроса
    symbol = symbol

    # отправляем запрос на получение текущей цены
    url = f'https://api.bybit.com/v2/public/tickers?symbol={symbol}'
    response = requests.get(url)

    # преобразование данных в словарь
    data = json.loads(response.text)['result'][0]
    

