def adx(high, low, close, n=14):
    """
    Расчет индикатора ADX

    Аргументы:
    high -- столбец с максимальными ценами
    low -- столбец с минимальными ценами
    close -- столбец с ценами закрытия
    n -- период

    Возвращает:
    adx -- столбец со значениями индикатора ADX
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
    
    return adx