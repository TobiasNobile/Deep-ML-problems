def tick_bars(ticks: list, bar_size: int) -> list:
    """
    Sample tick data into tick bars.
    
    Args:
        ticks: List of tuples (timestamp, price, volume) representing individual trades
        bar_size: Number of ticks per bar
    
    Returns:
        List of dictionaries with keys: 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    """
    keys_order = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    bars =[]
    bar = {}
    sum_volume = 0
    bar_count = 1
    for tick in ticks:
        bar["timestamp"] = tick[0]
        price = tick[1]
        if price > bar.get("high", 0):
            bar["high"] = price
        if price < bar.get("low", float("inf")):
            bar["low"] = price

        sum_volume += tick[2]

        if bar_count == 1:
            bar["open"] = price
        if bar_count == bar_size :
            bar["close"] = price
            bar["volume"] = float(sum_volume)
            bar_order = {k:bar[k] for k in keys_order}
            bars.append(bar_order)
            sum_volume = 0
            bar_count = 1
            bar = {}
            continue
        
        bar_count += 1

    return bars
