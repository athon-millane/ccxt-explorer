import streamlit as st
import ccxt
from coinpaprika import client as Coinpaprika
import pandas as pd

# @st.cache(allow_output_mutation=True)
def load_data():
    exception = False
    try: 
        markets = ccxt.huobipro().load_markets()
        return markets, exception
    except Exception as exception:
        return False, False, exception

# @st.cache()
def get_keys(markets, base='USDT'):
    keys = [k for k in markets.keys() if k[-len(base):] == base]
    return pd.Series(keys).sort_values().values

# @st.cache()
def get_names():
    ex = ccxt.bittrex(); ex.load_markets()
    names = {c['id']: c['name'] for c in ex.currencies.values()}
    return names
    
# @st.cache()
def get_ohlcv(market='BTC/USDT', timeframe='1d', limit=2000):
    """
    Fetches ohlcv data and returns in dataframe format.
    """
    ohlcv = ccxt.huobipro().fetch_ohlcv(market, timeframe, limit=limit)
    df = pd.DataFrame.from_records(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time']/1000, unit='s')
    df = df.set_index('time')
    return df

# @st.cache()
def get_history(keys, timeframe='1h', limit=100, dimension='close'):
    """
    Will provide history across list of markets (keys) for a given dimension ('open','high','low','close','volume').
    Timeframe and historical limit also configurable.
    """
    return pd.concat([get_ohlcv(key, timeframe)['close'].rename(key) for key in keys], axis=1).tail(limit)


def top_markets(exchange='huobi', quote='USDT'):
    # client
    client = Coinpaprika.Client()
    coins  = client.coins()
    markets = client.exchange_markets(exchange, quotes="USD")
    return [m['pair'] for c in coins for m in markets if (m['base_currency_id'] == c['id'] and m['pair'].split('/')[-1] == quote)]


def get_market_cap(markets, base = None):
    """
    DEPRECATED
    Load coinmarketcap data as well as base names and return a sorted dict with {cap, name} tuple.
    """
    markets_ = ccxt.coinmarketcap().load_markets()
    caps = {k_:{'name':v['baseId'], 'cap':v['info']['market_cap_usd']} for k_ in markets for k,v in markets_.items() if (
        (k.split('/')[0] == k_.split('/')[0]) and (k.split('/')[1][:3] == k_.split('/')[1][:3]))}
    keys = sorted(caps.keys(), key=lambda x: float(caps[x]['cap']), reverse=True)
    if base is not None:
        keys = [k for k in keys if k[-len(base):] == base]
    return keys