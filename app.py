import pandas as pd
import streamlit as st
import ccxt
import plotly.graph_objects as go
from datetime import datetime

@st.cache(allow_output_mutation=True)
def load_data():
    exchange = ccxt.huobipro()
    markets = exchange.load_markets()
    return exchange, markets

# @st.cache()
def get_keys(exchange, base='USDT'):
    keys = [k for k in exchange.markets.keys() if k[-len(base):] == base]
    return pd.Series(keys).sort_values().values

@st.cache()
def get_names():
    ex = ccxt.bittrex(); ex.load_markets()
    names = {c['id']: c['name'] for c in ex.currencies.values()}
    return names
    
@st.cache()
def get_ohlcv(market='BTC/USDT', timeframe='1d', limit=2000):
    exchange = ccxt.huobipro()
    ohlcv = exchange.fetch_ohlcv(market, timeframe, limit=limit)
    df = pd.DataFrame.from_records(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time']/1000, unit='s')
    df = df.set_index('time')
    return df

def plot_candlestick(df):
    layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
    up = (df['close'] - df['open'] > 0).values
    colours = ["#009933" if marker==True else "#ff0000" for marker in up]
    tr1 = go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], yaxis='y2', showlegend=False)
    tr2 = go.Bar(x=df.index, y=df['volume'], yaxis='y', showlegend=False, opacity=0.5, marker={"color":colours, "line":{"width":0}})       
    fig = go.Figure(data=[tr1,tr2], layout=layout)
    fig.update_layout(xaxis_rangeslider_visible=False, yaxis_showticklabels=False, yaxis_domain=[0,0.2], yaxis2={'domain':[0.2,0.8]})
    st.plotly_chart(fig)

def main():
    exchange, markets = load_data()
    names = get_names()

    title = st.empty()
    st.sidebar.title("Crypto Explorer")

    # OHLC Visualisation
    st.sidebar.subheader('Choose your asset:')
    base = st.sidebar.selectbox('Select base', ['USDT', 'BTC', 'ETH'])
    keys = get_keys(exchange, base=base)
    
    market = st.sidebar.selectbox('Select market', keys)
    resolution = st.sidebar.selectbox('Select resulution', ['1d', '1h', '1m'])

    code = market.split('/')[0]
    name = names[code] if code in names else code
    title.header(market + ' - ' + name)
    data = get_ohlcv(market, timeframe=resolution)
    range_ = st.sidebar.slider('Historical range', min_value=min([30, data.shape[0]]),
                        max_value=min([2000, data.shape[0]]), value=min([1000, int(data.shape[0]/2)]),  step=10)

    plot_candlestick(data[-range_:])

    # # Cointegration
    # st.sidebar.subheader('Investigate cointegration:')
    # asset1 = st.sidebar.selectbox('Pair 1:', keys)
    # asset2 = st.sidebar.selectbox('Pair 2:', keys)


if __name__ == '__main__':
    main()