import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from utils import load_data, get_keys, get_names, get_ohlcv

def load_page():

    markets, exception = load_data()

    if exception:
        st.sidebar.text(str(exception))
        st.title("⭕️The data was not correctly loaded")
        return

    names = get_names()

    title = st.empty()
    st.sidebar.title("Crypto Explorer")

    # OHLC Visualisation
    st.sidebar.subheader('Choose your asset:')
    base = st.sidebar.selectbox('Select base', ['USDT', 'BTC', 'ETH'])
    keys = get_keys(markets, base=base)
    
    market = st.sidebar.selectbox('Select market', keys)
    resolution = st.sidebar.selectbox('Select resulution', ['1d', '1h', '1m'])

    code = market.split('/')[0]
    name = names[code] if code in names else code
    title.header(market + ' - ' + name)
    data = get_ohlcv(market, timeframe=resolution)
    range_ = st.sidebar.slider('Historical range', min_value=min([30, data.shape[0]]),
                        max_value=min([2000, data.shape[0]]), value=min([1000, int(data.shape[0]/2)]),  step=10)

    plot_candlestick(data[-range_:])

def plot_candlestick(df):
    layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
    up = (df['close'] - df['open'] > 0).values
    colours = ["#009933" if marker==True else "#ff0000" for marker in up]
    tr1 = go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], yaxis='y2', showlegend=False)
    tr2 = go.Bar(x=df.index, y=df['volume'], yaxis='y', showlegend=False, opacity=0.5, marker={"color":colours, "line":{"width":0}})       
    fig = go.Figure(data=[tr1,tr2], layout=layout)
    fig.update_layout(xaxis_rangeslider_visible=False, yaxis_showticklabels=False, yaxis_domain=[0,0.2], yaxis2={'domain':[0.2,0.8]})
    st.plotly_chart(fig)