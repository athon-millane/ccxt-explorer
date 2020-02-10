import streamlit as st
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data import load_data, get_keys, get_names, get_market_cap, get_history
from cointegration import find_cointegrated_pairs

def load_page():

    markets, exception = load_data()

    if exception:
        st.sidebar.text(str(exception))
        st.title("⭕️The data was not correctly loaded")
        return

    title = st.empty()
    st.sidebar.title("Distributions of Time Series")
    resolution = st.sidebar.selectbox('Select resolution', ['1d', '1h', '1m'], index=2)

    keys = get_market_cap(markets, 'USDT')

    data_close = get_history(keys[:10], timeframe=resolution, limit=2000)
    data_close = data_close.dropna()
    scores, pvalues, pairs = find_cointegrated_pairs(data_close)

    data = {'market1': [k[0] for k in pairs.keys()],
            'market2': [k[1] for k in pairs.keys()],
            'p-value': [v[1] for v in pairs.values()]}
    
    cm = sns.light_palette("green", as_cmap=True)
    df = pd.DataFrame(data).sort_values('p-value').reset_index(drop=True)
    df_ = df.copy()
    df = df.style.background_gradient(subset=['p-value'], cmap=cm)
    df_['pair'] = df_["market1"].map(str) + ' -- ' + df_["market2"]
    st.sidebar.table(df)

    # Choose pairs
    pair = st.sidebar.selectbox('Select pair', df_['pair'])
    [market1, market2] = pair.split(' -- ')
    st.header('Pairs Cointegration: {} vs {}'.format(market1, market2))
    
    # Fit residual
    x1 = data_close[market1].values
    x2 = data_close[market2].values
    r = sm.OLS(x1,x2).fit().params[0]

    df_x = pd.DataFrame({'x1':x1, 'r*x2':r*x2, 'z':x1 - r*x2}, index=data_close.index)

    # Plot
    st.latex(r'\text{Pairs, } x_1 \text{and } r\times x_2')
    fig1,axs = plt.subplots(1, 2, figsize=(16,5))
    _=sns.lineplot(data=df_x[['x1', 'r*x2']], ax=axs[0]).set_title("time series")
    _=sns.distplot(df_x[['x1']], ax=axs[1], bins=100, color='blue').set_title("distribution")
    _=sns.distplot(df_x[['r*x2']], ax=axs[1], bins=100, color='orange')
    plt.tight_layout()
    st.pyplot(fig1) 

    st.latex(r'\text{Regressed residual } z = x_1 - (r\times x_2)')
    fig2,axs = plt.subplots(1, 2, figsize=(16,5))
    _=sns.lineplot(data=df_x[['z']], ax=axs[0], palette=['green']).set_title("time series")
    _=sns.distplot(df_x[['z']], ax=axs[1], bins=100, color='green').set_title("distribution")
    plt.tight_layout()
    st.pyplot(fig2) 

    st.latex(r'\text{First difference of residual } z^\prime')
    fig3,axs = plt.subplots(1, 2, figsize=(16,5))
    _=sns.lineplot(data=df_x.rename({'z':"z\'"}, axis=1)[["z\'"]].diff(), ax=axs[0], palette=['purple']).set_title("time series")
    _=sns.distplot(df_x[['z']].diff(), ax=axs[1], bins=100, color='purple').set_title("distribution")
    plt.tight_layout()
    st.pyplot(fig3)