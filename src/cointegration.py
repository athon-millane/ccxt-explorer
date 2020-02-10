import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils.data import load_data, get_keys, get_names, get_market_cap, get_history

def load_page():

    markets, exception = load_data()

    if exception:
        st.sidebar.text(str(exception))
        st.title("⭕️The data was not correctly loaded")
        return

    title = st.empty()

    # Total cointegration
    st.sidebar.title("Total Cointegration")
    base = st.sidebar.selectbox('Select base', ['USDT', 'BTC', 'ETH'])
    resolution = st.sidebar.selectbox('Select resulution', ['1d', '1h', '1m'])
    window = st.sidebar.slider('Select window', min_value=100, max_value=1000, value=500)
    keys = get_market_cap(markets, base)
    range_ = st.sidebar.slider('Choose top n assets by volume:', min_value=10, max_value=len(keys))
    data_close = get_history(keys[:range_], timeframe=resolution, limit=window)
    data_close = data_close.dropna()
    scores, pvalues, pairs = find_cointegrated_pairs(data_close)
    
    title.header('Cointegration matrix')
    st.text('Top {} assets.\nResolution: {}\nHistorical time steps: {}'.format(range_, resolution, data_close.shape[0]))

    # plot heatmap
    fig,ax = plt.subplots()
    g = sns.heatmap(pvalues, ax=ax, cmap='RdYlGn_r', mask = (pvalues >= 0.98), linewidths=.5, 
                    xticklabels=keys[:range_], yticklabels=keys[:range_])
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize='x-small')
    g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right', fontsize='x-small')
    st.pyplot(fig)

    st.header('P-value < 0.05')
    # add dataframe of pairs with lowest p-value
    data = {'market1': [k[0] for k in pairs.keys()],
            'market2': [k[1] for k in pairs.keys()],
            'p-value': [v[1] for v in pairs.values()]}

    cm = sns.light_palette("green", as_cmap=True)
    df = pd.DataFrame(data).sort_values('p-value').reset_index(drop=True)
    df_ = df.copy()
    df = df.style.background_gradient(subset=['p-value'], cmap=cm)
    df_['pair'] = df_["market1"].map(str) + ' -- ' + df_["market2"]
    st.table(df)
    # -------------------------------------------------------------------------
    st.sidebar.title("Pairs Cointegration")

    # Choose pairs
    keys = get_market_cap(markets, base)
    pair = st.sidebar.selectbox('Select pair', df_['pair'])
    [market1, market2] = pair.split(' -- ')
    st.header('Pairs Cointegration: {} vs {}'.format(market1, market2))
    # Fit residual
    x1 = data_close[market1].values
    x2 = data_close[market2].values
    r = sm.OLS(x1,x2).fit().params[0]

    df_x = pd.DataFrame({'x1':x1, 'r*x2':r*x2, 'z':x1 - r*x2}, index=data_close.index)

    fig,axs = plt.subplots(2, 1)
    a=sns.lineplot(data=df_x[['x1', 'r*x2']], ax=axs[0])
    b=sns.lineplot(data=df_x[['z']], ax=axs[1], palette=['green'])
    a.figure.canvas.draw(); a.set_xticklabels(a.get_xticklabels(), fontsize='x-small')
    b.figure.canvas.draw(); b.set_xticklabels(b.get_xticklabels(), fontsize='x-small')
    plt.tight_layout()
    st.pyplot(fig)
    
def generate_data(topn=10):
    return

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = {}
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs[(keys[i], keys[j])] = result
    return score_matrix, pvalue_matrix, pairs

