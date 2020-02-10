import streamlit as st
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.data import load_data, get_keys, get_names, get_market_cap, get_history
from cointegration import find_cointegrated_pairs

def load_page():

    markets, exception = load_data()

    if exception:
        st.sidebar.text(str(exception))
        st.title("⭕️The data was not correctly loaded")
        return

    title = st.empty()
    st.sidebar.title("Fit SDE to Time Series")
    resolution = st.sidebar.selectbox('Select resolution', ['1d', '1h', '1m'], index=2)

    keys = get_market_cap(markets, 'USDT')

    data_close = get_history(keys[:10], timeframe=resolution, limit=2000)
    data_close = data_close.dropna()
    scores, pvalues, pairs = find_cointegrated_pairs(data_close)

    data = {'market1': [k[0] for k in pairs.keys()],
            'market2': [k[1] for k in pairs.keys()],
            'p-value': [v[1] for v in pairs.values()]}
    
    df = pd.DataFrame(data).sort_values('p-value').reset_index(drop=True)
    df['pair'] = df["market1"].map(str) + ' -- ' + df["market2"]
    pair = st.sidebar.selectbox('Select pair', df['pair'])
    [market1, market2] = pair.split(' -- ')
    st.header('Pairs Cointegration: {} vs {}'.format(market1, market2))
    
    # Fit residual
    x1 = data_close[market1].values
    x2 = data_close[market2].values
    r = sm.OLS(x1,x2).fit().params[0]

    df_x = pd.DataFrame({'x1':x1, 'r*x2':r*x2, 'z':x1 - r*x2}, index=data_close.index)

    # Plot
    st.latex(r'\text{Regressed residual: } z = x_1 - (r\times x_2)')
    fig1,axs = plt.subplots(3, 2, figsize=(16,8))
    y_real = df_x[['z']]
    _=sns.lineplot(data=y_real, ax=axs[0,0], color='green').set_title("time series")
    _=sns.distplot(y_real, ax=axs[0,1], bins=100, color='green').set_title("distribution")
    _=sns.lineplot(data=pd.Series(y_real['z']).diff(), ax=axs[1,0], color='purple').set_title("time series")
    _=sns.distplot(pd.Series(y_real['z']).diff(), ax=axs[1,1], bins=100, color='purple').set_title("distribution")
    _=sns.lineplot(data=pd.Series(y_real['z']).diff().diff().diff(), ax=axs[2,0], color='orange').set_title("time series")
    _=sns.distplot(pd.Series(y_real['z']).diff().diff().diff(), ax=axs[2,1], bins=100, color='orange').set_title("distribution")
    plt.tight_layout()
    st.pyplot(fig1) 
    st.latex(r'\text{{Higuchi Fractal Dimension: }} {0:0.3f} \space\Bigm\lvert '.format(hfd(y_real['z'].values)) \
    + r'\text{{ Hurst Exponent: }} {0:0.3f}'.format(hurst(y_real['z'].values)))

    st.markdown(r'---')

    # ---------------- Fit SDE to data with CMA-ES -------------- # 
    import cma
    from stochastic import ornstein_uhlenbeck_levels, cox_ingersoll_ross_levels, plot_stochastic_processes, ModelParameters

    process = st.sidebar.selectbox("Select process", ["Ornstein Uhlenbeck", "Cox Ingersall Ross"])

    mp = ModelParameters(all_s0=1000,
                     all_r0=0.5,
                     all_time=2000,
                     all_delta=0.00396825396,
                     all_sigma=0.125,
                     gbm_mu=0.058,
                     jumps_lamda=0.00125,
                     jumps_sigma=0.001,
                     jumps_mu=-0.2,
                     cir_a=3.0,
                     cir_mu=0.5,
                     cir_rho=0.5,
                     ou_a=3.0,
                     ou_mu=0.5,
                     heston_a=0.25,
                     heston_mu=0.35,
                     heston_vol0=0.06125)

    mp_dict = {
        'all_s0':       {'val':1000, 'min':500, 'max':5000},
        'all_r0':       {'val':0.5, 'min':0.0, 'max':1.0},
        'all_time':     {'val':800, 'min':100, 'max':2000},
        'all_delta':    {'val':0.004, 'min':0.0, 'max':1.0},
        'all_sigma':    {'val':0.125, 'min':0.0, 'max':1.0},
        'gbm_mu':       {'val':0.058, 'min':0.0, 'max':1.0},
        'jumps_lambda': {'val':0.00125, 'min':0.0, 'max':1.0},
        'jumps_mu':     {'val':-0.2, 'min':-1.0, 'max':1.0},
        'cir_a':        {'val':3.0, 'min':0.0, 'max':10.0},
        'cir_mu':       {'val':0.5, 'min':0.0, 'max':1.0},
        'cir_rho':      {'val':0.5, 'min':0.0, 'max':1.0},
        'ou_a':         {'val':3.0, 'min':0.0, 'max':10.0},
        'ou_mu':        {'val':0.5, 'min':0.0, 'max':1.0},
        'heston_a':     {'val':0.25, 'min':0.0, 'max':1.0},
        'heston_mu':    {'val':0.35, 'min':0.0, 'max':1.0},
        'heston_vol0':  {'val':0.06125, 'min':0.0, 'max':1.0}
    }

    # train parameters with cma-es
    process_map = {"Ornstein Uhlenbeck":
                        {'process':ornstein_uhlenbeck_levels,
                         'params':['all_time', 'all_r0', 'all_delta', 'ou_a', 'ou_mu']},
                   "Cox Ingersall Ross":
                        {'process':cox_ingersoll_ross_levels,
                         'params': ['all_time', 'all_r0', 'all_delta', 'cir_a', 'cir_mu']}
    }
    
    fit_cmaes = st.sidebar.button('Fit CMAES')

    #---------------------------------------------------------
    
    #---------------------------------------------------------

    # Rerun model and sample from distribution with new parameters
    first_run = True
    run = st.button('Resample')
    if first_run or run:
        # Plot trajectory
        examples = []
        examples.append(process_map[process]['process'](mp))
        first_run = False
    
    st.latex(r'\text{Ornstein Uhlenbeck Process: } d r_t = a(b - r_t)dt + \sigma r_t d W_t')
    
    fig2,axs = plt.subplots(3, 2, figsize=(16,8))
    for i in range(len(examples)):
        y_sim = match(examples[i], df_x['z'])
        _=sns.lineplot(x=np.arange(0,len(y_sim),1), y=y_sim, ax=axs[0,0], color='green').set_title("time series")
        _=sns.distplot(y_sim, ax=axs[0,1], bins=100, color='green').set_title("distribution")
        _=sns.lineplot(x=np.arange(0,len(y_sim),1), y=pd.Series(y_sim).diff(), ax=axs[1,0], color='purple').set_title("time series")
        _=sns.distplot(pd.Series(y_sim).diff(), ax=axs[1,1], bins=100, color='purple').set_title("distribution")
        _=sns.lineplot(x=np.arange(0,len(y_sim),1), y=pd.Series(y_sim).diff().diff(), ax=axs[2,0], color='orange').set_title("time series")
        _=sns.distplot(pd.Series(y_sim).diff().diff(), ax=axs[2,1], bins=100, color='orange').set_title("distribution")
    plt.tight_layout()
    st.pyplot(fig2) 
    st.latex(r'\text{{Higuchi Fractal Dimension: }} {0:0.3f} \space\Bigm\lvert '.format(hfd(y_sim)) \
    + r'\text{{ Hurst Exponent: }} {0:0.3f}'.format(hurst(y_sim)))

    st.markdown(r'---')
    from scipy.stats import norm, entropy
    import numpy.random as r

    header_kld = st.empty()
    fig3=plt.figure(figsize=(16,8))
    sns.kdeplot(df_x['z'], label='real')
    kde = sns.kdeplot(match(examples[0], df_x['z']), label='simulated')
    (pdf1_x, pdf1_y) = kde.get_lines()[0].get_data()
    (pdf2_x, pdf2_y) = kde.get_lines()[1].get_data()
    kld = entropy(pdf1_y, pdf2_y)
    header_kld.latex(r'\text{{KL Divergence: }} {0:.5f}'.format(kld))
    plt.tight_layout()
    st.pyplot(fig3)
    

def match(target, source):
        """Match target time series to have same mean and standard deviation and source time series.
        """
        target = np.array(target)
        source = np.array(source)
        return ((target - target.mean()) / target.std()) * source.std() + source.mean()

def katz(data):
    """Katz fractal dimension.
    """
    n = len(data)-1
    L = np.hypot(np.diff(data), 1).sum() # Sum of distances
    d = np.hypot(data - data[0], np.arange(len(data))).max() # furthest distance from first point
    return np.log10(n) / (np.log10(d/L) + np.log10(n))

def hfd(a, k_max=10):
    """Higuchi fractal dimension
    """
    L = []
    x = []
    N = len(a)

    for k in range(1,k_max):
        Lk = 0
        for m in range(0,k):
            #we pregenerate all idxs
            idxs = np.arange(1,int(np.floor((N-m)/k)),dtype=np.int32)
            Lmk = np.sum(np.abs(a[m+idxs*k] - a[m+k*(idxs-1)]))
            Lmk = (Lmk*(N - 1)/(((N - m)/ k)* k)) / k
            Lk += Lmk

        L.append(np.log(Lk/(m+1)))
        x.append([np.log(1.0/ k), 1])

    (p, r1, r2, s)=np.linalg.lstsq(x, L)
    return p[0]

def hurst(ts):
    """Returns the Hurst Exponent of the time series vector t
    """
    from numpy import sqrt, subtract, log, polyfit, std
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses 
    # standard deviation and then make a root of it?
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0
