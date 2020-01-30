import pandas as pd
import streamlit as st
import ccxt
import plotly.graph_objects as go
from datetime import datetime

import explorer
import cointegration
import stochastic
import distributions

def main():
    create_layout()
    
    # # Cointegration
    # st.sidebar.subheader('Investigate cointegration:')
    # asset1 = st.sidebar.selectbox('Pair 1:', keys)
    # asset2 = st.sidebar.selectbox('Pair 2:', keys)

def load_homepage() -> None:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT8nmggnnHgxkEbXxLbgMIpi5JC1_HKMA5lig1eOyHS9owHX6Z2",
             use_column_width=True)

def create_layout():

    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Please select a page", ["Homepage",
                                                             "Crypto Explorer",
                                                             "Cointegration",
                                                             "Stochastic Simulation",
                                                             "Distributions"
                                                            ])

    if app_mode == 'Homepage':
        load_homepage()
    
    elif app_mode == "Crypto Explorer":
        explorer.load_page()

    elif app_mode == "Cointegration":
        cointegration.load_page()
    
    elif app_mode == "Stochastic Simulation":
        stochastic.load_page()

    elif app_mode == "Distributions":
        distributions.load_page()

if __name__ == '__main__':
    main()