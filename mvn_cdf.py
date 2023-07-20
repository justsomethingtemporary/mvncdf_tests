from scipy.stats import mvn
import numpy as np
import time
from sklearn.datasets import make_spd_matrix
import streamlit as st

with st.echo(code_location='below'):
    def mvncdf(mu, sigma, max):
        maxar = np.full(
            shape=np.shape(mu)[0],
            fill_value=max
        )
        newloc = 2*mu - maxar
        upp = np.full(
            shape=np.shape(mu)[0],
            fill_value=np.inf,
            )
        p, i = mvn.mvnun(newloc, upp, mu, sigma)
        return p

    dimension = st.slider("Dimension of multivariate gaussian distribution", 1, 500, 5)
    sigma = make_spd_matrix(n_dim=dimension, random_state=1)
    max_val = st.slider("Maximum value achieved", -10, 10, 0)
    
    start_time = time.time()
    p = mvncdf(np.zeros(dimension), sigma, max_val)
    st.write("Probability of a lower value is", str(p))
    s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
    st.write(s)
    
