from scipy.stats import mvn
import numpy as np
import time
from sklearn.datasets import make_spd_matrix
import streamlit as st
from botorch.sampling import qmc
import torch

with st.echo(code_location='below'):
    def mvncdf(mu, sigma, max, maxpts):
        maxar = np.full(
            shape=np.shape(mu)[0],
            fill_value=max
        )
        newloc = 2*mu - maxar
        upp = np.full(
            shape=np.shape(mu)[0],
            fill_value=np.inf,
            )
        p, i = mvn.mvnun(newloc, upp, mu, sigma, maxpts=maxpts)
        return p

    def qmc_box_muller(mu, sigma, max, maxpts):
        mean = torch.tensor(mu)
        cov = torch.tensor(sigma)
        engine = qmc.MultivariateNormalQMCEngine(mean,cov,seed=1)
        g = torch.empty(mu.shape[0]).fill_(max) # (max,) * mu.shape[0]
        i = 0
        o = 0
        for p, k in enumerate(engine.draw(maxpts)):
            check = k < g
            if check.all(0,False) == torch.tensor(True):
                i += 1
            else:
                o += 1
        return i / (i + o)

    dimension = st.slider("Dimension of multivariate gaussian distribution", 1, 1000, 5)
    max_val = st.slider("Maximum value achieved", -10.0, 10.0, 0.0)
    maxpts = st.slider("Max number of points (increase to improve accuracy)", 1000, 1000000, 1000, step=1000)

    option = st.selectbox(
        "What type of covariance matrix should be used?",
        ("Random", "Identity")
    )
    if option == "Random":
        sigma = make_spd_matrix(n_dim=dimension, random_state=1)
    elif option == "Identity":
        sigma = np.identity(dimension)

    st.write("Utilizing Genz's QMC sampling in scipy")
    start_time = time.time()
    p = mvncdf(np.zeros(dimension), sigma, max_val, maxpts)
    st.write("Probability of a lower value is", str(p))
    s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
    st.write(s)

    st.write("Utilizing Box-Mueller QMC sampling in botorch")
    start_time = time.time()
    p = qmc_box_muller(np.zeros(dimension), sigma, max_val, maxpts)
    st.write("Probability of a lower value is", str(p))
    s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
    st.write(s)
    
