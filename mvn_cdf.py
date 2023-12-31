from scipy.stats import mvn
from scipy.stats import norm
from scipy.linalg import eigh
import numpy as np
import time
import torch
from sklearn.datasets import make_spd_matrix
import streamlit as stfrom scipy.stats import mvn
from scipy.stats import norm
from scipy.linalg import eigh
import numpy as np
import time
import torch
from sklearn.datasets import make_spd_matrix
import streamlit as st
from botorch.sampling import qmc

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

with st.echo(code_location='below'):
    def wrapper(funct, mu, sigma, maxval):
            start_time = time.time()
            maxpts = 1000*mu.shape[0]
            cdf = funct(mu, sigma, maxval, maxpts)
            end_time = time.time()
            return cdf, end_time - start_time

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
        samples = engine.draw(1000 * mu.shape[0])
        dimensions = samples.size()    
        row_check = torch.sum(samples < max, dim = 1)
        in_pts = torch.sum(row_check == dimensions[1], dim = 0)
        return in_pts.item()/dimensions[0]

    def split(mu, sigma, max, maxpts):
        prob_cdf = 1
        t = np.diagonal(sigma)
        for i in range(mu.shape[0]):
            prob_cdf = prob_cdf * norm.cdf(max, 0, np.sqrt(t[i]))
        return prob_cdf
        return np.prod(norm.cdf(alt, 0, eigenvalues))

    def variance_only(mu, sigma, max, maxpts):
        return qmc_box_muller(mu, np.diag(np.diagonal(sigma)), max, maxpts)
        
    def eigenvalues_only(mu, sigma, max, maxpts):
        return qmc_box_muller(mu, np.diag(eigh(sigma, eigvals_only=True)), max, maxpts)
    
    def variance_only_decomposition(mu, sigma, max, maxpts):
        return np.prod(norm.cdf(max, mu, [np.sqrt(i) for i in np.diagonal(sigma)]))
    
    def eigenvalues_only_decomposition(mu, sigma, max, maxpts):
        return np.prod(norm.cdf(max, mu, [np.sqrt(i) for i in eigh(sigma, eigvals_only=True)]))

    opt = st.selectbox(
        "Should we examine for a single input or multiple inputs",
        ("Single", "Multiple")
    )
    if opt == "Single":
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

        st.write("Utilizing Box-Mueller transformed Sobol QMC sampling in botorch")
        start_time = time.time()
        p = qmc_box_muller(np.zeros(dimension), sigma, max_val, maxpts)
        st.write("Probability of a lower value is", str(p))
        s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
        st.write(s)
        
        st.write("Sepparating multivariate normal distribution into single dimensional problems evaluated with scipy norm.cdf")
        start_time = time.time()
        p = split(np.zeros(dimension), sigma, max_val, maxpts)
        st.write("Probability of a lower value is", str(np.log10(p)))
        s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
        st.write(s)
        
        st.write("Assumption of independence between experiments using qmc")
        start_time = time.time()
        p = variance_only(np.zeros(dimension), sigma, max_val, maxpts)
        st.write("Probability of a lower value is", str(p))
        s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
        st.write(s)

        st.write("Assumption of independence taking eigenvalues using qmc")
        start_time = time.time()
        p = eigenvalues_only(np.zeros(dimension), sigma, max_val, maxpts)
        st.write("Probability of a lower value is", str(p))
        s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
        st.write(s)
        
        st.write("Assumption of independence using decomposition")
        start_time = time.time()
        p = variance_only_decomposition(np.zeros(dimension), sigma, max_val, maxpts)
        st.write("Probability of a lower value is", str(np.log10(p)))
        s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
        st.write(s)
        
        st.write("Assumption of independence taking eigenvalues using decomposition")
        start_time = time.time()
        p = eigenvalues_only_decomposition(np.zeros(dimension), sigma, max_val, maxpts)
        st.write("Probability of a lower value is", str(p))
        s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
        st.write(s)
        
    elif opt == "Multiple":
        input_dims = st.selectbox(
            "Choose which dimensions to test:",
            ("10, 20, 50, 100, 200, 300, 400, 500, 600", "0")
        )
        option = st.selectbox(
            "What type of covariance matrix should be used?",
            ("Identity", "Random")
        )
        max_val = st.slider("Maximum value achieved", -10.0, 10.0, 2.0)

        if input_dims == "10, 20, 50, 100, 200, 300, 400, 500, 600":
            dims = [10, 20, 50, 100,  200, 300, 400, 500, 600]
        
        # mvncdf, qmc_box_muller, variance_only_decomposition, eigenvalues_only_decomposition
        functions = [mvncdf, qmc_box_muller, variance_only_decomposition, eigenvalues_only_decomposition]
        plotables = []
        for f in functions:
            plotables.append([[],[]])
        for dimension in dims:
            if option == "Random":
                sigma = make_spd_matrix(n_dim=dimension, random_state=1)
            elif option == "Identity":
                sigma = np.identity(dimension)
            trio = (np.zeros(dimension), sigma, max_val)
            for num in range(0,len(functions)):
                cdf, t = wrapper(functions[num], *trio)
                plotables[num][0].append(cdf)
                plotables[num][1].append(t)
        
        marking_types = ['ro', 'bo', 'co', 'ko', 'mo', 'yo', 'co', 'wo']
        functnames = ["mvncdf", "qmc_box_muller", "variance_only_decomposition", "eigenvalues_only_decomposition"]
        for i in range(0, len(functions)):
            st.write(functnames[i]+":", marking_types[i])
        plotting = []
        for i in range(0, len(functions)):
            plotting.append(dims)
            plotting.append(plotables[i][0])
            plotting.append(marking_types[i])
        plt.plot(*plotting, alpha = 0.3)
        st.pyplot(plt)

        plotting = []
        for i in range(0, len(functions)):
            plotting.append(dims)
            plotting.append(plotables[i][1])
            plotting.append(marking_types[i])
        plt.plot(*plotting, alpha = 0.3)
        st.pyplot(plt)
from botorch.sampling import qmc

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
        samples = engine.draw(1000 * mu.shape[0])
        dimensions = samples.size()    
        row_check = torch.sum(samples < max, dim = 1)
        in_pts = torch.sum(row_check == dimensions[1], dim = 0)
        return in_pts.item()/dimensions[0]
        """
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
        """
    def split(mu, sigma, max):
        eigenvalues, eigenvectors = eigh(sigma)
        eigvec_t = np.transpose(eigenvectors)
        altstart = np.matmul(eigvec_t, np.repeat(max, mu.shape[0]) - mu)
        bound = np.matmul(eigvec_t, np.repeat(-50000, mu.shape[0]))
        alt = []
        for i in range(0, mu.shape[0]):
            if altstart[i] > bound[i]:
                alt.append(altstart[i])
            else:
                alt.append(abs(altstart[i]))
        return np.prod(norm.cdf(alt, 0, eigenvalues))

    def variance_only(mu, sigma, max, maxpts):
        return qmc_box_muller(mu, np.diag(np.diagonal(sigma)), max, maxpts)
        
    def eigenvalues_only(mu, sigma, max, maxpts):
        return qmc_box_muller(mu, np.diag(eigh(sigma, eigvals_only=True)), max, maxpts)
    
    def variance_only_decomposition(mu, sigma, max):
        return np.prod(norm.cdf(max, mu, [np.sqrt(i) for i in np.diagonal(sigma)]))
    
    def eigenvalues_only_decomposition(mu, sigma, max):
        return np.prod(norm.cdf(max, mu, [np.sqrt(i) for i in eigh(sigma, eigvals_only=True)]))

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

    st.write("Utilizing Box-Mueller transformed Sobol QMC sampling in botorch")
    start_time = time.time()
    p = qmc_box_muller(np.zeros(dimension), sigma, max_val, maxpts)
    st.write("Probability of a lower value is", str(p))
    s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
    st.write(s)

    st.write("Eigenvalue decomposition into univariate normal distributions")
    start_time = time.time()
    p = split(np.zeros(dimension), sigma, max_val)
    st.write("Probability of a lower value is", str(p))
    s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
    st.write(s)
    
    st.write("Assumption of independence between experiments using qmc")
    start_time = time.time()
    p = variance_only(np.zeros(dimension), sigma, max_val, maxpts)
    st.write("Probability of a lower value is", str(p))
    s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
    st.write(s)
    
    st.write("Assumption of independence taking eigenvalues using qmc")
    start_time = time.time()
    p = eigenvalues_only(np.zeros(dimension), sigma, max_val, maxpts)
    st.write("Probability of a lower value is", str(p))
    s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
    st.write(s)
    
    st.write("Assumption of independence using decomposition")
    start_time = time.time()
    p = variance_only_decomposition(np.zeros(dimension), sigma, max_val)
    st.write("Probability of a lower value is", str(p))
    s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
    st.write(s)
    
    st.write("Assumption of independence taking eigenvalues using decomposition")
    start_time = time.time()
    p = eigenvalues_only_decomposition(np.zeros(dimension), sigma, max_val)
    st.write("Probability of a lower value is", str(p))
    s = "Time to calculate: " + str(time.time() - start_time) + " seconds"
    st.write(s)
