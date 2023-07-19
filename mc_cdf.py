from scipy.stats import mvn
import numpy as np

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
