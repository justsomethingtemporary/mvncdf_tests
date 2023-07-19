from scipy.stats import mvn
import numpy as np
low = np.array([0, 0,0,0,0,0])
upp = np.array([np.inf, np.inf,np.inf,np.inf,np.inf,np.inf])
mu = np.array([0, 0,0,0,0,0])
S = np.array([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
p,i = mvn.mvnun(low,upp,mu,S)
print (p)
print(1/p)
