from sklearn.datasets import make_spd_matrix

k = 100
sigma = make_spd_matrix(n_dim=k, random_state=1)
mvncdf(np.zeros(k),sigma,3)
