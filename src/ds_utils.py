import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import nakagami
import pickle


def plot_data(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    for i in np.arange(data['K']):
        # ax.scatter(data['R'][i][0], data['R'][i][1], color='C%d' % (i))
        for j in np.arange(data['N'][i]):
            c = plt.Circle(data['R'][i], data['delta'][i, j], fill=False, color='C%d' % i, alpha=0.2)
            ax.add_patch(c)

    ax.plot(data['t'][0][0], data['t'][0][1], 'k*')
    return ax


def generate_artificial_data(k, n, radius, ref_p_noise_std, theta, q_std=10):
    Alpha = [(2 * np.pi) * (i / k) for i in 1 + np.arange(k)]
    noi = np.random.multivariate_normal([0, 0], np.eye(2) * (q_std ** 2), size=1)
    R = np.array([noi + [radius * np.cos(alpha), radius * np.sin(alpha)] for alpha in Alpha])[:, 0, :]

    dNaka = lambda d, theta_i: nakagami(nu=(theta_i + d ** 2) / (2 * theta_i), scale=np.sqrt(theta_i + d ** 2)).rvs(1)[0]

    D = np.zeros([k, n])
    for i in np.arange(k):
        R_sample = np.random.multivariate_normal(R[i, :], np.eye(2) * ref_p_noise_std ** 2, n)
        x = distance.cdist(R_sample, noi)
        D[i] = [dNaka(xi, theta[i]) for xi in x[:, 0]]

    return {'t': noi, 'R': R, 'delta': D, 'K': k, 'N': int(n)*np.ones(k, dtype=np.int), 'N_max': n, 'D': 2, 'sigma_r': ref_p_noise_std}


def save_data(data, radius, run=0):
    file_name = './datasets/dt_k_%d_rd_%1.1f_n_%d_nl_%1.1f_r_%d.pkl' % (data['K'],
                                                                        radius,
                                                                        data['N_max'],
                                                                        data['sigma_r'],
                                                                        run)
    pickle.dump(data, open(file_name, 'wb'))


def load_data(k, radius, n, ref_points_std, run=0):
    file_name = './datasets/dt_k_%d_rd_%1.1f_n_%d_nl_%1.1f_r_%d.pkl' % (k,
                                                                        radius,
                                                                        n,
                                                                        ref_points_std,
                                                                        run)
    return pickle.load(open(file_name, 'rb'))
