import numpy as np
from scipy.special import digamma
from scipy.optimize import least_squares
from scipy.spatial import distance
import stan_utils


def lmlat(R, d):
    r = R[0, :].reshape(1, -1)
    R = R[1:, :]
    K, dim = R.shape
    dr = d[0] * np.ones(K)
    d = d[1:]
    dri = distance.cdist(R, r)

    A = R - r
    b = np.array([0.5 * (np.power(a, 2) + np.power(b, 2) - np.power(c, 2))[0] for a, b, c in zip(dr, dri, d)])

    theta = np.linalg.pinv(A).dot(b)
    return theta + r


def mlat(R, d):
    K, dim = R.shape
    f = lambda q: np.power(distance.cdist(R, q.reshape(1, -1), 'sqeuclidean') - np.power(d.reshape(-1, 1), 2), 2)[:, 0]
    return least_squares(f, np.zeros(dim))['x'].reshape(1, -1)


def pmlat_obj_func(R, d, q_hat, s):
    d_hat = distance.cdist(R, [q_hat])

    value = np.zeros(2)
    for i in np.arange(d.shape[0]):
        factor = (d_hat[i] - d[i]) / (d_hat[i] * s[i])
        value[0] = value[0] + (factor * (q_hat[0] - R[i, 0])) ** 2
        value[1] = value[1] + (factor * (q_hat[1] - R[i, 1])) ** 2
    return value.sum()


def pmlat(R, d, s=None):
    if s is None:
        s = np.ones(d.shape[0])

    f = lambda x: pmlat_obj_func(R, d, x, s)
    return least_squares(f, R.mean(axis=0))['x'].reshape(1, -1)


def mmlat_gdFunction(d, q, r):
    m = (q - r)
    m_norm = np.linalg.norm(m, axis=1)
    S = 0
    for mi, mi_norm, di in zip(m, m_norm, d):
        parc = 0
        parc = parc + 2 * np.log(di) * mi_norm
        parc = parc - di
        parc = parc - 2 * mi_norm * digamma(mi_norm ** 2)
        parc = parc + mi_norm
        parc = parc - 2 * mi_norm * np.log(1 / mi_norm)
        S = S + (mi / mi_norm) * parc
    A = np.linalg.pinv(np.cov(r.T))
    return S - A.dot((q - r.mean(axis=0)).T)


def mmlat(r, d, q0=None, alpha=0.01, max_iter=100):
    if q0 is None:
        q0 = np.zeros(r.shape[1])
    q = q0
    for i in np.arange(max_iter):
        g = mmlat_gdFunction(d, q, r)
        q = q + alpha * g
    return np.array([q])


def bootstrap_sampler(method, data, n_samples=2000):
    if method == 'MLAT':
        f = lambda R, d: mlat(R, d)
    elif method == 'LMLAT':
        f = lambda R, d: lmlat(R, d)
    elif method == 'MMLAT':
        f = lambda R, d: mmlat(R, d)
    elif method == 'PMLAT':
        f = lambda R, d: pmlat(R, d)

    R = data['R']
    D = data['delta']
    k, dim = R.shape
    n = data['N']
    T = np.zeros([n_samples, dim])
    for i in np.arange(n_samples):
        d = np.array([D[ki, np.random.permutation(n[ki])[0]] for ki in np.arange(k)])
        T[i, :] = f(R, d)
    return T


def bmlat_sampler(data, mu, sigma_t, sigma_r, theta, sm=None):
    data['mu'] = mu
    data['sigma_t'] = sigma_t
    data['sigma_r'] = sigma_r
    data['theta'] = theta

    if sm is None:
        sm = stan_utils.compile_model('bmlat')
    fit = sm.sampling(data=data, chains=4, iter=1000, warmup=500, seed=1)
    return fit.extract()['t']
