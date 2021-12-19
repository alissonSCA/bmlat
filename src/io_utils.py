import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.spatial import distance
import ds_utils

class Save_samples:
    samples = {'t0': [], 't1': [], 'method': []}
    filename = './results/samples'

    def __init__(self, filename):
        self.filename = filename
        self.samples = {'t0': [], 't1': [], 'method': []}

    def add(self, Q, method):
        self.samples['t0'].extend(Q[:, 0])
        self.samples['t1'].extend(Q[:, 1])
        self.samples['method'].extend([method for x in np.arange(Q.shape[0])])

    def dump(self):
        df = pd.DataFrame(self.samples)
        df.to_csv(self.filename)
        self.samples = {'t0': [], 't1': [], 'method': []}


def sumarize_artificial_data(K, N, Sigma_r = None, n_runs = 30, verbose=False):

	if Sigma_r is None:
		Sigma_r = np.linspace(4, 10, 5)
	
	summary = {'method': [], 'sigma_r': [], 'r': [], 'distance_mean': [], 'mean_distance': [], 'likelihood': []}
	for sigma_r in Sigma_r:
		for r in np.arange(n_runs):
			df = pd.read_csv('./results/samples/artificial/K_%d_N_%d_sigma_%1.1f_r_%d.csv' % (K, N, sigma_r, r))
			data = ds_utils.load_data(K, 40, N, sigma_r, r)
			t_star = data['t']	
			for method in ['BMLAT', 'LMLAT', 'MLAT', 'PMLAT', 'MMLAT']:#np.unique(df['method']):
				df_m = df[df['method']==method].to_dict('list')
				T = np.array([df_m['t0'], df_m['t1']]).T
				
				t_mean = np.mean(T, axis=0)
				C = np.cov(T.T)

				distance_mean = distance.cdist(t_star, T).mean()   
				mean_distance = distance.cdist(t_star, t_mean.reshape(1, -1))[0, 0]				 
				likelihood    = multivariate_normal.pdf(t_star, t_mean, C)
				
				summary['method'].append(method)
				summary['sigma_r'].append(sigma_r)
				summary['r'].append(r)
				summary['distance_mean'].append(distance_mean)
				summary['mean_distance'].append(mean_distance)
				summary['likelihood'].append(likelihood)
					
				if verbose:
					print('method: %s' % method)
					print('\tdistance to mean: %1.4f\n\tmean of distance: %1.4f\n\tlikelihood: %1.4f' % (mean_distance, distance_mean, likelihood))

	return summary


def sumarize_real_data(id, x, y, verbose=False):

	summary = {'method': [], 'ds': [], 'distance_mean': [], 'mean_distance': [], 'likelihood': []}

	df = pd.read_csv('./results/samples/real/ds_%d_t_%1.1f_%1.1f.csv' % (id, x, y))
	data = ds_utils.generate_rssi_ds(x, y)
	t_star = data['t']
	for method in ['BMLAT', 'LMLAT', 'MLAT', 'PMLAT', 'MMLAT']:  # np.unique(df['method']):

		df_m = df[df['method'] == method].to_dict('list')
		T = np.array([df_m['t0'], df_m['t1']]).T

		t_mean = np.mean(T, axis=0)
		C = np.cov(T.T)

		distance_mean = distance.cdist(t_star, T).mean()
		mean_distance = distance.cdist(t_star, t_mean.reshape(1, -1))[0, 0]
		likelihood = multivariate_normal.pdf(t_star, t_mean, C)

		summary['method'].append(method)
		summary['ds'].append(id)
		summary['distance_mean'].append(distance_mean)
		summary['mean_distance'].append(mean_distance)
		summary['likelihood'].append(likelihood)
		summary[method] = [distance_mean, likelihood, mean_distance, id]

		if verbose:
			print('method: %s' % method)
			print('\tdistance to mean: %1.4f\n\tmean of distance: %1.4f\n\tlikelihood: %1.4f' % (
			mean_distance, distance_mean, likelihood))

	return summary

