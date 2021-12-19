functions{  
  real nakagammi_log(real x, real m, real omega){
    return log(2) +m*log(m) + (2*m - 1)*log(x) - log(tgamma(m)) - m*log(omega) - (m*pow(x, 2))/omega;
  }
  real dnakagammi_log(real x, real d, real theta){
    real m;
    real omega;
    omega = pow(d, 2) + theta;
    m = omega/(2*(omega-pow(d,2)));    
    return nakagammi_log(x, m, omega);
  }  
}
data {
    int<lower=1> K;           // number of reference points
    int<lower=1> D;           // data in R^D
    int<lower=1> N_max;       // max number of observed distance for each reference points
    int N[K];                 // number of observed distance for each reference points
    matrix[K, N_max] delta;   // observed distances
    matrix[K, D] R;           // reference points mean
    vector[D] mu;             // initial guess
    real sigma_t;             // scale for t prior
    vector[K] theta;          // scale for dNaka likilihood
    real sigma_r;             // scale for reference points noise
}
parameters {
    vector[D] t;
    matrix[K,D] r;
}
model {
  t ~ multi_normal(mu, diag_matrix(rep_vector(sigma_t*sigma_t,D)));
  for (k in 1:K){
    //priors
    r[k] ~ multi_normal(R[k], diag_matrix(rep_vector(sigma_r*sigma_r,D)));
    for (n in 1:N[k]){
        //likilihood
        delta[k, n] ~ dnakagammi(distance(r[k], t), theta[k]);
    }
  }
}