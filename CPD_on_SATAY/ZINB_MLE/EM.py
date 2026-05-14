import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def nb_pmf_zero(mu, theta, eps=1e-10):
    """
    Compute the NB probability mass at zero: NB(0|mu, theta).
    
    Parameters:
    -----------
    mu : float
        Mean parameter
    theta : float
        Dispersion parameter
    eps : float
        Small value for numerical stability
    
    Returns:
    --------
    float : Probability NB(0|mu, theta)
    """
    mu = np.clip(mu, eps, None)
    theta = np.clip(theta, eps, None)
    log_p0 = theta * (np.log(theta) - np.log(theta + mu))
    return float(np.exp(log_p0))


def em_zinb_step(data, pi, mu, theta, eps=1e-10, pi_min = 0.15, pi_max = 0.9):
    """
    Perform ONE iteration of the EM algorithm for ZINB (E-step + M-step).
    
    Parameters:
    -----------
    data : array-like
        Observed count data
    pi : float
        Current zero-inflation parameter (0 < pi < 1)
    mu : float
        Current mean parameter (mu > 0)
    theta : float
        Current dispersion parameter (theta > 0)
    eps : float
        Small value for numerical stability
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'pi': updated zero-inflation parameter
        - 'mu': updated mean parameter
        - 'weights': array of weights a_i = P(z_i=0|y_i) for Newton-Raphson
    """
    data = np.asarray(data, dtype=np.float64)
    N = len(data)
    
    # Clip parameters for numerical stability
    pi = float(np.clip(pi, eps, 1.0 - eps))
    mu = float(np.clip(mu, eps, None))
    theta = float(np.clip(theta, eps, None))
    
    # Identify zero observations
    is_zero = (data == 0)
    n_zeros = np.sum(is_zero)
    
    # ===== E-STEP =====
    # Compute P(z_i=1|y_i) - probability of being from zero-inflation
    z_prob = np.zeros(N)
    
    if n_zeros > 0:
        nb_zero = nb_pmf_zero(mu, theta, eps)
        denominator = pi + (1.0 - pi) * nb_zero
        denominator = np.clip(denominator, eps, None)
        z_prob[is_zero] = pi / denominator
    
    # Non-zeros automatically have z_prob = 0
    
    # ===== M-STEP =====
    # Update pi: pi = (1/N) * sum P(z_i=1|y_i)
    pi_new = np.mean(z_prob)
    pi_new = np.clip(pi_new, pi_min, pi_max)  # Ensure pi stays within reasonable bounds
    
    # Compute weights for mu and theta estimation: a_i = P(z_i=0|y_i) = 1 - P(z_i=1|y_i)
    weights = 1.0 - z_prob
    
    # Update mu: mu = sum[P(z_i=0|y_i) * y_i] / sum[P(z_i=0|y_i)]
    denominator = np.sum(weights)
    if denominator < eps:
        mu_new = eps  # essentially no NB mass
    else:
        mu_new = np.sum(weights * data) / denominator
        mu_new = np.clip(mu_new, eps, None)
    
    return {
        'pi': pi_new,
        'mu': mu_new,
        'weights': weights
    }
