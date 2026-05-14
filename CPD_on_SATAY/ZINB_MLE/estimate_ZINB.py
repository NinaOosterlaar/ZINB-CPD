import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CPD_on_SATAY.ZINB_MLE.log_likelihoods import zinb_log_likelihood
from CPD_on_SATAY.ZINB_MLE.EM import em_zinb_step


def estimate_zinb(data, max_iter=100, tol=1e-6, eps=1e-10, theta_min=0.1, theta_max=1, 
                  n_theta_grid=200):
    """
    Estimate ZINB parameters (pi, mu, theta) using profile likelihood for theta.
    
    The algorithm:
    1. Initialize parameters using method of moments
    2. Create a grid of theta values (log-spaced)
    3. For each theta value:
       a. Run EM to optimize pi and mu (holding theta fixed)
       b. Record the final log-likelihood
    4. Choose theta that maximizes the log-likelihood
    5. Return the best parameters
    
    Parameters:
    -----------
    data : array-like
        Observed count data
    max_iter : int
        Maximum number of EM iterations for each theta value
    tol : float
        Convergence tolerance for EM
    eps : float
        Small value for numerical stability
    theta_min : float
        Minimum value for theta grid (default: 0.05)
    theta_max : float
        Maximum value for theta grid (default: 1e6)
    n_theta_grid : int
        Number of theta values to try in the grid (default: 60)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'pi': final estimate of zero-inflation parameter
        - 'mu': final estimate of mean parameter
        - 'theta': final estimate of dispersion parameter
        - 'iterations': number of EM iterations for best theta
        - 'converged': boolean indicating if convergence was reached
        - 'log_likelihood': final log-likelihood
        - 'theta_grid': array of theta values tested
        - 'll_grid': array of log-likelihoods for each theta
    """
    data = np.asarray(data, dtype=np.float64)
    N = len(data)
    
    # ===== CREATE THETA GRID =====
    # Log-spaced grid from theta_min to theta_max
    theta_grid = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta_grid)
    # Uniform grid from theta_min to theta_max
    # theta_grid = np.linspace(theta_min, theta_max, n_theta_grid)
    
    # Storage for results
    ll_grid = np.zeros(n_theta_grid)
    pi_grid = np.zeros(n_theta_grid)
    mu_grid = np.zeros(n_theta_grid)
    converged_grid = np.zeros(n_theta_grid, dtype=bool)
    iterations_grid = np.zeros(n_theta_grid, dtype=int)
    
    # ===== PROFILE LIKELIHOOD: TRY EACH THETA =====
    for idx, theta in enumerate(theta_grid):
        # Initialize pi and mu for this theta
        pi = np.clip(np.mean(data == 0), eps, 1 - eps)
        ybar = np.mean(data)
        mu = np.clip(ybar / (1 - pi), eps, None)
        
        # Run EM to convergence for this fixed theta
        for iteration in range(max_iter):
            pi_old = pi
            mu_old = mu
            
            # EM step (optimize pi and mu, theta is fixed)
            em_result = em_zinb_step(data, pi, mu, theta, eps=eps)
            pi = em_result['pi']
            mu = em_result['mu']
            
            # Check convergence
            pi_change = abs(pi - pi_old)
            mu_change = abs(mu - mu_old) / (mu_old + eps)
            
            if pi_change < tol and mu_change < tol:
                converged_grid[idx] = True
                iterations_grid[idx] = iteration + 1
                break
        else:
            # Did not converge
            converged_grid[idx] = False
            iterations_grid[idx] = max_iter
        
        # Compute final log-likelihood for this theta
        ll = zinb_log_likelihood(data, mu, theta, pi, eps)
        ll_grid[idx] = ll
        pi_grid[idx] = pi
        mu_grid[idx] = mu
        
        # print(f"theta={theta:.2e}: pi={pi:.4f}, mu={mu:.4f}, ll={ll:.2f}, converged={converged_grid[idx]}")
        
        # if (idx + 1) % 10 == 0 or idx == 0 or idx == n_theta_grid - 1:
        #     print(f"  theta={theta:.2e}: pi={pi:.4f}, mu={mu:.4f}, ll={ll:.2f}, "
        #           f"converged={converged_grid[idx]}")
    
    # ===== SELECT BEST THETA =====
    best_idx = np.argmax(ll_grid)
    best_theta = theta_grid[best_idx]
    best_pi = pi_grid[best_idx]
    best_mu = mu_grid[best_idx]
    best_ll = ll_grid[best_idx]
    best_converged = converged_grid[best_idx]
    best_iterations = iterations_grid[best_idx]
    
    # print(f"\nBest theta: {best_theta:.2e} with ll={best_ll:.2f}")
    # print(f"Final parameters: pi={best_pi:.4f}, mu={best_mu:.4f}, theta={best_theta:.4f}")
    
    return {
        'pi': best_pi,
        'mu': best_mu,
        'theta': best_theta,
        'iterations': best_iterations,
        'converged': best_converged,
        'log_likelihood': best_ll,
        'theta_grid': theta_grid,
        'll_grid': ll_grid
    }


