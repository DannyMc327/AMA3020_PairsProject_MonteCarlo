

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import math

# --------------------------------------------------
# 1. Define the problem and helper functions
# --------------------------------------------------

# In 2D, define the integrand f(x,y) = exp(-(x^2+y^2)) for x,y in [-2,2],
# and f(x,y)=0 outside.
def f(x, y):
    inside = (np.abs(x) <= 2) & (np.abs(y) <= 2)
    return np.where(inside, np.exp(-(x**2 + y**2)), 0)

# True value of the integral:
# In one dimension: I1 = ∫₋₂² exp(–x²) dx = √π * erf(2),
# so in two dimensions: I_true = (I1)^2 = π * (erf(2))^2.
I_true = np.pi * (erf(2))**2

# --------------------------------------------------
# 2. Uniform Monte Carlo Integration in 2D
# --------------------------------------------------
# We sample points uniformly in the square [-2,2] x [-2,2].
# The area of the square is 4*4 = 16.
def mc_uniform_2d(N):
    xs = np.random.uniform(-2, 2, size=N)
    ys = np.random.uniform(-2, 2, size=N)
    # Since our samples lie in [-2,2]^2, we can directly evaluate f(x,y)=exp(-(x^2+y^2))
    vals = np.exp(-(xs**2 + ys**2))
    return 16 * np.mean(vals)

# --------------------------------------------------
# 3. Importance Sampling Monte Carlo Integration in 2D
# --------------------------------------------------
# We use a 2D standard normal distribution as the proposal:
#    g(x,y) = (1/(2π)) exp( –(x²+y²)/2 ).
#
# For samples (x,y) inside [-2,2]^2, the weight is:
#    w(x,y) = f(x,y) / g(x,y)
#           = exp(–(x²+y²)) / [(1/(2π)) exp(–(x²+y²)/2)]
#           = 2π * exp(–(x²+y²)/2).
# Samples outside [-2,2]^2 get a weight of 0.
def mc_importance_2d(N):
    xs = np.random.normal(0, 1, size=N)
    ys = np.random.normal(0, 1, size=N)
    inside = (np.abs(xs) <= 2) & (np.abs(ys) <= 2)
    weights = np.where(inside, 2 * np.pi * np.exp(-(xs**2 + ys**2)/2), 0)
    return np.mean(weights)

# For the convergence plot, we also define a function that returns the running estimate.
def importance_sampling_running_estimate(N):
    xs = np.random.normal(0, 1, size=N)
    ys = np.random.normal(0, 1, size=N)
    inside = (np.abs(xs) <= 2) & (np.abs(ys) <= 2)
    weights = np.where(inside, 2 * np.pi * np.exp(-(xs**2 + ys**2)/2), 0)
    # Compute the cumulative average (running estimate)
    running_estimates = np.cumsum(weights) / np.arange(1, N+1)
    return running_estimates

# --------------------------------------------------
# 4. Compare the Methods over a Range of Sample Sizes
# --------------------------------------------------
Ns = np.logspace(2, 6, num=20, dtype=int)  # sample sizes between 10^2 and 10^6
n_repeats = 50  # number of independent runs per sample size

errors_uniform = []
errors_importance = []

for N in Ns:
    estimates_uni = np.array([mc_uniform_2d(N) for _ in range(n_repeats)])
    estimates_imp = np.array([mc_importance_2d(N) for _ in range(n_repeats)])
    
    error_uni = np.mean(np.abs(estimates_uni - I_true))
    error_imp = np.mean(np.abs(estimates_imp - I_true))
    
    errors_uniform.append(error_uni)
    errors_importance.append(error_imp)

# Plot the average absolute error versus number of samples (log-log scale)
plt.figure(figsize=(8,6))
plt.loglog(Ns, errors_uniform, 'o-', label='Uniform MC')
plt.loglog(Ns, errors_importance, 's-', label='Importance Sampling MC')
plt.xlabel('Number of samples')
plt.ylabel('Average absolute error')
plt.title('2D Monte Carlo Integration Errors')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# --------------------------------------------------
# 5. Plot the Convergence of the Importance Sampling Estimate
# --------------------------------------------------
# Here we plot the running estimate (cumulative average) for one simulation
# using importance sampling to see how it converges toward the true value.
N_run = 100000  # total number of samples for the running estimate
running_est = importance_sampling_running_estimate(N_run)

plt.figure(figsize=(8,6))
plt.loglog(np.arange(1, N_run+1), running_est, label='Running Estimate')
plt.axhline(I_true, color='red', linestyle='--', label='True Value')
plt.xlabel('Number of samples')
plt.ylabel('Estimate of Integral')
plt.title('Convergence of Importance Sampling Estimate')
plt.legend()
plt.grid(True)
plt.show()

