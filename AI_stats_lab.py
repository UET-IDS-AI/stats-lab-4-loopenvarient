"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    STEP 1
    Compute analytically

        P(X > 5)
        P(X < 5)
        P(3 < X < 7)

    STEP 2
    Simulate 100000 samples from Exp(1)

    STEP 3
    Estimate P(X > 5) using simulation

    RETURN

        analytic_gt5
        analytic_lt5
        analytic_interval
        simulated_gt5
    """
    analytic_gt5 = math.exp(-5)                 
    analytic_lt5 = 1 - math.exp(-5)            
    analytic_interval = math.exp(-3) - math.exp(-7) 
    range = np.random.default_rng(42)
    samples = range.exponential(scale=1.0, size=100000)  
    simulated_gt5 = float(np.mean(samples > 5))

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5



# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Candidate PDF

        f(x) = 2x e^{-x^2} for x >= 0

    STEP 1
    Verify non-negativity

    STEP 2
    Compute

        integral_0^∞ f(x) dx

    STEP 3
    Determine if valid PDF

    STEP 4
    Plot f(x) on [0,3]

    RETURN

        integral_value
        is_valid_pdf
    """

    def f(x):
        # 2x e^{-x^2} for x>=0, 0 otherwise
        return 2 * x * math.exp(-x * x) if x >= 0 else 0.0
    integral_value, _ = quad(lambda t: 2 * t * math.exp(-t * t), 0, np.inf)

    is_valid_pdf = abs(integral_value - 1.0) < 1e-3

    xs = np.linspace(0, 3, 400)
    ys = 2 * xs * np.exp(-xs**2)
    plt.figure()
    plt.plot(xs, ys)
    plt.title(r"Candidate PDF: $f(x)=2x e^{-x^2}u(x)$")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return float(integral_value), bool(is_valid_pdf)


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    X ~ Exp(1)

    STEP 1
    Compute analytically

        P(X > 5)
        P(1 < X < 3)

    STEP 2
    Simulate 100000 samples

    STEP 3
    Estimate probabilities using simulation

    RETURN

        analytic_gt5
        analytic_interval
        simulated_gt5
        simulated_interval
    """

    analytic_gt5 = math.exp(-5)                         
    analytic_interval = math.exp(-1) - math.exp(-3)      
    range = np.random.default_rng(42)
    samples = range.exponential(scale=1.0, size=100000)
    simulated_gt5 = float(np.mean(samples > 5))
    simulated_interval = float(np.mean((samples > 1) & (samples < 3)))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(10,2^2)

    STEP 1
    Standardize variable

        Z = (X - 10)/2

    STEP 2
    Compute analytically

        P(X ≤ 12)
        P(8 < X < 12)

    STEP 3
    Simulate 100000 samples

    STEP 4
    Estimate probabilities

    RETURN

        analytic_le12
        analytic_interval
        simulated_le12
        simulated_interval
    """

    mu = 10.0
    sigma = 2.0
    z12 = (12 - mu) / sigma
    analytic_le12 = float(norm.cdf(z12))
    z8 = (8 - mu) / sigma
    analytic_interval = float(norm.cdf(z12) - norm.cdf(z8))
    range = np.random.default_rng(42)
    samples = range.normal(loc=mu, scale=sigma, size=100000)

    simulated_le12 = float(np.mean(samples <= 12))
    simulated_interval = float(np.mean((samples > 8) & (samples < 12)))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval
