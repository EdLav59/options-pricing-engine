# Quantitative Option Pricing Engine

A powerful CLI-based tool for pricing European and American options. It features Black-Scholes, Monte Carlo, and Binomial Tree models, with native Python visualizations using Matplotlib and Seaborn.

## Features

1.  **Black-Scholes Model (European)**
    * Calculates Call and Put prices.
    * Computes full Greeks (Delta, Gamma, Vega, Theta, Rho).

2.  **Monte Carlo Simulation (European)**
    * Vectorized NumPy implementation (optimized for NumPy 2.0+).
    * Handles 50,000+ simulations efficiently.

3.  **Binomial Tree Model (American)**
    * Cox-Ross-Rubinstein (CRR) method.
    * Prices American options (early exercise).

4.  **Visualizations (Matplotlib/Seaborn)**
    * **3D Price Surface:** Interactive window showing Option Price vs Spot vs Time.
    * **Greeks Heatmap:** Gamma sensitivity analysis.
    * **Volatility Smile:** Real-time implied volatility from market data.

