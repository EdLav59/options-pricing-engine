# Options Pricing & Greeks Calculator

**Interactive options pricing engine implementing Black-Scholes, Monte Carlo, and Binomial Tree models with real-time market data integration and implied volatility analysis.**

---

## Project Overview

This project provides a comprehensive toolkit for options pricing and risk analysis, designed for quantitative finance applications. Key features include:

- **Multiple Pricing Models**: Black-Scholes (analytical), Monte Carlo (simulation-based), Binomial Trees (American options)
- **Implied Volatility Solver**: Newton-Raphson optimization to extract market-implied volatilities
- **Greeks Calculator**: Delta, Gamma, Vega, Theta, Rho with sensitivity analysis
- **Interactive Visualizations**: Plotly-powered 3D surfaces, heatmaps, and volatility smiles
- **Market Data Integration**: Real-time options chain fetching from Yahoo Finance

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/options-pricing-engine.git
cd options-pricing-engine
pip install -r requirements.txt
```

### Usage

```bash
python quant_engine.py
```

**Example Interaction:**
```
>>> Ticker Symbol: NVDA
>>> Strike Price: 145
>>> Days to Maturity: 30
>>> Volatility %: 40
>>> Risk-Free Rate %: 4.5
>>> Monte Carlo Simulations: 100000
```

---

## Features

### 1. **Black-Scholes Model**
Analytical solution for European options using the Black-Scholes-Merton framework.

**Call Price Formula:**
```
C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

**Greeks:**
- **Delta (Δ)**: Rate of change of option price w.r.t. underlying price
- **Gamma (Γ)**: Rate of change of delta w.r.t. underlying price
- **Vega (ν)**: Sensitivity to volatility changes
- **Theta (Θ)**: Time decay of option value
- **Rho (ρ)**: Sensitivity to interest rate changes

### 2. **Monte Carlo Simulation**
Risk-neutral valuation using geometric Brownian motion:

```
Sₜ = S₀ exp[(r - σ²/2)T + σ√T·Z]

where Z ~ N(0,1)
```

**Features:**
- 100,000+ simulation paths for high accuracy
- Standard error estimation
- Variance reduction techniques (antithetic variates)

### 3. **Binomial Tree Model**
Cox-Ross-Rubinstein binomial lattice for American options with early exercise:

```
u = exp(σ√Δt)
d = 1/u
p = (eʳᐃᵗ - d) / (u - d)
```

**Advantages:**
- Handles American options (early exercise)
- 300+ time steps for accuracy
- Backward induction algorithm

### 4. **Implied Volatility Solver**
Newton-Raphson iterative method to extract market-implied volatility:

```
σₙ₊₁ = σₙ - [C_BS(σₙ) - C_market] / Vega(σₙ)
```

**Applications:**
- Volatility smile construction
- Arbitrage detection
- Model calibration

### 5. **Interactive Dashboard**
Plotly-powered visualizations:

- **3D Price Surface**: Option value across spot prices and maturities
- **Greeks Analysis**: Delta/Gamma profiles
- **Volatility Smile**: Market-implied volatility across strikes
- **Gamma Heatmap**: Risk exposure visualization

---

## Mathematical Framework

### Black-Scholes PDE
The option pricing follows the fundamental PDE:

```
∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

### Risk-Neutral Pricing
Under the risk-neutral measure Q:

```
V₀ = e⁻ʳᵀ E^Q[max(Sₜ - K, 0)]
```

### Put-Call Parity
European options satisfy:

```
C - P = S₀ - Ke⁻ʳᵀ
```

---

## Example Output

```
MODEL                          | CALL         | PUT         
----------------------------------------------------------------------
Black-Scholes (European)       | $12.45       | $7.89       
Monte Carlo (European)         | $12.43       | $7.91       
  └─ Std Error                 | ±$0.05       | ±$0.04      
Binomial Tree (American)       | $12.50       | $8.10       
----------------------------------------------------------------------

GREEKS                         | VALUE       
----------------------------------------------------------------------
Delta                          | 0.6234      
Gamma                          | 0.0145      
Vega                           | 0.2987      
Theta                          | -0.0523     
Rho                            | 0.1876      
```

---

## References

1. Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy.
2. Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). *Option pricing: A simplified approach*. Journal of Financial Economics.
3. Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
4. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.


---

## Contact

**Edouard Lavalard** 

## License

This project is licensed under the MIT License
