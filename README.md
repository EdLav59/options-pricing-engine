# Options Pricing & Greeks Calculator

A comprehensive options pricing engine implementing multiple valuation models with advanced sensitivity analysis and risk management features.

## Overview

This project implements three fundamental option pricing methodologies alongside sensitivity analysis and delta-hedging simulation capabilities. The engine supports both European and American options, with real-time market data integration for implied volatility analysis.

## Features

**Core Pricing Models:**
- Black-Scholes analytical solution for European options
- Monte Carlo simulation with 100,000+ paths and variance estimates
- Cox-Ross-Rubinstein binomial tree for American options

**Advanced Analytics:**
- Newton-Raphson implied volatility solver
- Comprehensive Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Multi-dimensional sensitivity analysis (volatility, maturity, interest rates)
- Delta-hedging simulator with dynamic rebalancing
- Market price comparison and arbitrage detection

**Visualization:**
- Interactive Plotly dashboards
- 3D option price surfaces
- Greeks sensitivity profiles
- Volatility smile construction
- Hedging performance metrics

## Installation

```bash
git clone https://github.com/EdLav59/options-pricing-engine.git
cd options-pricing-engine
pip install -r requirements.txt
```

## Usage

### Basic Pricing

```bash
python quant_engine.py
```

Provides pricing from three models, Greeks calculation, and interactive visualization.

### Comprehensive Analysis

```bash
python quant_engine_full.py
```

Includes all basic features plus sensitivity analysis, hedging simulation, and risk management recommendations.

### Example

```
>>> Ticker Symbol: NVDA
>>> Strike Price: 188
>>> Days to Maturity: 30
>>> Volatility: 42
>>> Risk-Free Rate: 4.5
>>> Monte Carlo Simulations: 100000
```

## Mathematical Framework

### Black-Scholes Model

The Black-Scholes formula provides closed-form solutions for European options:

```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Greeks

First and second-order partial derivatives measure option sensitivities:

- **Delta (Δ):** ∂V/∂S - sensitivity to underlying price changes
- **Gamma (Γ):** ∂²V/∂S² - rate of change of delta
- **Vega (ν):** ∂V/∂σ - sensitivity to volatility changes
- **Theta (Θ):** ∂V/∂t - time decay rate
- **Rho (ρ):** ∂V/∂r - sensitivity to interest rate changes

### Monte Carlo Simulation

Options are valued through risk-neutral expectation:

```
V₀ = e^(-rT) E^Q[max(S_T - K, 0)]

where S_T = S₀ exp[(r - σ²/2)T + σ√T·Z], Z ~ N(0,1)
```

## Model Validation

All three pricing models demonstrate convergence:

```
Strike: $188, Spot: $188.52, T: 30 days, σ: 42%

Black-Scholes:  $9.64
Monte Carlo:    $9.66 (±0.05)
Binomial Tree:  $9.64

Maximum deviation: $0.02 (0.2%)
```

## Sensitivity Analysis

The engine performs systematic parameter sweeps:

**Volatility Sensitivity:** Tests option response across 10%-80% volatility range
**Maturity Sensitivity:** Analyzes time decay from 5 to 180 days
**Rate Sensitivity:** Measures interest rate impact from 0% to 10%

## Delta-Hedging Simulation

Simulates dynamic hedging strategies:
- Geometric Brownian motion price paths
- Configurable rebalancing frequency
- P&L decomposition (directional vs gamma)
- Transaction cost analysis

## Results

Example comparative analysis across moneyness levels demonstrates Greek concentration at-the-money:

| Metric | ITM (K=150) | ATM (K=188) | OTM (K=220) |
|--------|-------------|-------------|-------------|
| Delta | 0.977 | 0.545 | 0.117 |
| Gamma | 0.0024 | 0.0175 | 0.0086 |
| Vega | 0.030 | 0.214 | 0.106 |

The ATM position exhibits 7x higher gamma and vega compared to deep ITM, validating theoretical predictions.

## References

1. Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.

2. Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach. *Journal of Financial Economics*, 7(3), 229-263.

3. Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.

4. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.

## License

MIT License - See LICENSE file for details.

## Author

Edouard Lavaud  
