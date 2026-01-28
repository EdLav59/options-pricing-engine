# Options Pricing & Greeks Calculator

A multi-model options pricing engine implementing Black-Scholes, Monte Carlo, and Binomial Tree methodologies with comprehensive sensitivity analysis and dynamic delta-hedging simulation.

## Overview

This project implements three fundamental option pricing models alongside advanced risk analytics including Greeks calculation, multi-dimensional sensitivity analysis, and delta-hedging performance simulation. The engine supports both European and American options with real-time market data integration.

## Core Features

**Pricing Models:**
- Black-Scholes analytical solution for European options
- Monte Carlo simulation with 100,000+ paths and variance estimates
- Cox-Ross-Rubinstein binomial tree for American options

**Risk Analytics:**
- Newton-Raphson implied volatility solver
- Comprehensive Greeks (Delta, Gamma, Vega, Theta, Rho)
- Multi-dimensional sensitivity analysis across volatility, maturity, and interest rates
- Dynamic delta-hedging simulator with configurable rebalancing frequency
- Gamma P&L decomposition and transaction cost analysis

**Visualization:**
- Interactive Plotly dashboards
- 3D option price surfaces
- Greeks sensitivity profiles across moneyness spectrum
- Volatility smile construction
- Real-time hedging performance metrics

**Example inputs:**
```
Ticker: NVDA
Strike: 188
Days to Maturity: 30
Volatility: 42%
Risk-Free Rate: 4.5%
Simulations: 100000
```

## Mathematical Framework

### Black-Scholes Model

European option valuation through closed-form solution:

```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Greeks

First and second-order partial derivatives quantifying option sensitivities:

- **Delta (Δ):** ∂V/∂S - directional exposure to underlying price
- **Gamma (Γ):** ∂²V/∂S² - delta convexity and rehedging frequency
- **Vega (ν):** ∂V/∂σ - exposure to volatility changes
- **Theta (Θ):** ∂V/∂t - time decay rate
- **Rho (ρ):** ∂V/∂r - interest rate sensitivity

### Monte Carlo Simulation

Risk-neutral valuation through path simulation:

```
V₀ = e^(-rT) E^Q[max(S_T - K, 0)]

where S_T = S₀ exp[(r - σ²/2)T + σ√T·Z], Z ~ N(0,1)
```

## Empirical Results

### Test Configuration

**Underlying:** NVIDIA (NVDA)  
**Spot Price:** $188.52  
**Time to Maturity:** 30 days  
**Volatility:** 42% annualized  
**Risk-Free Rate:** 4.5%

### Comparative Analysis Across Moneyness

| Metric | Deep ITM (K=150) | ATM (K=188) | OTM (K=220) |
|--------|------------------|-------------|-------------|
| **Call Price** | $39.28 | $9.64 | $1.23 |
| **Intrinsic Value** | $38.52 | $0.52 | $0.00 |
| **Time Value** | $0.76 | $9.12 | $1.23 |
| **Delta** | 0.977 | 0.545 | 0.117 |
| **Gamma** | 0.0024 | 0.0175 | 0.0086 |
| **Vega** | 0.030 | 0.214 | 0.106 |
| **Theta (daily)** | -$0.04 | -$0.16 | -$0.08 |

### Model Convergence

All three pricing models demonstrate convergence within acceptable tolerance:

**ATM Option (K=$188):**
```
Black-Scholes:  $9.64
Monte Carlo:    $9.66 ± $0.05
Binomial Tree:  $9.64
Maximum Deviation: $0.02 (0.2%)
```

### Key Findings

**1. Greek Concentration at ATM**

Gamma and vega maximize at-the-money, with values approximately 7x higher than deep ITM positions:
- ATM gamma (0.0175) drives frequent rehedging requirements
- ATM vega (0.214) creates maximum volatility exposure
- Confirms theoretical predictions of Greek behavior across strike spectrum

**2. Volatility Sensitivity**

Systematic analysis across 10%-80% volatility range:
- Deep ITM: 8% total price change (minimal optionality)
- ATM: 532% price change ($2.81 → $17.77)
- OTM: High percentage sensitivity on small base

ATM options optimal for volatility trading strategies (straddles/strangles).

**3. Time Decay Patterns**

Non-linear theta effect quantified across maturity spectrum:
- ATM theta 4x higher than deep ITM
- Time value accelerates toward expiration
- Daily decay: ITM -$0.04 (1%), ATM -$0.16 (1.7%), OTM -$0.08 (6.5%)

**4. Delta-Hedging Performance**

Dynamic hedging simulation with daily rebalancing (27 rebalances):

| Strike | Final P&L | Gamma P&L | P&L Volatility |
|--------|-----------|-----------|----------------|
| K=150 (ITM) | -$6.68 | +$1.97 | $20.34 |
| K=188 (ATM) | -$9.58 | +$4.94 | $9.24 |
| K=220 (OTM) | -$1.51 | +$1.14 | $1.51 |

All simulations exhibited losses due to 12% adverse stock movement ($188.52 → $165.77). ATM case demonstrated gamma/transaction cost tradeoff: despite 27 daily rebalances, P&L volatility remained at $9.24, though gamma P&L of +$4.94 partially offset directional losses.

## Results Interpretation

The comparative analysis validates theoretical option pricing frameworks:

1. **Moneyness Effects:** Greeks concentration at ATM has direct implications for volatility trading and hedging frequency
2. **Model Robustness:** <$0.07 deviation across models demonstrates implementation accuracy
3. **Risk Management:** Sensitivity analysis quantifies exposure across multiple dimensions
4. **Practical Hedging:** Simulation demonstrates gamma/transaction cost tradeoffs inherent in dynamic hedging

## References

1. Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.

2. Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach. *Journal of Financial Economics*, 7(3), 229-263.

3. Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.

4. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.

## License

MIT License

## Author

Edouard Lavalard  
