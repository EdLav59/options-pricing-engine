import sys
import subprocess

# --- AUTO-INSTALLER ---
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"[SYSTEM] Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_and_import('yfinance')
install_and_import('requests')

# --- IMPORTS ---
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Visual Configuration
sns.set_theme(style="darkgrid")

# ======================================================
# DATA FETCHING (ROBUST FALLBACK)
# ======================================================
def fetch_price_robust(ticker):
    print(f"   [INFO] Attempting to fetch live data for {ticker}...")
    
    # METHOD 1: Try yf.download (often more robust than Ticker)
    try:
        # Suppress progress bar to keep output clean
        data = yf.download(ticker, period="1d", progress=False)
        if not data.empty:
            price = data['Close'].iloc[-1]
            # Handle cases where data might be a Series or scalar
            if isinstance(price, pd.Series):
                price = price.iloc[0]
            print(f"   [SUCCESS] Live Price Found: ${float(price):.2f}")
            return float(price)
    except Exception:
        pass # If fails, silently move to method 2

    # METHOD 2: Manual Fallback
    print(f"   [WARNING] Yahoo Finance blocked the request (common on Colab).")
    print(f"   [ACTION] Please verify the price of {ticker} manually.")
    
    while True:
        try:
            manual_price = input(f"   >>> Enter the current price of {ticker}: ")
            return float(manual_price)
        except ValueError:
            print("   Invalid number. Please try again (e.g., 150.50).")

# ======================================================
# COMPUTATIONAL ENGINES
# ======================================================
class BlackScholesEngine:
    def __init__(self, S, K, T, r, sigma):
        self.S, self.K, self.T, self.r, self.sigma = float(S), float(K), float(T), float(r), float(sigma)

    def price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        call = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        put = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return call, put

    def get_greeks(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return {
            "Delta": norm.cdf(d1),
            "Gamma": norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T)),
            "Vega": self.S * norm.pdf(d1) * np.sqrt(self.T) / 100,
            "Theta": (- (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) / 365,
            "Rho": (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)) / 100
        }

class MonteCarloEngine:
    def __init__(self, S, K, T, r, sigma, simulations):
        self.S, self.K, self.T, self.r, self.sigma, self.sims = S, K, T, r, sigma, int(simulations)

    def price(self):
        # Vectorized for performance
        z = np.random.standard_normal(self.sims)
        ST = self.S * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma * np.sqrt(self.T) * z)
        disc = np.exp(-self.r * self.T)
        return disc * np.mean(np.maximum(ST - self.K, 0)), disc * np.mean(np.maximum(self.K - ST, 0))

class BinomialTreeEngine:
    def __init__(self, S, K, T, r, sigma, steps=200):
        self.S, self.K, self.T, self.r, self.sigma, self.N = S, K, T, r, sigma, int(steps)

    def price_american(self):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d, p = 1 / u, (np.exp(self.r * dt) - (1/u)) / (u - (1/u))
        
        # Initialize leaves
        vals_c = np.maximum(self.S * (u ** np.arange(self.N, -1, -1)) * (d ** np.arange(0, self.N + 1, 1)) - self.K, 0)
        vals_p = np.maximum(self.K - self.S * (u ** np.arange(self.N, -1, -1)) * (d ** np.arange(0, self.N + 1, 1)), 0)

        # Backward induction
        for i in range(self.N - 1, -1, -1):
            asset_prices = self.S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1, 1))
            vals_c = np.maximum(asset_prices - self.K, np.exp(-self.r * dt) * (p * vals_c[:-1] + (1 - p) * vals_c[1:]))
            vals_p = np.maximum(self.K - asset_prices, np.exp(-self.r * dt) * (p * vals_p[:-1] + (1 - p) * vals_p[1:]))
        return vals_c[0], vals_p[0]

class Visualizer:
    def __init__(self, ticker, S, K, T, r, sigma):
        self.ticker, self.S, self.K, self.T, self.r, self.sigma = ticker, S, K, T, r, sigma

    def plot_all(self):
        print("   Generating Visualizations...")
        fig = plt.figure(figsize=(18, 5))
        
        # 1. 3D Price Surface
        ax1 = fig.add_subplot(131, projection='3d')
        spot_range = np.linspace(self.S * 0.5, self.S * 1.5, 20)
        time_range = np.linspace(0.01, self.T, 20)
        S_m, T_m = np.meshgrid(spot_range, time_range)
        d1 = (np.log(S_m / self.K) + (self.r + 0.5 * self.sigma ** 2) * T_m) / (self.sigma * np.sqrt(T_m))
        Z = S_m * norm.cdf(d1) - self.K * np.exp(-self.r * T_m) * norm.cdf(d1 - self.sigma * np.sqrt(T_m))
        ax1.plot_surface(S_m, T_m, Z, cmap=cm.viridis)
        ax1.set_title(f'Price Surface: {self.ticker}')
        ax1.set_xlabel('Spot Price'); ax1.set_ylabel('Time (Years)')

        # 2. Gamma Heatmap
        ax2 = fig.add_subplot(132)
        v_range = np.linspace(0.1, 0.8, 20)
        gamma_map = np.zeros((20, 20))
        for i, v in enumerate(v_range):
            for j, s in enumerate(spot_range):
                d1_loc = (np.log(s / self.K) + (self.r + 0.5 * v**2) * self.T) / (v * np.sqrt(self.T))
                gamma_map[i, j] = norm.pdf(d1_loc) / (s * v * np.sqrt(self.T))
        sns.heatmap(gamma_map, xticklabels=np.round(spot_range,0), yticklabels=np.round(v_range,2), cmap="magma", ax=ax2)
        ax2.set_title("Gamma Heatmap (Risk)"); ax2.set_xlabel("Spot"); ax2.set_ylabel("Volatility")

        # 3. Smile (Static Message)
        ax3 = fig.add_subplot(133)
        ax3.text(0.5, 0.5, "Market Data Visualization\nDisabled (Connection Limit)", ha='center')
        ax3.set_title("Implied Volatility Smile")
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()

# ======================================================
# MAIN EXECUTION
# ======================================================
def main():
    print("="*60)
    print(f"      QUANT ENGINE (Robust Mode)")
    print("="*60)
    
    ticker = input(">>> Ticker (e.g., NVDA): ").upper().strip()
    
    # Robust data fetch
    spot = fetch_price_robust(ticker)
    
    try:
        strike = float(input(">>> Strike Price: "))
        days = float(input(">>> Days to Maturity: "))
        vol = float(input(">>> Volatility % (e.g., 40): "))
        rf = float(input(">>> Risk Free Rate % (e.g., 4): "))
    except ValueError:
        print("Invalid input. Using defaults (Strike=100, Days=30, Vol=40%, RiskFree=4%).")
        strike, days, vol, rf = 100.0, 30.0, 40.0, 4.0

    # Parameter normalization
    T = days / 365.0
    r = rf / 100.0
    sigma = vol / 100.0
    
    # Initialize Engines
    bs = BlackScholesEngine(spot, strike, T, r, sigma)
    mc = MonteCarloEngine(spot, strike, T, r, sigma, 50000)
    bt = BinomialTreeEngine(spot, strike, T, r, sigma)
    
    # Run Calculations
    bs_c, bs_p = bs.price()
    mc_c, mc_p = mc.price()
    am_c, am_p = bt.price_american()
    greeks = bs.get_greeks()
    
    # Output Results
    print("\n" + "-"*60)
    print(f"{'MODEL':<25} | {'CALL':<10} | {'PUT':<10}")
    print("-" * 60)
    print(f"{'Black-Scholes (Eur)':<25} | {bs_c:<10.2f} | {bs_p:<10.2f}")
    print(f"{'Monte Carlo (Eur)':<25} | {mc_c:<10.2f} | {mc_p:<10.2f}")
    print(f"{'Binomial (Amer)':<25} | {am_c:<10.2f} | {am_p:<10.2f}")
    print("-" * 60)
    print(f"GREEKS | Delta: {greeks['Delta']:.2f} | Gamma: {greeks['Gamma']:.2f} | Vega: {greeks['Vega']:.2f} | Theta: {greeks['Theta']:.2f}")
    print("="*60)

    # Plot
    Visualizer(ticker, spot, strike, T, r, sigma).plot_all()

if __name__ == "__main__":
    main()
