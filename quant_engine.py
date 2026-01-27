import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import sys

# Configuration du style des graphiques
sns.set_theme(style="darkgrid")

# ==========================================
# MODEL 1: BLACK-SCHOLES (Analytical)
# ==========================================
class BlackScholesEngine:
    def __init__(self, S, K, T, r, sigma):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    def _d1_d2(self):
        # Protection contre la division par zéro si T est très proche de 0
        safe_T = max(self.T, 1e-5)
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * safe_T) / (self.sigma * np.sqrt(safe_T))
        d2 = d1 - self.sigma * np.sqrt(safe_T)
        return d1, d2

    def price(self):
        d1, d2 = self._d1_d2()
        call = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        put = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return call, put

    def get_greeks(self):
        d1, d2 = self._d1_d2()
        safe_T = max(self.T, 1e-5)
        
        delta_c = norm.cdf(d1)
        delta_p = delta_c - 1
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(safe_T))
        vega = self.S * norm.pdf(d1) * np.sqrt(safe_T) / 100
        theta_c = (- (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(safe_T)) 
                   - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) / 365
        rho_c = (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)) / 100
        
        return {
            "Delta Call": delta_c, "Delta Put": delta_p,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta_c,
            "Rho": rho_c
        }

# ==========================================
# MODEL 2: MONTE CARLO (Simulation)
# ==========================================
class MonteCarloEngine:
    def __init__(self, S, K, T, r, sigma, simulations=50000):
        self.S = S; self.K = K; self.T = T; self.r = r; self.sigma = sigma
        self.sims = int(simulations)

    def price(self):
        z = np.random.standard_normal(self.sims)
        ST = self.S * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma * np.sqrt(self.T) * z)
        
        call_payoff = np.maximum(ST - self.K, 0)
        put_payoff = np.maximum(self.K - ST, 0)
        
        discount = np.exp(-self.r * self.T)
        return discount * np.mean(call_payoff), discount * np.mean(put_payoff)

# ==========================================
# MODEL 3: BINOMIAL TREE (American Options)
# ==========================================
class BinomialTreeEngine:
    def __init__(self, S, K, T, r, sigma, steps=200):
        self.S = S; self.K = K; self.T = T; self.r = r; self.sigma = sigma
        self.N = int(steps)

    def price_american(self):
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        asset_prices = self.S * (u ** np.arange(self.N, -1, -1)) * (d ** np.arange(0, self.N + 1, 1))
        
        call_values = np.maximum(asset_prices - self.K, 0)
        put_values = np.maximum(self.K - asset_prices, 0)

        for i in range(self.N - 1, -1, -1):
            asset_prices = self.S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1, 1))
            
            call_hold = np.exp(-self.r * dt) * (p * call_values[:-1] + (1 - p) * call_values[1:])
            put_hold = np.exp(-self.r * dt) * (p * put_values[:-1] + (1 - p) * put_values[1:])
            
            call_values = np.maximum(asset_prices - self.K, call_hold)
            put_values = np.maximum(self.K - asset_prices, put_hold)

        return call_values[0], put_values[0]

# ==========================================
# VISUALIZATION ENGINE (Matplotlib)
# ==========================================
class Visualizer:
    def __init__(self, ticker, S, K, T, r, sigma):
        self.ticker = ticker; self.S = S; self.K = K; self.T = T; self.r = r; self.sigma = sigma

    def show_dashboard(self):
        print("   [PLOT] Generating Dashboard windows...")
        
        # Figure 1: 3D Surface
        self._plot_3d_surface()
        
        # Figure 2: Greeks Heatmap
        self._plot_greeks_heatmap()
        
        # Figure 3: Volatility Smile
        self._plot_vol_smile()
        
        print("   [INFO] Rendering plots. Close windows to exit.")
        plt.show()

    def _plot_3d_surface(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        spot_range = np.linspace(self.S * 0.5, self.S * 1.5, 30)
        time_range = np.linspace(0.01, self.T, 30)
        S_mesh, T_mesh = np.meshgrid(spot_range, time_range)
        
        # Vectorized Black Scholes for the mesh
        d1 = (np.log(S_mesh / self.K) + (self.r + 0.5 * self.sigma ** 2) * T_mesh) / (self.sigma * np.sqrt(T_mesh))
        d2 = d1 - self.sigma * np.sqrt(T_mesh)
        Z = S_mesh * norm.cdf(d1) - self.K * np.exp(-self.r * T_mesh) * norm.cdf(d2)

        surf = ax.plot_surface(S_mesh, T_mesh, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.set_title(f'3D Price Surface: {self.ticker}')
        ax.set_xlabel('Spot Price')
        ax.set_ylabel('Time (Years)')
        ax.set_zlabel('Option Price')
        fig.colorbar(surf, shrink=0.5, aspect=5)

    def _plot_greeks_heatmap(self):
        plt.figure(figsize=(10, 6))
        
        spot_range = np.linspace(self.S * 0.8, self.S * 1.2, 20)
        vol_range = np.linspace(0.1, 0.6, 20)
        
        gamma_matrix = np.zeros((len(vol_range), len(spot_range)))
        
        for i, v in enumerate(vol_range):
            for j, s in enumerate(spot_range):
                safe_T = max(self.T, 1e-5)
                d1_loc = (np.log(s / self.K) + (self.r + 0.5 * v ** 2) * safe_T) / (v * np.sqrt(safe_T))
                gamma_matrix[i, j] = norm.pdf(d1_loc) / (s * v * np.sqrt(safe_T))

        sns.heatmap(gamma_matrix, xticklabels=np.round(spot_range, 1), yticklabels=np.round(vol_range, 2), cmap="magma")
        plt.title('Gamma Sensitivity Heatmap (Spot vs Volatility)')
        plt.xlabel('Spot Price')
        plt.ylabel('Volatility')

    def _plot_vol_smile(self):
        try:
            stock = yf.Ticker(self.ticker)
            if not stock.options:
                return
            
            exp_date = stock.options[0]
            opt_chain = stock.option_chain(exp_date)
            calls = opt_chain.calls
            
            # Filter
            calls = calls[(calls['volume'] > 2) & (calls['impliedVolatility'] > 0.05)]
            
            if calls.empty:
                return

            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=calls, x='strike', y='impliedVolatility', size='volume', hue='impliedVolatility', palette='coolwarm', sizes=(20, 200))
            
            # Smooth curve fit
            sns.lineplot(data=calls, x='strike', y='impliedVolatility', color='black', alpha=0.3)
            
            plt.title(f'Implied Volatility Smile: {self.ticker} (Exp: {exp_date})')
            plt.xlabel('Strike Price')
            plt.ylabel('Implied Volatility')
            
        except Exception as e:
            print(f"   [WARNING] Could not fetch market data for smile: {e}")

# ==========================================
# MAIN INTERFACE
# ==========================================
def main():
    print("\n" + "="*60)
    print("      QUANT ENGINE: OPTION PRICING & VISUALIZATION      ")
    print("="*60)

    # 1. Inputs
    ticker = input(">>> Ticker (e.g., SPY, NVDA, BTC-USD): ").upper().strip()
    
    print(f"   Fetching live data for {ticker}...")
    try:
        spot_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
        print(f"   Detected Spot Price: ${spot_price:.2f}")
        choice = input(f"   Use this price? (y/n): ").lower()
        if choice == 'n':
            spot_price = float(input("   >>> Enter Manual Spot Price: "))
    except:
        print("   [!] Could not fetch live price.")
        spot_price = float(input("   >>> Enter Manual Spot Price: "))

    print("-" * 60)
    K = float(input(">>> Strike Price ($): "))
    days = float(input(">>> Days to Maturity: "))
    T = days / 365.0
    vol = float(input(">>> Volatility % (e.g., 20): ")) / 100.0
    r = float(input(">>> Risk-Free Rate % (e.g., 4.5): ")) / 100.0

    print("\n" + "="*60)
    print("RUNNING COMPUTATIONAL ENGINES...")
    print("="*60)

    # 2. Engines Execution
    bs = BlackScholesEngine(spot_price, K, T, r, vol)
    bs_c, bs_p = bs.price()
    greeks = bs.get_greeks()

    mc = MonteCarloEngine(spot_price, K, T, r, vol)
    mc_c, mc_p = mc.price()

    bt = BinomialTreeEngine(spot_price, K, T, r, vol)
    am_c, am_p = bt.price_american()

    # 3. Output Table
    print(f"{'MODEL':<25} | {'CALL PRICE':<15} | {'PUT PRICE':<15}")
    print("-" * 60)
    print(f"{'Black-Scholes (European)':<25} | {bs_c:<15.4f} | {bs_p:<15.4f}")
    print(f"{'Monte Carlo (European)':<25} | {mc_c:<15.4f} | {mc_p:<15.4f}")
    print(f"{'Binomial Tree (American)':<25} | {am_c:<15.4f} | {am_p:<15.4f}")
    print("-" * 60)
    
    print("GREEKS (Black-Scholes):")
    print(f"Delta (C): {greeks['Delta Call']:<8.4f} | Gamma: {greeks['Gamma']:<8.4f} | Vega: {greeks['Vega']:<8.4f}")
    print(f"Theta:     {greeks['Theta']:<8.4f} | Rho:   {greeks['Rho']:<8.4f}")
    print("=" * 60)

    # 4. Visualizations
    gen_dash = input("\n>>> Open Visualizations (3D Surface/Heatmap/Smile)? (y/n): ").lower()
    if gen_dash == 'y':
        viz = Visualizer(ticker, spot_price, K, T, r, vol)
        viz.show_dashboard()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
