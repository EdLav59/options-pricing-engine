import sys
import subprocess

# --- AUTO-INSTALLER ---
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"[SYSTEM] Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

for pkg in ['yfinance', 'requests', 'plotly', 'scipy', 'numpy', 'pandas']:
    install_and_import(pkg)

# --- IMPORTS ---
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton
import yfinance as yf
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ======================================================
# DATA FETCHING MODULE
# ======================================================
class DataFetcher:
    """Handles robust price and options data fetching"""
    
    @staticmethod
    def fetch_spot_price(ticker):
        """Fetch current spot price with fallback mechanism"""
        print(f"   [INFO] Fetching live data for {ticker}...")
        
        # METHOD 1: Try yf.download
        try:
            data = yf.download(ticker, period="1d", progress=False)
            if not data.empty:
                price = data['Close'].iloc[-1]
                if isinstance(price, pd.Series):
                    price = price.iloc[0]
                print(f"   [SUCCESS] Spot Price: ${float(price):.2f}")
                return float(price)
        except Exception:
            pass
        
        # METHOD 2: Manual fallback
        print(f"   [WARNING] Yahoo Finance blocked. Manual input required.")
        while True:
            try:
                manual_price = input(f"   >>> Enter current price of {ticker}: ")
                return float(manual_price)
            except ValueError:
                print("   Invalid number. Try again (e.g., 150.50)")
    
    @staticmethod
    def fetch_options_chain(ticker, spot_price):
        """Attempt to fetch real options data for volatility smile"""
        print(f"   [INFO] Attempting to fetch options chain for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            if len(expirations) == 0:
                print("   [WARNING] No options data available")
                return None
            
            # Get nearest expiration
            nearest_exp = expirations[0]
            chain = stock.option_chain(nearest_exp)
            calls = chain.calls
            
            # Filter for liquid options
            liquid_calls = calls[
                (calls['volume'] > 10) & 
                (calls['bid'] > 0) & 
                (calls['ask'] > 0)
            ].copy()
            
            if len(liquid_calls) < 3:
                print("   [WARNING] Insufficient liquid options")
                return None
            
            # Calculate moneyness
            liquid_calls['moneyness'] = liquid_calls['strike'] / spot_price
            liquid_calls['mid_price'] = (liquid_calls['bid'] + liquid_calls['ask']) / 2
            
            print(f"   [SUCCESS] Found {len(liquid_calls)} liquid options")
            return liquid_calls[['strike', 'mid_price', 'moneyness', 'impliedVolatility']]
            
        except Exception as e:
            print(f"   [WARNING] Options chain fetch failed: {str(e)[:50]}")
            return None

# ======================================================
# PRICING ENGINES
# ======================================================
class BlackScholesEngine:
    """Black-Scholes analytical pricing model"""
    
    def __init__(self, S, K, T, r, sigma):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
    
    def _d1_d2(self):
        """Calculate d1 and d2 parameters"""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2
    
    def price_call(self):
        """European call option price"""
        d1, d2 = self._d1_d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def price_put(self):
        """European put option price"""
        d1, d2 = self._d1_d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
    
    def price(self):
        """Return both call and put prices"""
        return self.price_call(), self.price_put()
    
    def greeks(self):
        """Calculate all standard Greeks"""
        d1, d2 = self._d1_d2()
        sqrt_T = np.sqrt(self.T)
        
        delta_call = norm.cdf(d1)
        gamma = norm.pdf(d1) / (self.S * self.sigma * sqrt_T)
        vega = self.S * norm.pdf(d1) * sqrt_T / 100  # Per 1% change
        theta_call = (
            -(self.S * norm.pdf(d1) * self.sigma) / (2 * sqrt_T) 
            - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        ) / 365  # Per day
        rho_call = (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)) / 100
        
        return {
            'Delta': delta_call,
            'Gamma': gamma,
            'Vega': vega,
            'Theta': theta_call,
            'Rho': rho_call
        }

class MonteCarloEngine:
    """Monte Carlo simulation for option pricing"""
    
    def __init__(self, S, K, T, r, sigma, n_simulations=100000):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.n_sims = int(n_simulations)
    
    def price(self):
        """Price European options via Monte Carlo"""
        np.random.seed(42)  # Reproducibility
        
        # Simulate terminal stock prices
        z = np.random.standard_normal(self.n_sims)
        ST = self.S * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma * np.sqrt(self.T) * z)
        
        # Calculate payoffs
        call_payoffs = np.maximum(ST - self.K, 0)
        put_payoffs = np.maximum(self.K - ST, 0)
        
        # Discount to present value
        discount = np.exp(-self.r * self.T)
        call_price = discount * np.mean(call_payoffs)
        put_price = discount * np.mean(put_payoffs)
        
        # Calculate standard errors
        call_std = discount * np.std(call_payoffs) / np.sqrt(self.n_sims)
        put_std = discount * np.std(put_payoffs) / np.sqrt(self.n_sims)
        
        return call_price, put_price, call_std, put_std

class BinomialTreeEngine:
    """Binomial tree for American options"""
    
    def __init__(self, S, K, T, r, sigma, n_steps=300):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.N = int(n_steps)
    
    def price_american(self):
        """Price American options via binomial tree"""
        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)
        
        # Initialize terminal nodes
        asset_prices = self.S * (u ** np.arange(self.N, -1, -1)) * (d ** np.arange(0, self.N + 1))
        call_values = np.maximum(asset_prices - self.K, 0)
        put_values = np.maximum(self.K - asset_prices, 0)
        
        # Backward induction
        for i in range(self.N - 1, -1, -1):
            asset_prices = self.S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))
            
            # European value
            call_values = disc * (p * call_values[:-1] + (1 - p) * call_values[1:])
            put_values = disc * (p * put_values[:-1] + (1 - p) * put_values[1:])
            
            # American early exercise
            call_values = np.maximum(call_values, asset_prices - self.K)
            put_values = np.maximum(put_values, self.K - asset_prices)
        
        return call_values[0], put_values[0]

# ======================================================
# IMPLIED VOLATILITY MODULE
# ======================================================
class ImpliedVolatilityEngine:
    """Newton-Raphson solver for implied volatility"""
    
    @staticmethod
    def calculate_iv(market_price, S, K, T, r, option_type='call', initial_guess=0.3):
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price: Observed market price
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            initial_guess: Starting volatility guess
        
        Returns:
            Implied volatility (float) or None if convergence fails
        """
        def objective_function(sigma):
            """Price difference to minimize"""
            try:
                bs = BlackScholesEngine(S, K, T, r, sigma)
                if option_type == 'call':
                    theoretical_price = bs.price_call()
                else:
                    theoretical_price = bs.price_put()
                return theoretical_price - market_price
            except:
                return float('inf')
        
        def vega_function(sigma):
            """Derivative for Newton-Raphson"""
            try:
                bs = BlackScholesEngine(S, K, T, r, sigma)
                return bs.greeks()['Vega'] * 100  # Adjust for vega scaling
            except:
                return 1e-10
        
        try:
            # Newton-Raphson with bounds
            iv = newton(objective_function, initial_guess, fprime=vega_function, maxiter=100, tol=1e-6)
            
            # Sanity checks
            if 0.01 < iv < 3.0:  # Reasonable IV range (1% to 300%)
                return iv
            else:
                return None
        except:
            return None
    
    @staticmethod
    def build_volatility_smile(options_data, S, T, r):
        """
        Build volatility smile from options chain data
        
        Args:
            options_data: DataFrame with columns ['strike', 'mid_price', 'moneyness']
            S: Current spot price
            T: Time to maturity (years)
            r: Risk-free rate
        
        Returns:
            DataFrame with strikes, moneyness, and implied volatilities
        """
        if options_data is None or len(options_data) == 0:
            return None
        
        ivs = []
        valid_strikes = []
        valid_moneyness = []
        
        for _, row in options_data.iterrows():
            strike = row['strike']
            market_price = row['mid_price']
            
            # Calculate IV
            iv = ImpliedVolatilityEngine.calculate_iv(
                market_price, S, strike, T, r, option_type='call'
            )
            
            if iv is not None:
                ivs.append(iv * 100)  # Convert to percentage
                valid_strikes.append(strike)
                valid_moneyness.append(row['moneyness'])
        
        if len(ivs) < 3:
            return None
        
        return pd.DataFrame({
            'Strike': valid_strikes,
            'Moneyness': valid_moneyness,
            'Implied_Volatility': ivs
        }).sort_values('Strike')

# ======================================================
# VISUALIZATION MODULE
# ======================================================
class InteractiveDashboard:
    """Interactive Plotly visualizations"""
    
    def __init__(self, ticker, S, K, T, r, sigma, greeks, smile_data=None):
        self.ticker = ticker
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.greeks = greeks
        self.smile_data = smile_data
    
    def create_dashboard(self):
        """Generate comprehensive interactive dashboard"""
        print("\n   [INFO] Generating interactive visualizations...")
        
        # Create subplots with explicit domain control
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{self.ticker} Call Price Surface',
                'Greeks Sensitivity Analysis',
                'Implied Volatility Smile',
                'Gamma Risk Heatmap'
            ),
            specs=[
                [{'type': 'surface', 'is_3d': True}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}]
            ],
            vertical_spacing=0.18,
            horizontal_spacing=0.15,
            column_widths=[0.48, 0.48],
            row_heights=[0.48, 0.48]
        )
        
        # 1. 3D Price Surface
        self._add_price_surface(fig)
        
        # 2. Greeks Sensitivity
        self._add_greeks_analysis(fig)
        
        # 3. Implied Volatility Smile
        self._add_volatility_smile(fig)
        
        # 4. Gamma Heatmap
        self._add_gamma_heatmap(fig)
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f"<b>Options Analytics Dashboard - {self.ticker}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            showlegend=False,  # Disable main legend to avoid overlap
            height=900,
            width=1300,
            template='plotly_dark',
            margin=dict(l=60, r=80, t=100, b=60)
        )
        
        fig.show()
        print("   [SUCCESS] Dashboard generated")
    
    def _add_price_surface(self, fig):
        """3D call option price surface"""
        spot_range = np.linspace(self.S * 0.6, self.S * 1.4, 40)
        time_range = np.linspace(0.02, self.T * 2, 40)
        S_mesh, T_mesh = np.meshgrid(spot_range, time_range)
        
        # Vectorized pricing
        d1 = (np.log(S_mesh / self.K) + (self.r + 0.5 * self.sigma**2) * T_mesh) / (self.sigma * np.sqrt(T_mesh))
        d2 = d1 - self.sigma * np.sqrt(T_mesh)
        price_mesh = S_mesh * norm.cdf(d1) - self.K * np.exp(-self.r * T_mesh) * norm.cdf(d2)
        
        fig.add_trace(
            go.Surface(
                x=S_mesh,
                y=T_mesh,
                z=price_mesh,
                colorscale='Viridis',
                name='Call Price',
                showlegend=False,
                hovertemplate='Spot: $%{x:.0f}<br>Time: %{y:.2f}y<br>Price: $%{z:.2f}<extra></extra>',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Price ($)", side="right"),
                    x=0.44,
                    y=0.76,
                    len=0.35,
                    thickness=12
                )
            ),
            row=1, col=1
        )
        
        fig.update_scenes(
            xaxis_title='Spot Price ($)',
            yaxis_title='Time to Maturity (years)',
            zaxis_title='Call Price ($)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            domain=dict(x=[0, 0.43], y=[0.52, 1.0]),
            row=1, col=1
        )
    
    def _add_greeks_analysis(self, fig):
        """Greeks sensitivity to spot price"""
        spot_range = np.linspace(self.S * 0.7, self.S * 1.3, 100)
        deltas, gammas, vegas = [], [], []
        
        for s in spot_range:
            bs = BlackScholesEngine(s, self.K, self.T, self.r, self.sigma)
            g = bs.greeks()
            deltas.append(g['Delta'])
            gammas.append(g['Gamma'])
            vegas.append(g['Vega'])
        
        # Delta
        fig.add_trace(
            go.Scatter(
                x=spot_range, y=deltas,
                name='Delta',
                line=dict(color='cyan', width=3),
                hovertemplate='Spot: $%{x:.0f}<br>Delta: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Gamma
        fig.add_trace(
            go.Scatter(
                x=spot_range, y=gammas,
                name='Gamma',
                line=dict(color='magenta', width=3),
                yaxis='y2',
                hovertemplate='Spot: $%{x:.0f}<br>Gamma: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Spot Price ($)", row=1, col=2)
        fig.update_yaxes(title_text="Delta", row=1, col=2)
        
        # Add annotations to identify lines since legend is disabled
        fig.add_annotation(
            text="Delta (Cyan)",
            xref="x2", yref="y2",
            x=spot_range[-10], y=deltas[-10],
            showarrow=False,
            font=dict(size=11, color='cyan'),
            xanchor='left',
            row=1, col=2
        )
        fig.add_annotation(
            text="Gamma (Magenta)",
            xref="x2", yref="y2",
            x=spot_range[10], y=gammas[10],
            showarrow=False,
            font=dict(size=11, color='magenta'),
            xanchor='left',
            row=1, col=2
        )
    
    def _add_volatility_smile(self, fig):
        """Implied volatility smile from market data"""
        if self.smile_data is not None and len(self.smile_data) >= 3:
            # Real market data
            fig.add_trace(
                go.Scatter(
                    x=self.smile_data['Moneyness'],
                    y=self.smile_data['Implied_Volatility'],
                    mode='markers+lines',
                    name='Market IV',
                    marker=dict(size=10, color='lime', symbol='circle'),
                    line=dict(width=3, dash='dash'),
                    hovertemplate='Moneyness: %{x:.3f}<br>IV: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add current strike reference
            fig.add_vline(
                x=self.K/self.S, 
                line_dash="dot", 
                line_color="red",
                line_width=2,
                annotation_text="Current Strike",
                annotation_position="top",
                row=2, col=1
            )
            
            # Add label for market IV line
            mid_idx = len(self.smile_data) // 2
            fig.add_annotation(
                text="Market IV (Lime)",
                xref="x3", yref="y3",
                x=self.smile_data['Moneyness'].iloc[mid_idx],
                y=self.smile_data['Implied_Volatility'].iloc[mid_idx] + 2,
                showarrow=False,
                font=dict(size=10, color='lime'),
                xanchor='center',
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Moneyness (K/S)", row=2, col=1)
            fig.update_yaxes(title_text="Implied Volatility (%)", row=2, col=1)
        else:
            # Theoretical smile (placeholder)
            moneyness = np.linspace(0.7, 1.3, 50)
            theoretical_iv = self.sigma * 100 * (1 + 0.3 * (moneyness - 1)**2)
            
            fig.add_trace(
                go.Scatter(
                    x=moneyness,
                    y=theoretical_iv,
                    name='Theoretical IV',
                    line=dict(color='orange', width=3, dash='dot'),
                    hovertemplate='Moneyness: %{x:.3f}<br>IV: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_annotation(
                text="<i>Market data unavailable<br>Theoretical smile shown</i>",
                xref="x3", yref="y3",
                x=1.0, y=self.sigma * 100 + 0.3,
                showarrow=False,
                font=dict(size=11, color='gray'),
                row=2, col=1
            )
            
            # Add label for the line
            fig.add_annotation(
                text="Theoretical IV (Orange)",
                xref="x3", yref="y3",
                x=1.15, y=self.sigma * 100,
                showarrow=False,
                font=dict(size=10, color='orange'),
                xanchor='left',
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Moneyness (K/S)", row=2, col=1)
            fig.update_yaxes(title_text="Implied Volatility (%)", row=2, col=1)
    
    def _add_gamma_heatmap(self, fig):
        """Gamma exposure heatmap"""
        spot_range = np.linspace(self.S * 0.7, self.S * 1.3, 30)
        vol_range = np.linspace(self.sigma * 0.5, self.sigma * 1.5, 30)
        
        gamma_matrix = np.zeros((len(vol_range), len(spot_range)))
        
        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                bs = BlackScholesEngine(spot, self.K, self.T, self.r, vol)
                gamma_matrix[i, j] = bs.greeks()['Gamma']
        
        fig.add_trace(
            go.Heatmap(
                x=spot_range,
                y=vol_range * 100,
                z=gamma_matrix,
                colorscale='Hot',
                name='Gamma',
                showlegend=False,
                colorbar=dict(
                    title=dict(text="Gamma", side="right"),
                    x=1.0,
                    y=0.24,
                    len=0.35,
                    thickness=12
                ),
                hovertemplate='Spot: $%{x:.0f}<br>Vol: %{y:.1f}%<br>Gamma: %{z:.4f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Spot Price ($)", row=2, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)

# ======================================================
# MAIN EXECUTION
# ======================================================
def main():
    """Main execution flow"""
    print("=" * 70)
    print("     OPTIONS PRICING & GREEKS CALCULATOR (Interactive Dashboard)")
    print("=" * 70)
    
    # User inputs
    ticker = input("\n>>> Ticker Symbol (e.g., NVDA, AAPL): ").upper().strip()
    
    # Fetch data
    fetcher = DataFetcher()
    spot = fetcher.fetch_spot_price(ticker)
    
    # Parameters
    try:
        strike = float(input(">>> Strike Price: "))
        days = float(input(">>> Days to Maturity: "))
        vol = float(input(">>> Volatility % (e.g., 35): "))
        rf = float(input(">>> Risk-Free Rate % (e.g., 4.5): "))
        n_sims = int(input(">>> Monte Carlo Simulations (e.g., 100000): ") or "100000")
    except ValueError:
        print("\n[WARNING] Invalid input. Using defaults.")
        strike, days, vol, rf, n_sims = spot * 1.05, 30, 35, 4.5, 100000
    
    # Normalize parameters
    T = days / 365.0
    r = rf / 100.0
    sigma = vol / 100.0
    
    # Initialize engines
    print("\n" + "-" * 70)
    print("   PRICING ENGINES")
    print("-" * 70)
    
    bs = BlackScholesEngine(spot, strike, T, r, sigma)
    mc = MonteCarloEngine(spot, strike, T, r, sigma, n_sims)
    bt = BinomialTreeEngine(spot, strike, T, r, sigma)
    
    # Calculate prices
    bs_call, bs_put = bs.price()
    mc_call, mc_put, mc_call_std, mc_put_std = mc.price()
    am_call, am_put = bt.price_american()
    greeks = bs.greeks()
    
    # Results table
    print(f"\n{'MODEL':<30} | {'CALL':<12} | {'PUT':<12}")
    print("-" * 70)
    print(f"{'Black-Scholes (European)':<30} | ${bs_call:<11.2f} | ${bs_put:<11.2f}")
    print(f"{'Monte Carlo (European)':<30} | ${mc_call:<11.2f} | ${mc_put:<11.2f}")
    print(f"{'  └─ Std Error':<30} | ±${mc_call_std:<10.2f} | ±${mc_put_std:<10.2f}")
    print(f"{'Binomial Tree (American)':<30} | ${am_call:<11.2f} | ${am_put:<11.2f}")
    print("-" * 70)
    
    # Greeks
    print(f"\n{'GREEKS':<30} | {'VALUE':<12}")
    print("-" * 70)
    for greek, value in greeks.items():
        print(f"{greek:<30} | {value:<12.4f}")
    print("=" * 70)
    
    # Fetch options data and calculate IV smile
    print("\n" + "-" * 70)
    print("   IMPLIED VOLATILITY ANALYSIS")
    print("-" * 70)
    
    options_data = fetcher.fetch_options_chain(ticker, spot)
    smile_data = None
    
    if options_data is not None:
        iv_engine = ImpliedVolatilityEngine()
        smile_data = iv_engine.build_volatility_smile(options_data, spot, T, r)
        
        if smile_data is not None:
            print(f"\n   Volatility Smile (ATM = {sigma*100:.1f}%):")
            print(smile_data.to_string(index=False))
    
    # Interactive dashboard
    dashboard = InteractiveDashboard(ticker, spot, strike, T, r, sigma, greeks, smile_data)
    dashboard.create_dashboard()

if __name__ == "__main__":
    main()
