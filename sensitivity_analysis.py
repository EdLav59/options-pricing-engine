"""
Sensitivity Analysis & Hedging Simulation Module

This module extends the options pricing engine with:
- Systematic sensitivity analysis (volatility, maturity, interest rates)
- Delta-hedging simulation with P&L tracking
- Market vs Theoretical price comparison
- Greeks interpretation and risk management insights
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================================================
# SENSITIVITY ANALYSIS ENGINE
# ======================================================

class SensitivityAnalyzer:
    """Systematic analysis of option price sensitivities"""
    
    def __init__(self, base_params):
        """
        Args:
            base_params: dict with keys S, K, T, r, sigma
        """
        self.base = base_params
    
    def volatility_sensitivity(self, vol_range=None):
        """
        Analyze option price sensitivity to volatility changes
        
        Returns:
            DataFrame with volatility levels and corresponding prices/Greeks
        """
        from quant_engine import BlackScholesEngine
        
        if vol_range is None:
            vol_range = np.linspace(0.1, 0.8, 25)
        
        results = []
        for vol in vol_range:
            bs = BlackScholesEngine(
                self.base['S'], self.base['K'], self.base['T'], 
                self.base['r'], vol
            )
            call, put = bs.price()
            greeks = bs.greeks()
            
            results.append({
                'Volatility': vol * 100,
                'Call_Price': call,
                'Put_Price': put,
                'Delta': greeks['Delta'],
                'Vega': greeks['Vega'],
                'Gamma': greeks['Gamma']
            })
        
        return pd.DataFrame(results)
    
    def maturity_sensitivity(self, days_range=None):
        """
        Analyze option price sensitivity to time to maturity
        
        Returns:
            DataFrame with maturity levels and corresponding prices/Greeks
        """
        from quant_engine import BlackScholesEngine
        
        if days_range is None:
            days_range = np.linspace(5, 180, 25)
        
        results = []
        for days in days_range:
            T = days / 365
            bs = BlackScholesEngine(
                self.base['S'], self.base['K'], T, 
                self.base['r'], self.base['sigma']
            )
            call, put = bs.price()
            greeks = bs.greeks()
            
            results.append({
                'Days_to_Maturity': days,
                'Call_Price': call,
                'Put_Price': put,
                'Theta': greeks['Theta'],
                'Delta': greeks['Delta'],
                'Gamma': greeks['Gamma']
            })
        
        return pd.DataFrame(results)
    
    def rate_sensitivity(self, rate_range=None):
        """
        Analyze option price sensitivity to interest rate changes
        
        Returns:
            DataFrame with rate levels and corresponding prices/Greeks
        """
        from quant_engine import BlackScholesEngine
        
        if rate_range is None:
            rate_range = np.linspace(0.0, 0.10, 25)
        
        results = []
        for rate in rate_range:
            bs = BlackScholesEngine(
                self.base['S'], self.base['K'], self.base['T'], 
                rate, self.base['sigma']
            )
            call, put = bs.price()
            greeks = bs.greeks()
            
            results.append({
                'Interest_Rate': rate * 100,
                'Call_Price': call,
                'Put_Price': put,
                'Rho': greeks['Rho'],
                'Delta': greeks['Delta']
            })
        
        return pd.DataFrame(results)
    
    def comprehensive_sensitivity_report(self):
        """
        Generate comprehensive sensitivity analysis across all parameters
        
        Returns:
            dict with DataFrames for each sensitivity dimension
        """
        print("\n" + "="*70)
        print("   COMPREHENSIVE SENSITIVITY ANALYSIS")
        print("="*70)
        
        # Run all analyses
        vol_sens = self.volatility_sensitivity()
        mat_sens = self.maturity_sensitivity()
        rate_sens = self.rate_sensitivity()
        
        # Summary statistics
        print("\n1. VOLATILITY SENSITIVITY")
        print("-" * 70)
        vol_impact = vol_sens['Call_Price'].max() - vol_sens['Call_Price'].min()
        print(f"   Volatility range: {vol_sens['Volatility'].min():.0f}% - {vol_sens['Volatility'].max():.0f}%")
        print(f"   Call price range: ${vol_sens['Call_Price'].min():.2f} - ${vol_sens['Call_Price'].max():.2f}")
        print(f"   Total impact: ${vol_impact:.2f} ({vol_impact/vol_sens['Call_Price'].min()*100:.1f}% change)")
        
        print("\n2. MATURITY SENSITIVITY")
        print("-" * 70)
        mat_impact = mat_sens['Call_Price'].iloc[0] - mat_sens['Call_Price'].iloc[-1]
        print(f"   Maturity range: {mat_sens['Days_to_Maturity'].min():.0f} - {mat_sens['Days_to_Maturity'].max():.0f} days")
        print(f"   Call price decay: ${mat_sens['Call_Price'].max():.2f} → ${mat_sens['Call_Price'].min():.2f}")
        print(f"   Time decay impact: ${mat_impact:.2f} (theta effect)")
        
        print("\n3. INTEREST RATE SENSITIVITY")
        print("-" * 70)
        rate_impact = rate_sens['Call_Price'].max() - rate_sens['Call_Price'].min()
        print(f"   Rate range: {rate_sens['Interest_Rate'].min():.1f}% - {rate_sens['Interest_Rate'].max():.1f}%")
        print(f"   Call price range: ${rate_sens['Call_Price'].min():.2f} - ${rate_sens['Call_Price'].max():.2f}")
        print(f"   Total impact: ${rate_impact:.2f} (rho effect)")
        
        return {
            'volatility': vol_sens,
            'maturity': mat_sens,
            'interest_rate': rate_sens
        }
    
    def visualize_sensitivities(self):
        """Create interactive visualization of all sensitivities"""
        
        # Generate data
        vol_data = self.volatility_sensitivity()
        mat_data = self.maturity_sensitivity()
        rate_data = self.rate_sensitivity()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Volatility Sensitivity',
                'Time Decay (Theta Effect)',
                'Interest Rate Sensitivity',
                'Greeks vs Spot Price'
            )
        )
        
        # 1. Volatility sensitivity
        fig.add_trace(
            go.Scatter(
                x=vol_data['Volatility'],
                y=vol_data['Call_Price'],
                name='Call',
                line=dict(color='cyan', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=vol_data['Volatility'],
                y=vol_data['Put_Price'],
                name='Put',
                line=dict(color='magenta', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Maturity sensitivity
        fig.add_trace(
            go.Scatter(
                x=mat_data['Days_to_Maturity'],
                y=mat_data['Call_Price'],
                name='Call',
                line=dict(color='lime', width=2),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Rate sensitivity
        fig.add_trace(
            go.Scatter(
                x=rate_data['Interest_Rate'],
                y=rate_data['Call_Price'],
                name='Call',
                line=dict(color='orange', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Greeks comparison
        spot_range = np.linspace(self.base['S']*0.7, self.base['S']*1.3, 50)
        from quant_engine import BlackScholesEngine
        deltas, gammas, vegas = [], [], []
        
        for s in spot_range:
            bs = BlackScholesEngine(s, self.base['K'], self.base['T'], self.base['r'], self.base['sigma'])
            g = bs.greeks()
            deltas.append(g['Delta'])
            gammas.append(g['Gamma'] * 10)  # Scale for visibility
            vegas.append(g['Vega'] / 10)
        
        fig.add_trace(
            go.Scatter(x=spot_range, y=deltas, name='Delta', line=dict(color='cyan')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=spot_range, y=gammas, name='Gamma×10', line=dict(color='red')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=spot_range, y=vegas, name='Vega/10', line=dict(color='yellow')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_xaxes(title_text="Days to Maturity", row=1, col=2)
        fig.update_xaxes(title_text="Interest Rate (%)", row=2, col=1)
        fig.update_xaxes(title_text="Spot Price", row=2, col=2)
        
        fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Call Price ($)", row=1, col=2)
        fig.update_yaxes(title_text="Call Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Greek Value", row=2, col=2)
        
        fig.update_layout(
            title="<b>Comprehensive Sensitivity Analysis</b>",
            height=800,
            template='plotly_dark',
            showlegend=True
        )
        
        fig.show()

# ======================================================
# DELTA HEDGING SIMULATOR
# ======================================================

class DeltaHedgingSimulator:
    """Simulate delta-hedging strategy with P&L tracking"""
    
    def __init__(self, S0, K, T, r, sigma, position_size=1):
        """
        Args:
            S0: Initial spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            position_size: Number of options (positive = long, negative = short)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.position = position_size
    
    def simulate_hedging_path(self, n_steps=50, rebalance_frequency='daily'):
        """
        Simulate delta-hedging over the option's life
        
        Args:
            n_steps: Number of price simulation steps
            rebalance_frequency: 'daily', 'every_step', or 'weekly'
        
        Returns:
            DataFrame with hedging performance metrics
        """
        from quant_engine import BlackScholesEngine
        
        dt = self.T / n_steps
        time_steps = np.linspace(0, self.T, n_steps + 1)
        
        # Simulate stock price path (GBM)
        np.random.seed(42)
        dW = np.random.standard_normal(n_steps)
        prices = [self.S0]
        
        for i in range(n_steps):
            S_new = prices[-1] * np.exp((self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*dW[i])
            prices.append(S_new)
        
        # Initialize tracking
        results = []
        shares_held = 0
        cumulative_pnl = 0
        cash_account = 0
        
        # Set rebalance schedule
        if rebalance_frequency == 'daily':
            rebalance_interval = int(n_steps / (self.T * 252))  # Daily
        elif rebalance_frequency == 'weekly':
            rebalance_interval = int(n_steps / (self.T * 52))   # Weekly
        else:
            rebalance_interval = 1  # Every step
        
        for i, t in enumerate(time_steps):
            S = prices[i]
            T_remaining = self.T - t
            
            if T_remaining > 0.001:  # Not at expiry
                bs = BlackScholesEngine(S, self.K, T_remaining, self.r, self.sigma)
                option_value = bs.price_call() * self.position
                delta = bs.greeks()['Delta']
                gamma = bs.greeks()['Gamma']
                theta = bs.greeks()['Theta']
                
                # Rebalance if needed
                if i == 0 or i % rebalance_interval == 0:
                    target_shares = delta * self.position
                    shares_to_trade = target_shares - shares_held
                    cash_account -= shares_to_trade * S  # Buy/sell shares
                    shares_held = target_shares
                    rebalanced = True
                else:
                    rebalanced = False
                
                # Calculate P&L
                option_pnl = option_value
                hedge_pnl = shares_held * S + cash_account
                total_pnl = option_pnl + hedge_pnl
                
            else:  # At expiry
                option_value = max(S - self.K, 0) * self.position
                delta = 1 if S > self.K else 0
                gamma = 0
                theta = 0
                
                # Close hedge
                cash_account += shares_held * S
                shares_held = 0
                
                option_pnl = option_value
                hedge_pnl = cash_account
                total_pnl = option_pnl + hedge_pnl
                rebalanced = True
            
            results.append({
                'Time': t,
                'Days_Remaining': T_remaining * 365,
                'Spot_Price': S,
                'Option_Value': option_value,
                'Delta': delta,
                'Gamma': gamma,
                'Theta': theta,
                'Shares_Held': shares_held,
                'Cash_Account': cash_account,
                'Total_PnL': total_pnl,
                'Rebalanced': rebalanced
            })
        
        df = pd.DataFrame(results)
        
        # Print summary
        print("\n" + "="*70)
        print("   DELTA-HEDGING SIMULATION RESULTS")
        print("="*70)
        print(f"\nPosition: {'LONG' if self.position > 0 else 'SHORT'} {abs(self.position)} Call @ K=${self.K}")
        print(f"Initial Spot: ${self.S0:.2f} | Final Spot: ${prices[-1]:.2f}")
        print(f"Rebalance Frequency: {rebalance_frequency} ({len(df[df['Rebalanced']]):} times)")
        print(f"\nFinal P&L: ${df['Total_PnL'].iloc[-1]:.2f}")
        print(f"Max Drawdown: ${df['Total_PnL'].min():.2f}")
        print(f"P&L Volatility: ${df['Total_PnL'].std():.2f}")
        print(f"Cumulative Gamma P&L: ${(df['Gamma'] * (df['Spot_Price'].diff()**2)).sum():.2f}")
        print("="*70)
        
        return df
    
    def visualize_hedging_performance(self, hedging_df):
        """Visualize delta-hedging performance"""
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Stock Price Path & Option Value',
                'Delta Hedge Position',
                'Cumulative P&L'
            ),
            vertical_spacing=0.1
        )
        
        # 1. Price path
        fig.add_trace(
            go.Scatter(
                x=hedging_df['Time'],
                y=hedging_df['Spot_Price'],
                name='Spot Price',
                line=dict(color='cyan', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=hedging_df['Time'],
                y=hedging_df['Option_Value'],
                name='Option Value',
                line=dict(color='magenta', width=2),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # 2. Hedge position
        fig.add_trace(
            go.Scatter(
                x=hedging_df['Time'],
                y=hedging_df['Shares_Held'],
                name='Shares Held',
                line=dict(color='lime', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Mark rebalance points
        rebalances = hedging_df[hedging_df['Rebalanced']]
        fig.add_trace(
            go.Scatter(
                x=rebalances['Time'],
                y=rebalances['Shares_Held'],
                mode='markers',
                name='Rebalance',
                marker=dict(color='red', size=8, symbol='diamond')
            ),
            row=2, col=1
        )
        
        # 3. P&L evolution
        fig.add_trace(
            go.Scatter(
                x=hedging_df['Time'],
                y=hedging_df['Total_PnL'],
                name='Total P&L',
                line=dict(color='gold', width=2),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # Update axes
        fig.update_xaxes(title_text="Time (years)", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Shares", row=2, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
        
        fig.update_layout(
            title="<b>Delta-Hedging Performance Analysis</b>",
            height=900,
            template='plotly_dark',
            showlegend=True
        )
        
        fig.show()

# ======================================================
# MARKET VS THEORETICAL COMPARISON
# ======================================================

class MarketPriceAnalyzer:
    """Compare theoretical prices with market prices"""
    
    @staticmethod
    def analyze_pricing_errors(options_data, S, T, r):
        """
        Compare Black-Scholes theoretical prices with market prices
        
        Args:
            options_data: DataFrame with columns ['strike', 'mid_price', 'impliedVolatility']
            S: Current spot price
            T: Time to maturity
            r: Risk-free rate
        
        Returns:
            DataFrame with pricing errors and analysis
        """
        from quant_engine import BlackScholesEngine
        
        if options_data is None or len(options_data) == 0:
            print("No market data available for comparison")
            return None
        
        results = []
        
        for _, row in options_data.iterrows():
            K = row['strike']
            market_price = row['mid_price']
            market_iv = row['impliedVolatility']
            
            # Theoretical price using market IV
            bs = BlackScholesEngine(S, K, T, r, market_iv)
            theo_price = bs.price_call()
            
            # Calculate errors
            abs_error = market_price - theo_price
            pct_error = (abs_error / market_price) * 100 if market_price > 0 else 0
            
            results.append({
                'Strike': K,
                'Moneyness': K / S,
                'Market_Price': market_price,
                'Theoretical_Price': theo_price,
                'Absolute_Error': abs_error,
                'Percent_Error': pct_error,
                'Market_IV': market_iv * 100
            })
        
        df = pd.DataFrame(results)
        
        # Summary statistics
        print("\n" + "="*70)
        print("   MARKET VS THEORETICAL PRICE ANALYSIS")
        print("="*70)
        print(f"\nSample Size: {len(df)} options")
        print(f"Mean Absolute Error: ${df['Absolute_Error'].abs().mean():.3f}")
        print(f"Mean Percent Error: {df['Percent_Error'].abs().mean():.2f}%")
        print(f"Max Overpricing: ${df['Absolute_Error'].max():.3f} (K={df.loc[df['Absolute_Error'].idxmax(), 'Strike']:.0f})")
        print(f"Max Underpricing: ${df['Absolute_Error'].min():.3f} (K={df.loc[df['Absolute_Error'].idxmin(), 'Strike']:.0f})")
        
        # Identify arbitrage opportunities
        significant_mispricing = df[df['Percent_Error'].abs() > 5]
        if len(significant_mispricing) > 0:
            print(f"\n[WARNING] Found {len(significant_mispricing)} options with >5% pricing error")
            print("\nPotential Arbitrage Opportunities:")
            for _, row in significant_mispricing.iterrows():
                direction = "OVERPRICED" if row['Absolute_Error'] > 0 else "UNDERPRICED"
                print(f"   K=${row['Strike']:.0f}: {direction} by {abs(row['Percent_Error']):.1f}%")
        
        print("="*70)
        
        return df

# ======================================================
# MAIN ANALYSIS RUNNER
# ======================================================

def run_comprehensive_analysis(ticker, S, K, T, r, sigma, options_data=None):
    """
    Run complete sensitivity analysis, hedging simulation, and market comparison
    
    Args:
        ticker: Stock ticker
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        options_data: Optional market options data
    """
    base_params = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
    
    print("\n" + "="*70)
    print("=" + " "*22 + "ADVANCED ANALYSIS SUITE" + " "*23 + "=")
    print("="*70)
    
    # 1. Sensitivity Analysis
    print("\n[1/4] Running Sensitivity Analysis...")
    analyzer = SensitivityAnalyzer(base_params)
    sensitivity_results = analyzer.comprehensive_sensitivity_report()
    analyzer.visualize_sensitivities()
    
    # 2. Delta Hedging Simulation
    print("\n[2/4] Running Delta-Hedging Simulation...")
    hedger = DeltaHedgingSimulator(S, K, T, r, sigma, position_size=1)
    hedging_results = hedger.simulate_hedging_path(n_steps=100, rebalance_frequency='daily')
    hedger.visualize_hedging_performance(hedging_results)
    
    # 3. Market Price Comparison
    if options_data is not None:
        print("\n[3/4] Analyzing Market vs Theoretical Prices...")
        market_analyzer = MarketPriceAnalyzer()
        pricing_errors = market_analyzer.analyze_pricing_errors(options_data, S, T, r)
    else:
        print("\n[3/4] Skipping market comparison (no data available)")
        pricing_errors = None
    
    # 4. Risk Management Summary
    print("\n[4/4] Generating Risk Management Summary...")
    print_risk_management_report(base_params, sensitivity_results, hedging_results)
    
    return {
        'sensitivity': sensitivity_results,
        'hedging': hedging_results,
        'pricing_errors': pricing_errors
    }

def print_risk_management_report(params, sensitivity, hedging):
    """Print comprehensive risk management report"""
    from quant_engine import BlackScholesEngine
    
    bs = BlackScholesEngine(**params)
    greeks = bs.greeks()
    call_price = bs.price_call()
    
    print("\n" + "="*70)
    print("   RISK MANAGEMENT SUMMARY")
    print("="*70)
    
    print(f"\nPosition: Long 1 Call @ K=${params['K']:.2f}")
    print(f"Current Value: ${call_price:.2f}")
    
    print("\n1. DIRECTIONAL RISK (Delta)")
    print("-" * 70)
    print(f"   Delta: {greeks['Delta']:.4f}")
    print(f"   → For $1 move in stock: P&L changes by ${greeks['Delta']:.2f}")
    print(f"   → Delta-neutral hedge: Short {greeks['Delta']:.0f} shares")
    
    print("\n2. CONVEXITY RISK (Gamma)")
    print("-" * 70)
    print(f"   Gamma: {greeks['Gamma']:.4f}")
    print(f"   → For $1 move: Delta changes by {greeks['Gamma']:.4f}")
    rehedge_cost = greeks['Gamma'] * params['S'] * 0.01  # 1% move
    print(f"   → Rehedge cost for 1% move: ${rehedge_cost:.2f}")
    
    print("\n3. VOLATILITY RISK (Vega)")
    print("-" * 70)
    print(f"   Vega: {greeks['Vega']:.4f}")
    print(f"   → For 1% vol increase: P&L increases by ${greeks['Vega']:.2f}")
    vol_scenarios = [params['sigma'] - 0.05, params['sigma'], params['sigma'] + 0.05]
    print(f"   → Scenarios:")
    for v in vol_scenarios:
        bs_temp = BlackScholesEngine(params['S'], params['K'], params['T'], params['r'], v)
        price_temp = bs_temp.price_call()
        print(f"      Vol {v*100:.0f}%: ${price_temp:.2f} ({price_temp - call_price:+.2f})")
    
    print("\n4. TIME DECAY (Theta)")
    print("-" * 70)
    print(f"   Theta: {greeks['Theta']:.4f}")
    print(f"   → Daily P&L from time decay: ${greeks['Theta']:.2f}")
    print(f"   → Weekly decay: ${greeks['Theta'] * 7:.2f}")
    
    print("\n5. HEDGING PERFORMANCE")
    print("-" * 70)
    if hedging is not None:
        final_pnl = hedging['Total_PnL'].iloc[-1]
        pnl_vol = hedging['Total_PnL'].std()
        print(f"   Simulation Final P&L: ${final_pnl:.2f}")
        print(f"   P&L Volatility: ${pnl_vol:.2f}")
        print(f"   → Hedging {'reduced' if pnl_vol < call_price*0.5 else 'did not reduce'} risk")
    
    print("\n6. KEY RECOMMENDATIONS")
    print("-" * 70)
    if greeks['Gamma'] > 0.02:
        print("   [WARNING] High Gamma → Rehedge frequently (daily)")
    else:
        print("   [OK] Low Gamma → Rehedge less frequently (weekly)")
    
    if greeks['Vega'] > call_price * 0.05:
        print("   [WARNING] High Vega exposure → Consider vol hedging")
    
    if abs(greeks['Theta']) > call_price * 0.03:
        print("   [WARNING] Significant time decay → Monitor expiration risk")
    
    print("="*70)

if __name__ == "__main__":
    print("Sensitivity Analysis & Hedging Module Loaded")
    print("Use: run_comprehensive_analysis(ticker, S, K, T, r, sigma)")
