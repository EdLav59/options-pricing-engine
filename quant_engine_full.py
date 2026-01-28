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

# Import all classes from quant_engine
from quant_engine import (
    DataFetcher,
    BlackScholesEngine,
    MonteCarloEngine,
    BinomialTreeEngine,
    ImpliedVolatilityEngine,
    InteractiveDashboard
)

# Import sensitivity analysis module
from sensitivity_analysis import (
    SensitivityAnalyzer,
    DeltaHedgingSimulator,
    MarketPriceAnalyzer,
    run_comprehensive_analysis
)

# ======================================================
# MAIN EXECUTION WITH FULL ANALYSIS
# ======================================================
def main():
    """Main execution flow with comprehensive analysis"""
    print("=" * 80)
    print(" "*15 + "OPTIONS PRICING & RISK ANALYSIS ENGINE")
    print(" "*20 + "(Extended with Sensitivity Analysis)")
    print("=" * 80)
    
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
    
    # ══════════════════════════════════════════════════════
    # PART 1: STANDARD PRICING (from quant_engine.py)
    # ══════════════════════════════════════════════════════
    
    print("\n" + "─" * 80)
    print("   PART 1: OPTIONS PRICING MODELS")
    print("─" * 80)
    
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
    print("─" * 80)
    print(f"{'Black-Scholes (European)':<30} | ${bs_call:<11.2f} | ${bs_put:<11.2f}")
    print(f"{'Monte Carlo (European)':<30} | ${mc_call:<11.2f} | ${mc_put:<11.2f}")
    print(f"{'  └─ Std Error':<30} | ±${mc_call_std:<10.2f} | ±${mc_put_std:<10.2f}")
    print(f"{'Binomial Tree (American)':<30} | ${am_call:<11.2f} | ${am_put:<11.2f}")
    print("─" * 80)
    
    # Greeks
    print(f"\n{'GREEKS':<30} | {'VALUE':<12}")
    print("─" * 80)
    for greek, value in greeks.items():
        print(f"{greek:<30} | {value:<12.4f}")
    print("=" * 80)
    
    # ══════════════════════════════════════════════════════
    # PART 2: IMPLIED VOLATILITY & MARKET DATA
    # ══════════════════════════════════════════════════════
    
    print("\n" + "─" * 80)
    print("   PART 2: IMPLIED VOLATILITY ANALYSIS")
    print("─" * 80)
    
    options_data = fetcher.fetch_options_chain(ticker, spot)
    smile_data = None
    
    if options_data is not None:
        iv_engine = ImpliedVolatilityEngine()
        smile_data = iv_engine.build_volatility_smile(options_data, spot, T, r)
        
        if smile_data is not None:
            print(f"\n   Volatility Smile (ATM = {sigma*100:.1f}%):")
            print(smile_data.to_string(index=False))
    
    # Standard dashboard
    print("\n   Generating standard pricing dashboard...")
    dashboard = InteractiveDashboard(ticker, spot, strike, T, r, sigma, greeks, smile_data)
    dashboard.create_dashboard()
    
    # ══════════════════════════════════════════════════════
    # PART 3: ADVANCED ANALYSIS
    # ══════════════════════════════════════════════════════
    
    print("\n" + "─" * 80)
    print("   PART 3: ADVANCED SENSITIVITY & RISK ANALYSIS")
    print("─" * 80)
    
    # Ask user if they want advanced analysis
    run_advanced = input("\n>>> Run advanced sensitivity & hedging analysis? (y/n): ").lower()
    
    if run_advanced == 'y':
        base_params = {'S': spot, 'K': strike, 'T': T, 'r': r, 'sigma': sigma}
        
        # Run comprehensive analysis
        analysis_results = run_comprehensive_analysis(
            ticker, spot, strike, T, r, sigma, options_data
        )
        
        # ══════════════════════════════════════════════════════
        # PART 4: EXPORT RESULTS (Optional)
        # ══════════════════════════════════════════════════════
        
        export = input("\n>>> Export analysis results to CSV? (y/n): ").lower()
        if export == 'y':
            # Export sensitivity data
            for name, df in analysis_results['sensitivity'].items():
                filename = f"{ticker}_{name}_sensitivity.csv"
                df.to_csv(filename, index=False)
                print(f"   Exported: {filename}")
            
            # Export hedging data
            if analysis_results['hedging'] is not None:
                hedging_file = f"{ticker}_hedging_simulation.csv"
                analysis_results['hedging'].to_csv(hedging_file, index=False)
                print(f"   Exported: {hedging_file}")
            
            # Export pricing errors
            if analysis_results['pricing_errors'] is not None:
                errors_file = f"{ticker}_pricing_errors.csv"
                analysis_results['pricing_errors'].to_csv(errors_file, index=False)
                print(f"   Exported: {errors_file}")
    
    # ══════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("   ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n   Summary for {ticker} ${strike} Call:")
    print(f"   • Theoretical Price: ${bs_call:.2f}")
    print(f"   • Delta: {greeks['Delta']:.3f} (hedge with {int(greeks['Delta']*100)} shares)")
    print(f"   • Gamma: {greeks['Gamma']:.4f} (rehedge for ${greeks['Gamma']*spot*0.01:.2f} per 1% move)")
    print(f"   • Daily Theta: ${greeks['Theta']:.2f}")
    print(f"   • Vega: ${greeks['Vega']:.2f} per 1% vol")
    
    if smile_data is not None:
        iv_spread = smile_data['Implied_Volatility'].max() - smile_data['Implied_Volatility'].min()
        print(f"\n   Volatility Smile Spread: {iv_spread:.1f}%")
        if iv_spread > 10:
            print("   → Significant smile indicates tail risk pricing")
    
    print("\n" + "=" * 80)
    print("   Thank you for using the Options Pricing & Risk Analysis Engine!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
