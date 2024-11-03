import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from trade_simulator import TradeSimulator
from datetime import datetime, timedelta
import random

def run_simulation(strategy='random', seed=None):
    """
    Run simulation with different strategies
    strategy: 'random', 'momentum', 'mean_reversion', or 'mixed'
    seed: Random seed for reproducibility
    """
    if seed:
        np.random.seed(seed)
        random.seed(seed)
    
    # 1. Randomize parameters
    tickers = random.sample([
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 
        'JPM', 'BAC', 'GS', 'V', 'MA',
        'AMZN', 'TSLA', 'AMD', 'INTC', 'CRM'
    ], random.randint(5, 10))  # Random number of stocks
    
    # Random date range (within last 5 years)
    end_date = datetime.now()
    days_back = random.randint(365, 365*5)
    start_date = (end_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    print(f"Selected Tickers: {tickers}")
    print(f"Date Range: {start_date} to {end_date}")
    
    # 2. Fetch data
    print("\nFetching data from Yahoo Finance...")
    prices = pd.DataFrame()
    volumes = pd.DataFrame()
    
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        prices[ticker] = hist['Close']
        volumes[ticker] = hist['Volume']
    
    # 3. Generate weights based on selected strategy
    print(f"\nGenerating portfolio weights using {strategy} strategy...")
    portfolio_weights = generate_weights(prices, strategy)
    
    # 4. Initialize simulator with random parameters
    simulator = TradeSimulator(
        initial_capital=random.uniform(500_000, 5_000_000),
        commission_rate=random.uniform(0.0001, 0.001),
        slippage_rate=random.uniform(0.0001, 0.0005),
        market_impact=random.uniform(0.05, 0.15)
    )
    
    # 5. Run simulation
    results = simulator.simulate_trades(portfolio_weights, prices, volumes)
    
    # 6. Plot results
    plot_results(results, prices)
    
    return results

def generate_weights(prices, strategy='random'):
    """Generate portfolio weights based on different strategies with proper constraints"""
    portfolio_weights = pd.DataFrame(index=prices.index, columns=prices.columns)
    returns = prices.pct_change()
    
    if strategy == 'random':
        for date in prices.index:
            # Generate random weights between 0 and 1
            weights = np.random.uniform(0, 1, len(prices.columns))
            # Normalize to sum to 1
            weights = weights / weights.sum()
            portfolio_weights.loc[date] = weights
            
    elif strategy == 'momentum':
        momentum = returns.rolling(252).mean()
        for date in prices.index:
            if pd.isna(momentum.loc[date]).all():
                weights = np.ones(len(prices.columns)) / len(prices.columns)
            else:
                scores = momentum.loc[date].fillna(0)
                # Ensure no negative weights
                scores = np.maximum(scores, 0)
                # Add small constant to avoid division by zero
                weights = (scores + 1e-6) / (scores + 1e-6).sum()
            portfolio_weights.loc[date] = weights
            
    elif strategy == 'mean_reversion':
        zscore = (prices - prices.rolling(20).mean()) / prices.rolling(20).std()
        for date in prices.index:
            if pd.isna(zscore.loc[date]).all():
                weights = np.ones(len(prices.columns)) / len(prices.columns)
            else:
                scores = -zscore.loc[date].fillna(0)  # Negative zscore for mean reversion
                # Ensure no negative weights
                scores = np.maximum(scores, 0)
                # Add small constant to avoid division by zero
                weights = (scores + 1e-6) / (scores + 1e-6).sum()
            portfolio_weights.loc[date] = weights
            
    elif strategy == 'mixed':
        strategies = ['random', 'momentum', 'mean_reversion']
        base_weights = generate_weights(prices, random.choice(strategies))
        # Add small random noise but maintain non-negativity and sum to 1
        noise = pd.DataFrame(np.random.normal(0, 0.05, base_weights.shape), 
                           index=base_weights.index, columns=base_weights.columns)
        portfolio_weights = base_weights + noise
        # Ensure non-negative weights
        portfolio_weights = portfolio_weights.clip(lower=0)
        # Normalize each row to sum to 1
        portfolio_weights = portfolio_weights.div(portfolio_weights.sum(axis=1), axis=0)
    
    # Final safety check - ensure all weights are between 0 and 1 and sum to 1
    portfolio_weights = portfolio_weights.clip(lower=0, upper=1)
    portfolio_weights = portfolio_weights.div(portfolio_weights.sum(axis=1), axis=0)
    
    return portfolio_weights

def plot_results(results, prices):
    """Create comprehensive visualization of results"""
    # Use a built-in style that's guaranteed to be available
    plt.style.use('default')
    
    # Set figure size and create figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Portfolio Value
    ax1 = plt.subplot(3, 1, 1)
    results['pnl']['portfolio_value'].plot(ax=ax1, label='Portfolio Value')
    ax1.set_title('Portfolio Value Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Daily PnL
    ax2 = plt.subplot(3, 1, 2)
    results['pnl']['pnl'].plot(ax=ax2, label='Daily PnL')
    ax2.set_title('Daily Profit/Loss')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Asset Returns Heatmap
    ax3 = plt.subplot(3, 1, 3)
    returns_corr = prices.pct_change().corr()
    # Use built-in colormap instead of seaborn-specific one
    im = ax3.imshow(returns_corr, aspect='auto', cmap='RdYlBu')
    plt.colorbar(im)
    
    # Add labels for correlation matrix
    ax3.set_xticks(np.arange(len(returns_corr.columns)))
    ax3.set_yticks(np.arange(len(returns_corr.columns)))
    ax3.set_xticklabels(returns_corr.columns, rotation=45, ha='right')
    ax3.set_yticklabels(returns_corr.columns)
    ax3.set_title('Asset Return Correlations')
    
    plt.tight_layout()
    
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Save the main results figure
    plt.savefig('outputs/simulation_results.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create and save portfolio weights plot
    plt.figure(figsize=(15, 6))
    results['holdings'].plot(title='Portfolio Weights Over Time')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('outputs/portfolio_weights.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Run multiple simulations with different strategies
    strategies = ['random', 'momentum', 'mean_reversion', 'mixed']
    
    for strategy in strategies:
        print(f"\nRunning simulation with {strategy.upper()} strategy...")
        results = run_simulation(strategy=strategy, seed=None)  # Remove seed for randomness
        
        print(f"\nSimulation Results ({strategy}):")
        print(f"Initial Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Total Transaction Costs: ${results['total_costs']:,.2f}")
        print(f"Annual Turnover: {results['annual_turnover']:.2%}")
        print("-" * 50)