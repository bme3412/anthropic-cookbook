import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import signal
import warnings
from bokeh.plotting import figure, show, save, output_file
from bokeh.layouts import column, row
from bokeh.palettes import Spectral10
from bokeh.models import ColumnDataSource, HoverTool, Legend, DatetimeTickFormatter, Panel, Tabs
from bokeh.io import output_notebook

def create_interactive_plots(portfolio_weights, strategy_returns, save_path='outputs/interactive_backtest.html'):
    """Create interactive Bokeh plots with updated date formatting"""
    
    # Convert index to datetime if not already
    portfolio_weights.index = pd.to_datetime(portfolio_weights.index)
    strategy_returns.index = pd.to_datetime(strategy_returns.index)
    
    # Calculate cumulative returns and rolling volatility
    cumulative_returns = (1 + strategy_returns).cumprod()
    rolling_vol = strategy_returns.rolling(63).std() * np.sqrt(252)
    
    # Create ColumnDataSource for weights
    weights_source = ColumnDataSource(data={
        'date': portfolio_weights.index,
        **{col: portfolio_weights[col] for col in portfolio_weights.columns},
        'tooltip_date': [d.strftime('%Y-%m-%d') for d in portfolio_weights.index]
    })
    
    # Create ColumnDataSource for performance metrics
    perf_source = ColumnDataSource(data={
        'date': cumulative_returns.index,
        'cumulative_returns': cumulative_returns,
        'rolling_vol': rolling_vol,
        'tooltip_date': [d.strftime('%Y-%m-%d') for d in cumulative_returns.index],
        'returns_pct': (cumulative_returns - 1) * 100
    })
    
    # Common plot settings
    plot_settings = dict(
        height=400,
        width=1200,
        x_axis_type='datetime',
        tools='pan,box_zoom,wheel_zoom,reset,save',
    )
    
    # Portfolio Weights Plot
    weights_plot = figure(
        title='Portfolio Weights Over Time',
        **plot_settings
    )
    
    # Add lines for each asset
    for i, col in enumerate(portfolio_weights.columns):
        weights_plot.line(
            'date', col,
            line_width=2,
            color=Spectral10[i],
            legend_label=col,
            source=weights_source
        )
    
    # Add tooltips for weights
    weights_hover = HoverTool(
        tooltips=[
            ('Date', '@tooltip_date'),
            *[(col, '@{' + col + '}{0.00%}') for col in portfolio_weights.columns]
        ],
        formatters={"@date": "datetime"}
    )
    weights_plot.add_tools(weights_hover)
    
    # Cumulative Returns Plot
    returns_plot = figure(
        title='Strategy Cumulative Returns',
        **plot_settings
    )
    
    returns_plot.line(
        'date', 'returns_pct',
        line_width=2,
        color='navy',
        source=perf_source
    )
    
    # Add tooltips for returns
    returns_hover = HoverTool(
        tooltips=[
            ('Date', '@tooltip_date'),
            ('Return', '@returns_pct{0.00}%')
        ],
        formatters={"@date": "datetime"}
    )
    returns_plot.add_tools(returns_hover)
    
    # Volatility Plot
    vol_plot = figure(
        title='Rolling 63-day Volatility',
        **plot_settings
    )
    
    vol_plot.line(
        'date', 'rolling_vol',
        line_width=2,
        color='red',
        source=perf_source
    )
    
    # Add target volatility line
    vol_plot.line(
        cumulative_returns.index,
        [0.15] * len(cumulative_returns),
        line_width=2,
        color='gray',
        line_dash='dashed',
        legend_label='Target Vol'
    )
    
    # Add tooltips for volatility
    vol_hover = HoverTool(
        tooltips=[
            ('Date', '@tooltip_date'),
            ('Volatility', '@rolling_vol{0.00%}')
        ],
        formatters={"@date": "datetime"}
    )
    vol_plot.add_tools(vol_hover)
    
    # Updated datetime formatter for all plots
    date_formatter = DatetimeTickFormatter(
        hours="%H:%M",
        days="%Y-%m-%d",
        months="%Y-%m",
        years="%Y"
    )
    
    # Style improvements
    for plot in [weights_plot, returns_plot, vol_plot]:
        plot.grid.grid_line_alpha = 0.3
        plot.xaxis.formatter = date_formatter
        plot.xaxis.major_label_orientation = 0.7
        
        # Improve axis labels
        plot.xaxis.axis_label = 'Date'
        if plot == weights_plot:
            plot.yaxis.axis_label = 'Weight (%)'
        elif plot == returns_plot:
            plot.yaxis.axis_label = 'Cumulative Return (%)'
        else:
            plot.yaxis.axis_label = 'Volatility (%)'
    
    # Legend styling
    weights_plot.legend.location = "top_right"
    weights_plot.legend.click_policy = "hide"
    vol_plot.legend.location = "top_right"
    
    # Combine plots
    layout = column(returns_plot, weights_plot, vol_plot, sizing_mode="stretch_width")
    
    # Save to HTML file
    output_file(save_path)
    save(layout)
    
    return layout

class MarketData:
    """Enhanced market data handler with robust error handling"""
    
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.prices = None
        self.volumes = None
        
    def fetch_data(self) -> tuple:
        """Fetch price data with improved error handling and retry logic"""
        prices = pd.DataFrame()
        volumes = pd.DataFrame()
        failed_tickers = []
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date)
                if len(hist) > 0:
                    prices[ticker] = hist['Close']
                    volumes[ticker] = hist['Volume']
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                failed_tickers.append(ticker)
                
        if len(failed_tickers) > 0:
            print(f"Failed to fetch data for: {failed_tickers}")
            
        self.prices = prices
        self.volumes = volumes
        return prices, volumes
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate returns with proper handling of missing data"""
        if self.prices is None:
            raise ValueError("Prices not loaded. Call fetch_data first.")
        return self.prices.pct_change().fillna(0)

class SignalGenerator:
    """Enhanced signal generation with multiple signal types"""
    
    def __init__(self, lookback_periods: dict = None):
        self.lookback_periods = lookback_periods or {
            'momentum': 252,
            'mean_reversion': 20,
            'volume': 20,
            'volatility': 63
        }
    
    def calculate_momentum(self, prices: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Calculate momentum with customizable period"""
        period = period or self.lookback_periods['momentum']
        return prices.pct_change(period).fillna(0)
    
    def calculate_mean_reversion(self, prices: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Calculate mean reversion signal with zscore"""
        period = period or self.lookback_periods['mean_reversion']
        zscore = lambda x: (x - x.rolling(period).mean()) / x.rolling(period).std()
        return -zscore(prices).fillna(0)
    
    def calculate_volatility_signal(self, prices: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Calculate volatility-based signal"""
        period = period or self.lookback_periods['volatility']
        vol = prices.pct_change().rolling(period).std()
        return (1/vol).fillna(0)  # Inverse volatility
    
    def calculate_volume_signal(self, prices: pd.DataFrame, volumes: pd.DataFrame, 
                              period: int = None) -> pd.DataFrame:
        """Calculate volume-price signal"""
        period = period or self.lookback_periods['volume']
        volume_ma = volumes.rolling(period).mean()
        price_vol = prices.pct_change() * (volumes / volume_ma)
        return price_vol.fillna(0)
    
    def combine_signals(self, signals: dict, weights: dict = None) -> pd.DataFrame:
        """Combine multiple signals with optional weighting"""
        if weights is None:
            weights = {key: 1/len(signals) for key in signals.keys()}
            
        combined = pd.DataFrame(0, index=signals[list(signals.keys())[0]].index,
                              columns=signals[list(signals.keys())[0]].columns)
        
        for name, signal in signals.items():
            combined += signal * weights[name]
            
        return combined

class PortfolioOptimizer:
    """Enhanced portfolio optimizer with improved stability"""
    
    def __init__(self, risk_free_rate=0.02, max_position=0.2, target_vol=0.15):
        self.risk_free_rate = risk_free_rate
        self.max_position = max_position
        self.target_vol = target_vol
        
    def calculate_robust_covariance(self, returns: pd.DataFrame, halflife: int = 63) -> np.ndarray:
        """Calculate robust covariance matrix with exponential weighting"""
        weights = np.exp(-np.log(2)/halflife * np.arange(len(returns)))
        weights /= weights.sum()
        
        # Shrinkage to identity matrix for stability
        sample_cov = returns.cov().values
        identity = np.eye(len(sample_cov))
        shrinkage_factor = 0.1
        
        return (1-shrinkage_factor) * sample_cov + shrinkage_factor * identity
    
    def optimize_portfolio(self, returns: pd.DataFrame, signal_scores: pd.Series) -> np.ndarray:
        """Optimize portfolio with improved numerical stability"""
        n_assets = len(returns.columns)
        
        def objective(weights):
            try:
                # Clip weights to bounds
                weights = np.clip(weights, -self.max_position, self.max_position)
                
                # Normalize to sum to 1
                weights = weights / np.sum(np.abs(weights))
                
                # Calculate portfolio metrics
                port_return = np.sum(returns.mean() * weights) * 252
                cov_matrix = self.calculate_robust_covariance(returns)
                
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                
                if port_vol < 1e-8:
                    return np.inf
                
                # Combine Sharpe ratio with signal scores
                sharpe = (port_return - self.risk_free_rate) / port_vol
                signal_contribution = np.sum(weights * signal_scores)
                
                # Add penalty for excessive volatility
                vol_penalty = max(0, port_vol - self.target_vol) * 10
                
                return -(sharpe + 0.5 * signal_contribution - vol_penalty)
                
            except Exception:
                return np.inf
        
        # Initial portfolio: mix of equal weight and signal-based
        x0 = np.array([1/n_assets] * n_assets)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        bounds = tuple((-self.max_position, self.max_position) for _ in range(n_assets))
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-08,
                    }
                )
                
                if result.success:
                    weights = result.x
                else:
                    weights = x0
                    
                # Final cleanup of weights
                weights = np.clip(weights, -self.max_position, self.max_position)
                weights = weights / np.sum(np.abs(weights))
                
                return weights
                
            except Exception as e:
                print(f"Optimization failed: {str(e)}")
                return x0

def run_backtest(save_path: str = None):
    """Run backtest with comprehensive error handling and Bokeh visualization"""
    
    # Initialize components
    market_data = MarketData(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'JPM', 'BAC', 'GS', 'V', 'MA'],
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    signal_generator = SignalGenerator()
    optimizer = PortfolioOptimizer()
    
    try:
        # Fetch and prepare data
        print("Fetching market data...")
        prices, volumes = market_data.fetch_data()
        returns = market_data.calculate_returns()
        
        # Generate signals
        print("Generating trading signals...")
        signals = {
            'momentum': signal_generator.calculate_momentum(prices),
            'mean_reversion': signal_generator.calculate_mean_reversion(prices),
            'volatility': signal_generator.calculate_volatility_signal(prices),
            'volume': signal_generator.calculate_volume_signal(prices, volumes)
        }
        
        # Combine signals
        signal_weights = {
            'momentum': 0.4,
            'mean_reversion': 0.3,
            'volatility': 0.2,
            'volume': 0.1
        }
        
        combined_signal = signal_generator.combine_signals(signals, signal_weights)
        
        # Run optimization
        portfolio_weights = pd.DataFrame(index=prices.index, columns=prices.columns)
        lookback = 252
        
        print("\nRunning backtest...")
        total_days = len(prices.index[lookback:])
        
        for i, date in enumerate(prices.index[lookback:], 1):
            if i % 50 == 0:
                print(f"Processing day {i}/{total_days} ({(i/total_days)*100:.1f}%)")
                
            try:
                hist_returns = returns.loc[date - pd.DateOffset(years=1):date]
                signal_scores = combined_signal.loc[date]
                
                weights = optimizer.optimize_portfolio(hist_returns, signal_scores)
                portfolio_weights.loc[date] = weights
                
            except Exception as e:
                print(f"\nError on {date}: {e}")
                portfolio_weights.loc[date] = portfolio_weights.iloc[-1] if i > 1 else np.array([1/len(prices.columns)] * len(prices.columns))
        
        # Calculate performance
        strategy_returns = (portfolio_weights.shift() * returns).sum(axis=1)
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        annual_return = strategy_returns.mean() * 252
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - optimizer.risk_free_rate) / annual_vol
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        print("\nPerformance Metrics:")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Annual Volatility: {annual_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        
        # Create and save interactive visualization
        if save_path:
            bokeh_save_path = save_path.replace('.png', '.html')
            print(f"\nCreating interactive visualization...")
            create_interactive_plots(
                portfolio_weights=portfolio_weights,
                strategy_returns=strategy_returns,
                save_path=bokeh_save_path
            )
            print(f"Interactive visualization saved to: {bokeh_save_path}")
        
        return {
            'weights': portfolio_weights,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'metrics': {
                'annual_return': annual_return,
                'annual_vol': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
        }
        
    except Exception as e:
        print(f"Critical error: {e}")
        raise

if __name__ == "__main__":
    # Use .html extension for save_path since we're using Bokeh
    save_path = 'outputs/backtest_results.html'
    
    try:
        results = run_backtest(save_path=save_path)
    except Exception as e:
        print(f"Fatal error: {e}")