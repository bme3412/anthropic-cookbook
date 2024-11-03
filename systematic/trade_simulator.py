import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import random

@dataclass
class TradeStats:
    """Container for trade statistics"""
    entry_date: str
    exit_date: str
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    holding_period: int
    transaction_cost: float

class TradeSimulator:
    def __init__(self, 
                 initial_capital=random.uniform(1_000_000, 10_000_000),
                 commission_rate=random.uniform(0.0001, 0.0005),  # 5bps per trade
                 slippage_rate=random.uniform(0.0001, 0.0003),    # 2bps per trade
                 market_impact=random.uniform(0.005, 0.01)):   # 1% price impact per $1M traded
        
        # Validate parameters
        assert initial_capital > 0, "Initial capital must be positive"
        assert 0 <= commission_rate <= 0.01, "Commission rate must be between 0 and 1%"
        assert 0 <= slippage_rate <= 0.01, "Slippage rate must be between 0 and 1%"
        assert 0 <= market_impact <= 0.1, "Market impact must be between 0 and 10%"
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.market_impact = market_impact
        
        self.positions = {}
        self.trades = []
        self.daily_pnl = []
        self.transaction_costs = []

    def simulate_trades(self, portfolio_weights: pd.DataFrame,
                       prices: pd.DataFrame, 
                       volumes: pd.DataFrame) -> Dict:
        """Simulate trades with improved transaction cost modeling"""
        current_positions = pd.Series(0.0, index=portfolio_weights.columns)
        daily_holdings = []
        trade_log = []
        
        print("\nSimulating trades...")
        for date in portfolio_weights.index:
            try:
                # Get current prices and target weights
                current_prices = prices.loc[date]
                target_weights = portfolio_weights.loc[date]
                daily_volume = volumes.loc[date]
                
                # Calculate target positions
                portfolio_value = self.current_capital
                target_value = target_weights * portfolio_value
                target_positions = target_value / current_prices
                
                # Calculate required trades
                trades_needed = target_positions - current_positions
                
                # Execute trades with transaction costs
                total_cost = 0
                for symbol in trades_needed.index:
                    trade_size = trades_needed[symbol]
                    if abs(trade_size) > 0.0001:  # Minimum trade size
                        # Calculate trade value
                        trade_value = abs(trade_size * current_prices[symbol])
                        
                        # Commission cost
                        commission = trade_value * self.commission_rate
                        
                        # Slippage based on volume participation
                        daily_symbol_volume = daily_volume[symbol] * current_prices[symbol]
                        participation_rate = min(trade_value / daily_symbol_volume, 0.1)  # Cap at 10%
                        slippage = trade_value * self.slippage_rate * (1 + participation_rate)
                        
                        # Market impact - scaled by participation and capped
                        impact_factor = min(trade_value / 1_000_000, 1.0)  # Scale by trade size
                        impact = trade_value * self.market_impact * impact_factor * participation_rate
                        
                        total_cost += commission + slippage + impact
                        
                        # Record trade details
                        trade_log.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY' if trade_size > 0 else 'SELL',
                            'quantity': abs(trade_size),
                            'price': current_prices[symbol],
                            'value': trade_value,
                            'commission': commission,
                            'slippage': slippage,
                            'market_impact': impact,
                            'total_cost': commission + slippage + impact,
                            'volume_participation': participation_rate
                        })
                
                # Update positions
                current_positions = target_positions
                
                # Calculate daily P&L
                position_value = (current_positions * current_prices).sum()
                daily_pnl = position_value - portfolio_value - total_cost
                self.current_capital += daily_pnl
                
                # Record daily stats
                self.daily_pnl.append({
                    'date': date,
                    'pnl': daily_pnl,
                    'portfolio_value': self.current_capital,
                    'transaction_cost': total_cost,
                    'cash': self.current_capital - position_value
                })
                
                # Record positions
                daily_holdings.append({
                    'date': date,
                    **{symbol: pos for symbol, pos in current_positions.items()}
                })
                
            except Exception as e:
                print(f"Error simulating trades for {date}: {e}")
                continue
        
        # Convert records to DataFrames
        self.holdings_df = pd.DataFrame(daily_holdings).set_index('date')
        self.pnl_df = pd.DataFrame(self.daily_pnl).set_index('date')
        self.trade_log_df = pd.DataFrame(trade_log)
        
        # Calculate performance metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        total_costs = self.pnl_df['transaction_cost'].sum()
        turnover = self.calculate_turnover(portfolio_weights)
        
        return {
            'final_capital': self.current_capital,
            'total_return': total_return,
            'total_costs': total_costs,
            'annual_turnover': turnover,
            'holdings': self.holdings_df,
            'pnl': self.pnl_df,
            'trade_log': self.trade_log_df
        }
    
    def calculate_turnover(self, weights: pd.DataFrame) -> float:
        """Calculate annual turnover rate"""
        daily_turnover = np.abs(weights.diff()).sum(axis=1).fillna(0)
        annual_turnover = daily_turnover.sum() * 252 / len(daily_turnover)
        return annual_turnover
    
    def save_trade_log(self, filepath: str):
        """Save trade log to CSV"""
        if hasattr(self, 'trade_log_df'):
            self.trade_log_df.to_csv(filepath)
            print(f"Trade log saved to: {filepath}")
        else:
            print("No trades to save. Run simulate_trades first.")
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of trades by symbol"""
        if not hasattr(self, 'trade_log_df') or len(self.trade_log_df) == 0:
            return pd.DataFrame()
            
        summary = self.trade_log_df.groupby('symbol').agg({
            'value': 'sum',
            'commission': 'sum',
            'slippage': 'sum',
            'market_impact': 'sum',
            'total_cost': 'sum',
            'volume_participation': 'mean',
            'symbol': 'count'
        }).rename(columns={'symbol': 'number_of_trades'})
        
        summary['average_trade_size'] = summary['value'] / summary['number_of_trades']
        summary['cost_per_trade'] = summary['total_cost'] / summary['number_of_trades']
        
        return summary
    

