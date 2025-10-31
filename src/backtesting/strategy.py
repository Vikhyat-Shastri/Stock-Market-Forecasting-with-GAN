"""
Comprehensive backtesting framework for stock trading strategies.
Includes walk-forward validation, transaction costs, risk management, and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    ticker: str
    side: PositionSide
    entry_price: float
    quantity: int
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def close(
        self,
        exit_price: float,
        exit_timestamp: datetime,
        commission: float = 0.0,
        slippage: float = 0.0
    ):
        """Close the trade and calculate P&L."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.commission += commission
        self.slippage += slippage
        
        # Calculate P&L
        if self.side == PositionSide.LONG:
            gross_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            gross_pnl = (self.entry_price - exit_price) * self.quantity
        
        self.pnl = gross_pnl - self.commission - self.slippage
        self.pnl_pct = self.pnl / (self.entry_price * self.quantity)
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_price is None


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005  # 0.05% slippage
    max_position_size: float = 0.2  # 20% of portfolio per position
    stop_loss_pct: Optional[float] = 0.05  # 5% stop loss
    take_profit_pct: Optional[float] = 0.15  # 15% take profit
    max_drawdown_limit: Optional[float] = 0.25  # Stop trading at 25% drawdown
    allow_short: bool = False
    risk_free_rate: float = 0.02  # 2% annual risk-free rate


class PositionSizer:
    """Handles position sizing strategies."""
    
    @staticmethod
    def fixed_fractional(
        capital: float,
        max_risk_pct: float = 0.02
    ) -> float:
        """
        Fixed fractional position sizing.
        
        Args:
            capital: Available capital
            max_risk_pct: Maximum risk per trade (e.g., 0.02 = 2%)
        
        Returns:
            Position size in dollars
        """
        return capital * max_risk_pct
    
    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
        fraction: float = 0.5
    ) -> float:
        """
        Kelly Criterion position sizing.
        
        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade size
            avg_loss: Average losing trade size
            capital: Available capital
            fraction: Kelly fraction (e.g., 0.5 for half Kelly)
        
        Returns:
            Position size in dollars
        """
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        kelly_pct = max(0, min(kelly_pct, 1))  # Clamp between 0 and 1
        
        return capital * kelly_pct * fraction
    
    @staticmethod
    def volatility_adjusted(
        capital: float,
        volatility: float,
        target_volatility: float = 0.15
    ) -> float:
        """
        Volatility-adjusted position sizing.
        
        Args:
            capital: Available capital
            volatility: Current volatility
            target_volatility: Target portfolio volatility
        
        Returns:
            Position size in dollars
        """
        if volatility == 0:
            return 0.0
        
        size = capital * (target_volatility / volatility)
        return max(0, size)


class RiskManager:
    """Manages risk controls and position limits."""
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize risk manager.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.peak_equity = config.initial_capital
        self.current_drawdown = 0.0
        
    def check_stop_loss(self, trade: Trade, current_price: float) -> bool:
        """
        Check if stop loss is triggered.
        
        Args:
            trade: Open trade
            current_price: Current market price
        
        Returns:
            True if stop loss triggered
        """
        if self.config.stop_loss_pct is None:
            return False
        
        if trade.side == PositionSide.LONG:
            loss_pct = (current_price - trade.entry_price) / trade.entry_price
            return loss_pct <= -self.config.stop_loss_pct
        else:  # SHORT
            loss_pct = (trade.entry_price - current_price) / trade.entry_price
            return loss_pct <= -self.config.stop_loss_pct
    
    def check_take_profit(self, trade: Trade, current_price: float) -> bool:
        """
        Check if take profit is triggered.
        
        Args:
            trade: Open trade
            current_price: Current market price
        
        Returns:
            True if take profit triggered
        """
        if self.config.take_profit_pct is None:
            return False
        
        if trade.side == PositionSide.LONG:
            profit_pct = (current_price - trade.entry_price) / trade.entry_price
            return profit_pct >= self.config.take_profit_pct
        else:  # SHORT
            profit_pct = (trade.entry_price - current_price) / trade.entry_price
            return profit_pct >= self.config.take_profit_pct
    
    def update_drawdown(self, current_equity: float):
        """
        Update peak equity and drawdown.
        
        Args:
            current_equity: Current portfolio equity
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
    
    def can_trade(self) -> bool:
        """
        Check if trading is allowed based on drawdown limits.
        
        Returns:
            True if trading is allowed
        """
        if self.config.max_drawdown_limit is None:
            return True
        
        return self.current_drawdown < self.config.max_drawdown_limit


class Backtester:
    """
    Main backtesting engine.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.risk_manager = RiskManager(config)
        self.position_sizer = PositionSizer()
        
        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.equity_curve = []
        
        logger.info(f"Initialized Backtester with ${config.initial_capital:,.2f}")
    
    def run_backtest(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        position_sizing: str = 'fixed_fractional'
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            price_data: DataFrame with OHLCV data
            signals: DataFrame with trading signals (-1, 0, 1)
            position_sizing: Position sizing method
        
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest...")
        
        # Ensure data is aligned
        common_index = price_data.index.intersection(signals.index)
        price_data = price_data.loc[common_index]
        signals = signals.loc[common_index]
        
        # Iterate through data
        for timestamp, signal_row in signals.iterrows():
            prices = price_data.loc[timestamp]
            
            for ticker in signal_row.index:
                if ticker not in prices.index:
                    continue
                
                signal = signal_row[ticker]
                current_price = prices[ticker]
                
                # Check existing position
                if ticker in self.positions:
                    self._manage_position(ticker, current_price, timestamp)
                
                # Process new signals
                if signal != 0 and self.risk_manager.can_trade():
                    self._execute_signal(ticker, signal, current_price, timestamp, position_sizing)
            
            # Update equity curve
            equity = self._calculate_equity(prices)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': self.cash
            })
            
            self.risk_manager.update_drawdown(equity)
        
        # Close any remaining positions
        self._close_all_positions(price_data.iloc[-1], price_data.index[-1])
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        
        logger.info("Backtest completed")
        return results
    
    def _execute_signal(
        self,
        ticker: str,
        signal: float,
        price: float,
        timestamp: datetime,
        position_sizing: str
    ):
        """Execute trading signal."""
        # Determine position side
        if signal > 0:
            side = PositionSide.LONG
        elif signal < 0:
            if not self.config.allow_short:
                return
            side = PositionSide.SHORT
        else:
            return
        
        # Calculate position size
        if position_sizing == 'fixed_fractional':
            position_value = self.position_sizer.fixed_fractional(
                self.cash,
                self.config.max_position_size
            )
        else:
            position_value = self.cash * self.config.max_position_size
        
        # Calculate quantity
        quantity = int(position_value / price)
        if quantity == 0:
            return
        
        # Calculate costs
        trade_value = quantity * price
        commission = trade_value * self.config.commission_rate
        slippage = trade_value * self.config.slippage_rate
        total_cost = trade_value + commission + slippage
        
        # Check if we have enough cash
        if total_cost > self.cash:
            return
        
        # Create trade
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            side=side,
            entry_price=price,
            quantity=quantity,
            commission=commission,
            slippage=slippage
        )
        
        # Update portfolio
        self.positions[ticker] = trade
        self.cash -= total_cost
        
        logger.debug(f"Opened {side.value} position: {ticker} x{quantity} @ ${price:.2f}")
    
    def _manage_position(self, ticker: str, current_price: float, timestamp: datetime):
        """Manage existing position (stop loss, take profit)."""
        trade = self.positions[ticker]
        
        should_close = False
        
        # Check stop loss
        if self.risk_manager.check_stop_loss(trade, current_price):
            logger.debug(f"Stop loss triggered for {ticker}")
            should_close = True
        
        # Check take profit
        elif self.risk_manager.check_take_profit(trade, current_price):
            logger.debug(f"Take profit triggered for {ticker}")
            should_close = True
        
        if should_close:
            self._close_position(ticker, current_price, timestamp)
    
    def _close_position(self, ticker: str, exit_price: float, timestamp: datetime):
        """Close an open position."""
        if ticker not in self.positions:
            return
        
        trade = self.positions.pop(ticker)
        
        # Calculate exit costs
        trade_value = trade.quantity * exit_price
        commission = trade_value * self.config.commission_rate
        slippage = trade_value * self.config.slippage_rate
        
        # Close trade
        trade.close(exit_price, timestamp, commission, slippage)
        
        # Update cash
        if trade.side == PositionSide.LONG:
            self.cash += trade_value - commission - slippage
        else:  # SHORT
            self.cash += (2 * trade.entry_price * trade.quantity) - trade_value - commission - slippage
        
        # Record trade
        self.closed_trades.append(trade)
        
        logger.debug(f"Closed {trade.side.value} position: {ticker} P&L: ${trade.pnl:.2f}")
    
    def _close_all_positions(self, prices: pd.Series, timestamp: datetime):
        """Close all open positions at end of backtest."""
        for ticker in list(self.positions.keys()):
            if ticker in prices.index:
                self._close_position(ticker, prices[ticker], timestamp)
    
    def _calculate_equity(self, prices: pd.Series) -> float:
        """Calculate total portfolio equity."""
        equity = self.cash
        
        for ticker, trade in self.positions.items():
            if ticker in prices.index:
                current_price = prices[ticker]
                if trade.side == PositionSide.LONG:
                    equity += trade.quantity * current_price
                else:  # SHORT
                    equity += 2 * trade.entry_price * trade.quantity - trade.quantity * current_price
        
        return equity
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        equity_df = pd.DataFrame(self.equity_curve).set_index('timestamp')
        
        # Basic metrics
        initial_equity = self.config.initial_capital
        final_equity = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial_equity
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Trade statistics
        trades_df = pd.DataFrame([
            {
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'duration': (t.exit_timestamp - t.timestamp).days if t.exit_timestamp else 0
            }
            for t in self.closed_trades
        ])
        
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df)
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf
            avg_trade_duration = trades_df['duration'].mean()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = 0
        
        # Risk metrics
        if len(equity_df) > 0:
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_equity': final_equity,
            'total_trades': len(self.closed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'avg_trade_duration_days': avg_trade_duration,
            'equity_curve': equity_df
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (self.config.risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.config.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity) == 0:
            return 0.0
        
        cumulative_max = equity.expanding().max()
        drawdown = (equity - cumulative_max) / cumulative_max
        return drawdown.min()


def walk_forward_analysis(
    price_data: pd.DataFrame,
    signals_generator: Callable,
    train_window: int = 252,
    test_window: int = 63,
    config: Optional[BacktestConfig] = None
) -> List[Dict]:
    """
    Perform walk-forward analysis.
    
    Args:
        price_data: Historical price data
        signals_generator: Function to generate signals from training data
        train_window: Training window size (days)
        test_window: Testing window size (days)
        config: Backtesting configuration
    
    Returns:
        List of results for each walk-forward window
    """
    if config is None:
        config = BacktestConfig()
    
    results = []
    start = 0
    
    while start + train_window + test_window <= len(price_data):
        # Split data
        train_end = start + train_window
        test_end = train_end + test_window
        
        train_data = price_data.iloc[start:train_end]
        test_data = price_data.iloc[train_end:test_end]
        
        # Generate signals on training data
        signals = signals_generator(train_data, test_data)
        
        # Run backtest on test data
        backtester = Backtester(config)
        result = backtester.run_backtest(test_data, signals)
        
        result['train_start'] = train_data.index[0]
        result['train_end'] = train_data.index[-1]
        result['test_start'] = test_data.index[0]
        result['test_end'] = test_data.index[-1]
        
        results.append(result)
        
        # Move window
        start += test_window
    
    logger.info(f"Completed walk-forward analysis: {len(results)} windows")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_days = len(dates)
    
    # Simulate price data
    prices = pd.DataFrame({
        'AAPL': 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_days))),
        'MSFT': 150 * np.exp(np.cumsum(np.random.normal(0.0008, 0.018, n_days)))
    }, index=dates)
    
    # Simple momentum signals
    signals = pd.DataFrame(index=dates, columns=['AAPL', 'MSFT'])
    signals['AAPL'] = np.where(prices['AAPL'].pct_change(20) > 0.05, 1, 0)
    signals['MSFT'] = np.where(prices['MSFT'].pct_change(20) > 0.05, 1, 0)
    
    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        stop_loss_pct=0.05,
        take_profit_pct=0.15
    )
    
    backtester = Backtester(config)
    results = backtester.run_backtest(prices, signals)
    
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
