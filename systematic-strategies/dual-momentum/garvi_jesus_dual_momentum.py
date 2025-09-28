# dual_momentum.py
# Author: Jesús Garví
# Date: 09/25
#
# Description:
# This script is a standalone command-line backtester for a systematic Dual Momentum strategy.
# It downloads historical price data for a given universe of ETFs, calculates momentum signals,
# constructs a portfolio, and evaluates its performance with key metrics.
#
# The strategy combines:
#   1. Relative Momentum: Ranks assets based on a blended short-term and long-term return.
#   2. Absolute Momentum: A risk-management filter that moves the portfolio to cash
#      if the top-ranked asset has a negative long-term return.
#
# The entire process is designed to be reproducible and auditable.
#
# To Run from Command Line (Example):
# python dual_momentum.py --tickers SPY QQQ EFA EEM --start 2003-01-01 \
#                         --export_metrics output/metrics.csv \
#                         --export_signals output/signals.csv \
#                         --equity_png output/equity_curve.png

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Optional

# Optional plotting only if the user requests a PNG output and has matplotlib installed.
# This makes the core logic runnable on servers without a display environment.
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# --- Configuration ---

@dataclass
class Config:
    """
    A dataclass to hold all strategy parameters and execution settings.
    This provides a clean, type-hinted way to manage configuration.
    """
    tickers: List[str]
    start: str
    end: Optional[str]
    top_k: int
    lb_long_m: int
    lb_short_m: int
    cost_bps: float
    slippage_bps: float
    export_signals: Optional[str]
    export_metrics: Optional[str]
    equity_png: Optional[str]

# --- Core Functions ---

# Reasonable defaults so the script can run without CLI args
DEFAULT_TICKERS = ["SPY", "QQQ", "EFA", "EEM"]
DEFAULT_START = "2003-01-01"

def download_prices(tickers: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Downloads adjusted closing prices from Yahoo Finance for a list of tickers.

    Args:
        tickers: A list of ticker symbols.
        start: The start date in 'YYYY-MM-DD' format.
        end: The optional end date in 'YYYY-MM-DD' format.

    Returns:
        A pandas DataFrame with dates as the index and closing prices for each ticker.
        Handles both single and multiple ticker downloads and performs basic data cleaning.
    """
    # Use yfinance to download data. auto_adjust=True provides split/dividend-adjusted prices.
    # progress=False is used for cleaner script execution (no download status bar).
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns a DataFrame with multi-level columns if multiple tickers are requested,
    # or a Series if only one is. We standardize to a DataFrame with just the 'Close' prices.
    if isinstance(px, pd.DataFrame) and 'Close' in px.columns:
        px = px['Close']
    if isinstance(px, pd.Series):
        px = px.to_frame()

    # Critical data integrity check: if all data is NaN, the download likely failed.
    if px.isna().all().all():
        raise RuntimeError("Price download failed; received all-NaN data.")
        
    # Drop any columns or rows that are entirely NaN.
    return px.dropna(how='all').dropna(axis=1, how='all')

def build_weights(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Constructs the daily portfolio weights based on the Dual Momentum strategy logic.

    Args:
        prices: DataFrame of daily closing prices.
        cfg: The configuration dataclass.

    Returns:
        A DataFrame of daily target weights for each asset.
    """
    # 1. Resample to month-end ('ME') to get closing prices for signal calculation.
    # This ensures we only rebalance on information available at the end of the month.
    pm = prices.resample('ME').last()

    # 2. Calculate momentum components.
    # IMPORTANT: fill_method=None is crucial to prevent pandas from implicitly
    # forward-filling NaNs, which would introduce lookahead bias.
    ret_long = pm.pct_change(cfg.lb_long_m, fill_method=None)
    ret_short = pm.pct_change(cfg.lb_short_m, fill_method=None)
    
    # The Dual Momentum score is an equal-weighted average of long and short-term returns.
    score = 0.5 * ret_long + 0.5 * ret_short

    # Determine rebalancing dates. We can only start after the longest lookback period.
    rebal_dates = pm.index[cfg.lb_long_m:].copy()
    weights_m = pd.DataFrame(0.0, index=rebal_dates, columns=pm.columns)

    # 3. Iterate through rebalancing dates to determine monthly target weights.
    for dt in rebal_dates:
        s = score.loc[dt].dropna()
        rl = ret_long.loc[dt].dropna()

        # 4. Absolute Momentum Filter (Risk Management):
        # Only consider assets with positive long-term momentum (return > 0).
        # This acts as a filter against a risk-free asset (assumed return of 0).
        eligible = rl[rl > 0.0].index.intersection(s.index)
        
        if len(eligible) == 0:
            # If no assets have positive momentum, allocate 100% to CASH.
            # The weights for tickers remain 0.0.
            continue

        # 5. Relative Momentum Filter:
        # Of the eligible assets, rank them by their momentum score in descending order.
        s_eligible = s.loc[eligible].sort_values(ascending=False)
        
        # Select the top K winners.
        winners = s_eligible.index[:cfg.top_k]
        
        # Allocate weights equally among the winners.
        w = np.ones(len(winners)) / len(winners)
        weights_m.loc[dt, winners] = w

    # 6. Expand monthly weights to a daily frequency.
    # Forward-fill carries the month-end decision forward until the next rebalance.
    w_daily = weights_m.reindex(prices.index).ffill().fillna(0.0)

    # Before the first rebalance, the portfolio should hold no assets.
    first_rebal = weights_m.index.min() if not weights_m.empty else None
    if first_rebal is not None:
        w_daily.loc[w_daily.index < first_rebal, :] = 0.0
        
    return w_daily

def portfolio_equity(prices: pd.DataFrame, w_daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Calculates the portfolio equity curve from daily prices and target weights.

    Args:
        prices: DataFrame of daily closing prices.
        w_daily: DataFrame of daily target weights.
        cfg: The configuration dataclass.

    Returns:
        A DataFrame containing the daily equity curve and the chosen asset(s) for each day.
    """
    # Calculate daily returns. Again, disable implicit filling and clean inf/-inf values.
    ret = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret = ret.reindex(w_daily.index).fillna(0.0) # Align returns index with weights index.

    # Calculate total per-turn transaction costs (commission + slippage).
    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10000.0

    # CRITICAL: Use the previous day's weights (w_prev) to calculate today's return.
    # This prevents lookahead bias. The portfolio is rebalanced based on signals
    # from t-1, and its performance is measured on returns from t-1 to t.
    w_prev = w_daily.shift(1).fillna(0.0)

    # L1 Turnover is the sum of absolute changes in weights.
    turnover = (w_daily - w_prev).abs().sum(axis=1)
    
    # Raw portfolio return before costs.
    port_ret_raw = (w_prev * ret).sum(axis=1)
    
    # Net portfolio return after applying transaction costs.
    port_ret = port_ret_raw - (turnover * cost_rate)

    # Calculate the cumulative equity curve, starting from a base of 1.0.
    equity = (1.0 + port_ret).cumprod()
    equity.name = 'equity'

    # Create a column to show the chosen asset(s) or 'CASH' for each day.
    chosen = []
    for i in range(len(w_daily)):
        row = w_daily.iloc[i]
        if row.sum() <= 1e-12: # Check if all weights are effectively zero.
            chosen.append('CASH')
        else:
            # Join the names of tickers with non-zero weights.
            chosen.append('+'.join(row[row > 0].index.tolist()))
            
    # Combine the equity curve and chosen assets into a final signals DataFrame.
    out = pd.concat([equity, pd.Series(chosen, index=w_daily.index, name='chosen')], axis=1)
    return out

# --- Performance Metrics ---

def max_drawdown(equity: pd.Series) -> float:
    """Calculates the maximum drawdown of an equity curve."""
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())

def annualize_return(equity: pd.Series) -> float:
    """Calculates the Compound Annual Growth Rate (CAGR)."""
    if len(equity) < 2:
        return 0.0
    
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    
    if years <= 0:
        return 0.0 # Avoid division by zero or negative exponent.
        
    return total_return ** (1.0 / years) - 1.0

def sharpe_daily(daily_ret: pd.Series, rf: float = 0.0) -> float:
    """Calculates the annualized Sharpe Ratio from daily returns."""
    mu = daily_ret.mean() - (rf / 252.0)
    sd = daily_ret.std(ddof=0)
    
    if sd == 0:
        return 0.0
        
    return float(np.sqrt(252.0) * mu / sd)

def winrate_monthly(equity: pd.Series) -> float:
    """Calculates the percentage of months with positive returns."""
    # Resample to month-end and calculate monthly returns.
    em = equity.resample('ME').last().pct_change(fill_method=None).dropna()
    
    if len(em) == 0:
        return 0.0
        
    return float((em > 0).mean())

# --- Main Execution Logic ---

def run(cfg: Config):
    """
    Main function to run the entire backtesting process.
    Orchestrates data download, weight calculation, equity generation,
    metrics calculation, and result exporting.
    """
    # 1. Data Retrieval and Preparation
    prices = download_prices(cfg.tickers, cfg.start, cfg.end).sort_index()
    prices = prices.dropna(how='all', axis=1) # Final cleaning of any fully empty asset columns.
    
    # 2. Strategy Execution
    w_daily = build_weights(prices, cfg)
    signals = portfolio_equity(prices, w_daily, cfg)
    
    # 3. Metrics Calculation
    daily_ret = signals['equity'].pct_change(fill_method=None).dropna()
    cagr = annualize_return(signals['equity'])
    sharpe = sharpe_daily(daily_ret)
    mdd = max_drawdown(signals['equity'])
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    wr = winrate_monthly(signals['equity'])

    metrics = pd.DataFrame([{
        'CAGR': cagr,
        'Sharpe': sharpe,
        'MaxDrawdown': mdd,
        'Calmar': calmar,
        'WinRate': wr
    }])
    
    # 4. Exporting Results
    if cfg.export_signals:
        signals.to_csv(cfg.export_signals, index=True)
    
    if cfg.export_metrics:
        metrics.to_csv(cfg.export_metrics, index=False)
        
    # 5. Optional Visualization
    if cfg.equity_png and plt is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(signals.index, signals['equity'], lw=2)
        ax.set_yscale('log') # Log scale is standard for visualizing long-term equity curves.
        ax.set_title('Equity Curve (log scale)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.grid(True, which='both', ls='--', lw=0.5)
        fig.tight_layout()
        fig.savefig(cfg.equity_png, dpi=150)
        plt.close(fig)

    return signals, metrics

# --- Command-Line Interface ---

def parse_args(argv: Optional[List[str]] = None) -> Config:
    """
    Parses command-line arguments to configure the backtest.
    Uses argparse for a robust and self-documenting CLI.
    """
    p = argparse.ArgumentParser(description="Dual Momentum CLI Backtester")
    # Make args optional; we will apply sensible defaults if missing.
    p.add_argument('--tickers', nargs='+', help='List of tickers (Yahoo Finance).')
    p.add_argument('--start', type=str, help='Start date YYYY-MM-DD.')
    p.add_argument('--end', type=str, default=None, help='Optional end date YYYY-MM-DD.')
    p.add_argument('--top_k', type=int, default=1, help='Number of winners to hold, equally weighted.')
    p.add_argument('--lb_long_m', type=int, default=12, help='Long lookback in months (e.g., 12).')
    p.add_argument('--lb_short_m', type=int, default=3, help='Short lookback in months (e.g., 3).')
    p.add_argument('--cost_bps', type=float, default=0.0, help='Per-side commission in basis points.')
    p.add_argument('--slippage_bps', type=float, default=0.0, help='Per-side slippage in basis points.')
    p.add_argument('--export_signals', type=str, default=None, help='CSV path for daily equity/chosen.')
    p.add_argument('--export_metrics', type=str, default=None, help='CSV path for summary metrics.')
    p.add_argument('--equity_png', type=str, default=None, help='PNG path for equity curve.')
    p.add_argument('--demo', action='store_true', help='Run with built-in defaults if tickers/start omitted.')
    p.add_argument('--quiet', action='store_true', help='Suppress console summary output.')
    
    # Use argv for testing, otherwise sys.argv will be used by default.
    args = p.parse_args(argv)

    # Apply defaults when values are missing or demo requested
    tickers = args.tickers if args.tickers else DEFAULT_TICKERS
    start = args.start if args.start else DEFAULT_START

    # Allow strict validation if user explicitly opts out of defaults
    # (not exposed, but keeps behavior clear if needed later)

    if args.tickers is None or args.start is None:
        print(f"[info] Using defaults: tickers={tickers}, start={start}")

    return Config(
        tickers=tickers,
        start=start,
        end=args.end,
        top_k=args.top_k,
        lb_long_m=args.lb_long_m,
        lb_short_m=args.lb_short_m,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        export_signals=args.export_signals,
        export_metrics=args.export_metrics,
        equity_png=args.equity_png
    )

def _fmt_pct(x: Optional[float]) -> str:
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return "-"
        return f"{x:.2%}"
    except Exception:
        return "-"

def print_summary(signals: pd.DataFrame, metrics: pd.DataFrame, cfg: Config) -> None:
    row = metrics.iloc[0]
    start_dt = signals.index[0].date() if not signals.empty else None
    end_dt = signals.index[-1].date() if not signals.empty else None
    last_chosen = signals['chosen'].iloc[-1] if 'chosen' in signals.columns and not signals.empty else '-'
    final_eq = signals['equity'].iloc[-1] if 'equity' in signals.columns and not signals.empty else float('nan')
    calmar_str = "-" if (isinstance(row['Calmar'], float) and np.isnan(row['Calmar'])) else f"{row['Calmar']:.2f}"

    print("\n=== Dual Momentum Summary ===")
    print(f"Period: {start_dt} → {end_dt}")
    print(f"Tickers: {', '.join(cfg.tickers)} | top_k={cfg.top_k}")
    print(f"Final equity: {final_eq:.2f}")
    print(f"CAGR: {_fmt_pct(row['CAGR'])} | Sharpe: {row['Sharpe']:.2f}")
    print(f"MaxDD: {_fmt_pct(row['MaxDrawdown'])} | Calmar: {calmar_str}")
    print(f"Win rate (months): {_fmt_pct(row['WinRate'])}")
    print(f"Latest position: {last_chosen}")
    if cfg.export_metrics:
        print(f"Saved metrics to: {cfg.export_metrics}")
    if cfg.export_signals:
        print(f"Saved signals to: {cfg.export_signals}")

if __name__ == '__main__':
    """
    Entry point for running the script from the command line.
    """
    import sys
    config = parse_args(sys.argv[1:])
    signals, metrics = run(config)
    if not getattr(config, 'quiet', False):
        print_summary(signals, metrics, config)
