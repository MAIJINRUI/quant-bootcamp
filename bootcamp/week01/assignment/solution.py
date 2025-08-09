from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

TRADING_DAYS = 252

def load_prices_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    rets = np.log(prices / prices.shift(1))
    return rets.dropna(how="all")

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    rets = prices.pct_change()
    return rets.dropna(how="all")

def cumulative_returns(returns: pd.Series | pd.DataFrame, log: bool = False) -> pd.Series | pd.DataFrame:
    if log:
        return (returns.fillna(0).cumsum()).apply(np.exp)
    return (1 + returns.fillna(0)).cumprod()

def annualize_stats(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> tuple[pd.Series, pd.Series]:
    mu = returns.mean() * periods
    vol = returns.std() * np.sqrt(periods)
    return mu, vol

def sharpe_ratio(returns: pd.DataFrame | pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> pd.Series | float:
    if isinstance(returns, pd.Series):
        mu = returns.mean() * periods - rf
        vol = returns.std() * np.sqrt(periods)
        return float(mu / vol) if vol != 0 else np.nan
    mu = returns.mean() * periods - rf
    vol = returns.std() * np.sqrt(periods)
    return mu / vol.replace(0, np.nan)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def rolling_vol(returns: pd.DataFrame, window: int = 30, periods: int = TRADING_DAYS) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(periods)

def save_csv(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)