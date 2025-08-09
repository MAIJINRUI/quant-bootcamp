import numpy as np
import pandas as pd
from bootcamp.week01.assignment.solution import (
    log_returns, simple_returns, cumulative_returns, annualize_stats, sharpe_ratio, max_drawdown
)

def test_log_returns_shape():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    px = pd.DataFrame({"A":[100, 102, 101, 103, 106]}, index=idx)
    r = log_returns(px)
    assert r.shape == (4,1)
    assert np.isclose(r.iloc[0,0], np.log(102/100))

def test_cum_returns_equivalence():
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    r = pd.DataFrame({"A":[0.01, -0.02, 0.03]}, index=idx[1:])
    cum_simple = cumulative_returns(r["A"], log=False)
    cum_log = cumulative_returns(np.log(1+r["A"]), log=True)
    assert np.allclose(cum_simple.values, cum_log.values, atol=1e-12)

def test_max_drawdown():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    eq = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3], index=idx)
    mdd = max_drawdown(eq)
    assert np.isclose(mdd, (0.9/1.2 - 1.0))

def test_sharpe_sign():
    np.random.seed(0)
    idx = pd.date_range("2022-01-01", periods=252, freq="B")
    r = pd.Series(0.0004 + 0.01/np.sqrt(252)*np.random.randn(252), index=idx)
    s = sharpe_ratio(r)
    assert s > 0
