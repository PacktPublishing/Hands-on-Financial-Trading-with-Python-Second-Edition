"""
strategy/signal.py

The VIXY tail hedge signal and weight calculation. calculate_vixy_weights()
is carried over unchanged from Notebook 9.1's canonical version, the same
function used throughout the book for evaluation, backtesting, and
out-of-sample testing. get_today_snapshot() is a thin wrapper added for
this repository: it calls the canonical function for the weights, then
separately reports the signal inputs (VIX level, VIX moving average,
realized volatility) for the log and the Telegram message.
"""
import numpy as np
import pandas as pd


def calculate_vixy_weights(df_prices,
                            sma_period=90,
                            std_period=10,
                            risky_portion=0.80,
                            vixy_cap=0.20):
    """
    Calculate daily target weights for the VIXY tail hedge strategy.

    The strategy holds a constant SPY allocation. The remaining defensive
    sleeve sits in BIL by default and shifts into VIXY when SPY's 10-day
    annualized realized volatility exceeds the VIX simple moving average.
    The VIXY allocation is sized by VIX/100, capped at vixy_cap, with the
    BIL allocation absorbing the remainder.

    Parameters
    ----------
    df_prices : DataFrame
        Daily adjusted close prices with columns SPY, BIL, VIXY, ^VIX.
    sma_period : int
        Lookback window for the VIX simple moving average. Default 90.
    std_period : int
        Lookback window for SPY realized volatility. Default 10.
    risky_portion : float
        Constant SPY allocation. Default 0.80.
    vixy_cap : float
        Maximum VIXY allocation when the hedge signal is on. Default 0.20.

    Returns
    -------
    DataFrame with columns [SPY, BIL, VIXY] containing daily target
    weights, trimmed to the first date where both lookbacks are valid.
    """
    vix = df_prices['^VIX'].copy()
    vix_sma = vix.rolling(window=sma_period).mean()
    spy_returns = df_prices['SPY'].pct_change()
    spy_realized_vol = (spy_returns.rolling(window=std_period).std()
                         * np.sqrt(252) * 100)

    signal_on = spy_realized_vol > vix_sma
    vixy_alloc = np.where(signal_on, np.minimum(vix / 100, vixy_cap), 0.0)

    start_date = max(
        vix_sma.first_valid_index(),
        spy_realized_vol.first_valid_index()
    )

    df_weights = pd.DataFrame(index=df_prices.index)
    df_weights['SPY'] = risky_portion
    df_weights['BIL'] = (1.0 - risky_portion) - vixy_alloc
    df_weights['VIXY'] = vixy_alloc
    df_weights = df_weights.loc[start_date:]

    return df_weights


def get_today_snapshot(df_prices,
                        sma_period=90,
                        std_period=10,
                        risky_portion=0.80,
                        vixy_cap=0.20):
    """
    Build the full picture for the most recent trading day: today's
    target weights plus the underlying signal inputs, for the log and
    the Telegram message.

    Parameters
    ----------
    df_prices : DataFrame
        Output of strategy.data.get_price_data().
    sma_period, std_period, risky_portion, vixy_cap
        Passed straight through to calculate_vixy_weights().

    Returns
    -------
    dict with the most recent date, the signal state, the signal inputs,
    and today's target weights.
    """
    df_weights = calculate_vixy_weights(
        df_prices, sma_period, std_period, risky_portion, vixy_cap
    )
    latest_date = df_weights.index[-1]

    vix_level = df_prices['^VIX'].loc[latest_date]
    vix_sma = df_prices['^VIX'].rolling(window=sma_period).mean().loc[latest_date]
    spy_returns = df_prices['SPY'].pct_change()
    spy_realized_vol = (spy_returns.rolling(window=std_period).std()
                         * np.sqrt(252) * 100).loc[latest_date]
    signal_on = bool(spy_realized_vol > vix_sma)

    weights_today = df_weights.loc[latest_date]

    return {
        'date': latest_date,
        'signal_on': signal_on,
        'vix_level': round(float(vix_level), 2),
        'vix_sma_90': round(float(vix_sma), 2),
        'spy_realized_vol': round(float(spy_realized_vol), 2),
        'spy_weight': round(float(weights_today['SPY']), 4),
        'bil_weight': round(float(weights_today['BIL']), 4),
        'vixy_weight': round(float(weights_today['VIXY']), 4),
    }
