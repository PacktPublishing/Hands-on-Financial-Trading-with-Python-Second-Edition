"""
functions.py
------------
Reusable performance evaluation toolkit built in Notebook 7.1.
Hands-On Financial Trading with Python, 2nd Edition.

Drag and drop this file into your Colab session before importing.
"""

import pandas as pd
import numpy as np


def max_drawdown_duration(drawdown_series):
    """
    Calculate the maximum drawdown duration in trading days.

    Parameters
    ----------
    drawdown_series : pd.Series
        The drawdown at each point in time (0 at peaks, negative below peaks)

    Returns
    -------
    int
        Longest number of consecutive trading days spent below a prior peak
    """
    duration = 0
    max_duration = 0

    for dd in drawdown_series:
        if dd < 0:
            # Still underwater: add another day
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            # New peak reached: reset the counter
            duration = 0

    return max_duration


def ulcer_index(drawdown_series):
    """
    Calculate the Ulcer Index from a drawdown series.

    The Ulcer Index measures both the depth and duration of all drawdowns
    by taking the root mean square of the drawdown series. Squaring penalizes
    deeper drawdowns more heavily; averaging across all dates means longer
    drawdowns also contribute more.

    Parameters
    ----------
    drawdown_series : pd.Series
        Drawdown values (0 at peaks, negative below peaks)

    Returns
    -------
    float
        Ulcer Index expressed as a percentage
    """
    # Multiply by 100 to express drawdowns in percentage points
    # before squaring (otherwise the squared values are very small)
    dd_pct = drawdown_series * 100
    return np.sqrt(np.mean(dd_pct ** 2))


def find_drawdown_episodes(cumulative_returns, drawdown_series, top_n=5):
    """
    Identify the top N deepest drawdown episodes.

    A drawdown episode starts when the portfolio moves below its running
    peak and ends when it reaches a new peak (or the series ends).

    Parameters
    ----------
    cumulative_returns : pd.Series
        Growth of $1 series
    drawdown_series : pd.Series
        Drawdown at each point (0 at peaks, negative below peaks)
    top_n : int
        Number of deepest episodes to return

    Returns
    -------
    pd.DataFrame
        Columns: Start, Trough, Recovery, Depth, Duration (days)
    """
    episodes = []
    in_drawdown = False
    start = None

    for i, (date, dd) in enumerate(drawdown_series.items()):
        if dd < 0 and not in_drawdown:
            # Entering a new drawdown episode.
            # The start date is the previous day (the peak).
            in_drawdown = True
            start = drawdown_series.index[i - 1] if i > 0 else date

        elif dd >= 0 and in_drawdown:
            # Recovered to a new peak: close out this episode
            in_drawdown = False
            episode_dd = drawdown_series.loc[start:date]
            trough_date = episode_dd.idxmin()
            trough_depth = episode_dd.min()
            duration = len(episode_dd)
            episodes.append({
                'Start': start.strftime('%Y-%m-%d'),
                'Trough': trough_date.strftime('%Y-%m-%d'),
                'Recovery': date.strftime('%Y-%m-%d'),
                'Depth': f"{trough_depth:.2%}",
                'Duration (days)': duration
            })

    # Handle case where the series ends still in drawdown
    if in_drawdown:
        episode_dd = drawdown_series.loc[start:]
        trough_date = episode_dd.idxmin()
        trough_depth = episode_dd.min()
        duration = len(episode_dd)
        episodes.append({
            'Start': start.strftime('%Y-%m-%d'),
            'Trough': trough_date.strftime('%Y-%m-%d'),
            'Recovery': 'Ongoing',
            'Depth': f"{trough_depth:.2%}",
            'Duration (days)': duration
        })

    # Sort by depth (most negative first) and return top N
    df_episodes = pd.DataFrame(episodes)
    df_episodes['_sort'] = df_episodes['Depth'].str.rstrip('%').astype(float)
    df_episodes = df_episodes.sort_values('_sort').drop(columns='_sort').head(top_n)
    df_episodes.index = range(1, len(df_episodes) + 1)
    df_episodes.index.name = 'Rank'

    return df_episodes


def calculate_performance_metrics(returns, benchmark_returns=None,
                                  risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate a complete set of performance metrics for a return series.

    Consolidates every metric built in Notebook 7.1 into a single reusable
    call. Accepts daily returns and produces annualized metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns
    benchmark_returns : pd.Series, optional
        Daily benchmark returns. If provided, calculates the information
        ratio and excess return. Must be aligned to the same dates.
    risk_free_rate : float
        Annual risk-free rate for Sharpe and Sortino (default 0.0)
    periods_per_year : int
        Trading days per year for annualization (default 252)

    Returns
    -------
    dict
        Performance metrics with descriptive keys
    """
    # --- Return metrics ---
    total_return = (1 + returns).prod() - 1
    total_days = len(returns)
    ann_return = (1 + total_return) ** (periods_per_year / total_days) - 1

    # --- Risk metrics ---
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    # Downside deviation: volatility of negative returns only
    downside = np.sqrt(
        np.mean(np.minimum(returns, 0) ** 2)
    ) * np.sqrt(periods_per_year)

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Maximum drawdown duration
    duration = 0
    max_duration = 0
    for dd in drawdown:
        if dd < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    # Ulcer Index: root mean square of the drawdown series
    # Multiply by 100 to express in percentage points before squaring
    ulcer = np.sqrt(np.mean((drawdown * 100) ** 2))

    # --- Risk-adjusted metrics ---
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    sortino = (ann_return - risk_free_rate) / downside if downside > 0 else 0
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    # UPI uses annualized return in percentage points to match
    # the units of the Ulcer Index
    upi = (ann_return * 100 - risk_free_rate) / ulcer if ulcer > 0 else 0

    # --- Trading metrics (monthly) ---
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    wins = monthly[monthly > 0]
    losses = monthly[monthly < 0]
    win_rate = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    avg_wl = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and losses.mean() != 0 else np.inf

    # --- Build results dictionary ---
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Ann. Return (CAGR)': f"{ann_return:.2%}",
        'Ann. Volatility': f"{ann_vol:.2%}",
        'Downside Deviation': f"{downside:.2%}",
        'Max Drawdown': f"{max_dd:.2%}",
        'Max DD Duration (days)': max_duration,
        'Ulcer Index': f"{ulcer:.3f}",
        'Sharpe Ratio': f"{sharpe:.3f}",
        'Sortino Ratio': f"{sortino:.3f}",
        'Calmar Ratio': f"{calmar:.3f}",
        'Ulcer Performance Index': f"{upi:.3f}",
        'Win Rate (monthly)': f"{win_rate:.1%}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Avg Win/Loss': f"{avg_wl:.2f}",
    }

    # --- Information ratio (only if benchmark provided) ---
    if benchmark_returns is not None:
        excess = returns - benchmark_returns
        ann_excess = excess.mean() * periods_per_year
        tracking_err = excess.std() * np.sqrt(periods_per_year)
        ir = ann_excess / tracking_err if tracking_err > 0 else 0
        metrics['Information Ratio'] = f"{ir:.3f}"

    return metrics


def calculate_rolling_returns(df_returns, window):
    """
    Calculate annualized rolling returns over a fixed window.

    Parameters
    ----------
    df_returns : pd.DataFrame
        Monthly returns in decimal form. Each column is one strategy.
    window : int
        Rolling window size in months (36 for 3-year rolling returns)

    Returns
    -------
    pd.DataFrame
        Annualized rolling returns for each column.
        First (window - 1) rows will be NaN.
    """
    # Compound the returns within each rolling window:
    #   product of (1 + r) gives total return for the window
    #   minus 1 converts back from growth factor to return
    # Then annualize by raising to the power of (12 / window months)
    rolling_total = df_returns.rolling(window=window).apply(
        lambda x: np.prod(1 + x) - 1, raw=False
    )
    rolling_annualized = (1 + rolling_total) ** (12 / window) - 1
    return rolling_annualized


def rolling_return_stats(rolling_returns, benchmark_col):
    """
    Calculate key statistics for rolling returns vs a benchmark.

    Parameters
    ----------
    rolling_returns : pd.DataFrame
        Rolling returns from calculate_rolling_returns()
    benchmark_col : str
        Column name of the benchmark to compare against

    Returns
    -------
    pd.DataFrame
        Statistics including the percentage of periods where each column
        beat the benchmark, plus the maximum and minimum rolling returns
        observed.
    """
    stats = {}
    benchmark = rolling_returns[benchmark_col].dropna()

    for col in rolling_returns.columns:
        series = rolling_returns[col].dropna()
        common = series.index.intersection(benchmark.index)
        s = series.loc[common]
        b = benchmark.loc[common]

        beat_rate = (s > b).mean()
        stats[col] = {
            'Best Period': f"{s.max():.2%}",
            'Worst Period': f"{s.min():.2%}",
            'Avg Period': f"{s.mean():.2%}",
            'Beat Benchmark': f"{beat_rate:.1%}",
        }

    return pd.DataFrame(stats).T


def audit_normalization(df: pd.DataFrame, verbose: bool = True) -> list:
    """
    Scan a DataFrame for columns that appear to use full-sample normalization.
    From chapter 10

    Checks for:
    - Any column whose values match (x - x.mean()) / x.std() pattern
    - Any column whose mean is approximately 0 and std is approximately 1
      (signature of full-sample standardization)
    - Any column whose values match a full-sample percentile rank

    Parameters:
    -----------
    df      : DataFrame to audit
    verbose : print findings if True

    Returns:
    --------
    List of column names flagged as potentially using full-sample normalization.
    """
    flagged = []

    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()

        if len(series) < 30:
            # Too short to assess meaningfully
            continue

        col_mean = series.mean()
        col_std  = series.std()

        # Flag 1: mean near 0 and std near 1
        # This is the signature of full-sample z-score normalization.
        # An expanding z-score will NOT have mean=0, std=1 across the full series.
        if abs(col_mean) < 0.05 and abs(col_std - 1.0) < 0.05:
            flagged.append(col)
            if verbose:
                print(f"[FLAG] '{col}': mean={col_mean:.4f}, std={col_std:.4f}")
                print(f"       Signature of full-sample z-score normalization.")
                print(f"       Check whether an expanding or rolling window should be used.\n")

        # Flag 2: values fall exactly in [0, 1] with mean near 0.5
        # Signature of a full-sample percentile rank (.rank(pct=True))
        elif series.min() >= 0 and series.max() <= 1 and abs(col_mean - 0.5) < 0.05:
            flagged.append(col)
            if verbose:
                print(f"[FLAG] '{col}': values in [0,1], mean={col_mean:.4f}")
                print(f"       Possible full-sample percentile rank.")
                print(f"       Check whether .expanding().rank(pct=True) should be used.\n")

    if not flagged:
        print("No full-sample normalization patterns detected.")
    else:
        print(f"{len(flagged)} column(s) flagged for review: {flagged}")

    return flagged
