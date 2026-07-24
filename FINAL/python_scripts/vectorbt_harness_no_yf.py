"""
vectorbt_harness_no_yf.py
-------------------------
Reusable VectorBT backtest harness for Hands-On Financial Trading with Python,
2nd Edition.

This is the offline version of vectorbt_harness.py. It is identical in every
respect except one: it does not download anything. Prices are supplied by the
caller, either as a path to a Parquet or CSV file or as an already-loaded
DataFrame, and the benchmark is read from that same price data rather than
fetched separately.

That change matters for two reasons. Readers can run every backtest in the book
from the data files supplied with the repository, without depending on a data
provider being reachable or within its rate limit. And a harness that takes its
prices as an argument can be tested against known inputs, which one that
downloads inside itself cannot.

A strategy notebook produces a sparse weights file (one row per trade date,
tickers as columns, target weights as decimals). This module turns that file
into a backtest and a scored result, so the VectorBT machinery is written once
and every strategy is run and scored identically:

    from vectorbt_harness_no_yf import run_backtest
    result = run_backtest('weights.csv', 'prices.parquet', benchmark='SPY')
    print(result.metrics)
    result.plot()

Conventions (identical to vectorbt_harness.py and Notebook 8.2):
- Weights are dated to the signal day and fed to VectorBT unshifted; orders
  execute at that day's close.
- The weights file is sparse: a row is a trade date, NaN between rows means
  hold, and the position drifts with price until the next dated row.
- Prices are dividend-adjusted daily closes.
- Price columns are reordered to match the weights columns exactly, so each
  weight is paired with the correct asset's price.
- Trades are whole shares (size_granularity=1).
- If the benchmark has less history than the strategy, the strategy is
  truncated to the benchmark start so the comparison window matches.
- Metrics are computed on monthly-resampled returns via the Chapter 7
  calculate_performance_metrics, with periods_per_year (default 12).

The price data must contain a column for every ticker in the weights file and a
column for the benchmark. Any extra columns are ignored, so one panel file can
serve many strategies.

Requires functions.py (the Chapter 7 toolkit) in the import path.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt

from functions import calculate_performance_metrics


# --------------------------------------------------------------------------- #
# Data loading and preparation
# --------------------------------------------------------------------------- #
def load_weights(weights_csv):
    """
    Load a sparse strategy weights file under the chapter contract.

    The file has dates as the index (each row is a trade date on which the
    portfolio is rebalanced to the row's targets), tickers as columns, and
    target weights as decimals (0.20 meaning twenty percent of capital). Rows
    appear only on trade dates; there are no rows in between.

    Parameters
    ----------
    weights_csv : str
        Path to the weights CSV file.

    Returns
    -------
    pandas.DataFrame
        The weights, parsed with a DatetimeIndex and sorted ascending by date.
    """
    weights = pd.read_csv(weights_csv, index_col=0, parse_dates=True).sort_index()
    return weights


def load_prices(prices):
    """
    Load the price panel supplied by the caller.

    Accepts a path to a Parquet or CSV file, or a DataFrame that is already in
    memory. A DataFrame is copied rather than used in place, so the caller's
    object is never modified by the backtest. Whichever form is supplied, the
    result is sorted ascending by date and carries a DatetimeIndex.

    Parameters
    ----------
    prices : str or pathlib.Path or pandas.DataFrame
        Either a path ending in .parquet, .pq, or .csv, or a DataFrame of
        dividend-adjusted daily closes indexed by trading day with one column
        per ticker.

    Returns
    -------
    pandas.DataFrame
        Prices with a sorted DatetimeIndex.

    Raises
    ------
    ValueError
        If a path is supplied with an unrecognized extension, or if the loaded
        index cannot be interpreted as dates.
    """
    if isinstance(prices, pd.DataFrame):
        df = prices.copy()
    else:
        path = str(prices)
        lower = path.lower()
        if lower.endswith(('.parquet', '.pq')):
            df = pd.read_parquet(path)
        elif lower.endswith('.csv'):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            raise ValueError(
                f'Unrecognized price file type: {path}. '
                'Supply a .parquet, .pq, or .csv path, or a DataFrame.'
            )

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'Price data index could not be parsed as dates. The index must '
                'be the trading day.'
            ) from exc

    return df.sort_index()


def align_price_columns(prices, weights):
    """
    Reorder price columns to match the weights columns exactly.

    Price panels are rarely stored in the order of the weights file; a panel
    saved from a download is typically alphabetical. VectorBT pairs weights and
    prices by position in places, so a mismatched order applies each weight to
    the wrong asset's price. This function asserts every weights ticker has a
    price and returns the prices reordered to the weights column order.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price panel, columns in arbitrary order, possibly containing extra
        tickers that this strategy does not trade.
    weights : pandas.DataFrame
        The weights, whose column order is authoritative.

    Returns
    -------
    pandas.DataFrame
        Prices with columns reordered to exactly match the weights columns.

    Raises
    ------
    ValueError
        If any weights ticker is missing from the price panel.
    """
    missing = [t for t in weights.columns if t not in prices.columns]
    if missing:
        raise ValueError(
            f'Prices missing for tickers: {missing}. '
            f'Available columns: {list(prices.columns)}'
        )
    prices = prices[list(weights.columns)]
    assert list(prices.columns) == list(weights.columns), 'column alignment failed'
    return prices


def extract_benchmark(prices, benchmark):
    """
    Pull the benchmark price series out of the price panel.

    The benchmark is read from the same data as the strategy assets, so the two
    are guaranteed to share a calendar and an adjustment convention. This is the
    one place this harness differs in substance from the downloading version,
    which fetched the benchmark separately.

    Parameters
    ----------
    prices : pandas.DataFrame
        The full price panel, before column alignment trims it.
    benchmark : str
        Benchmark ticker, which must be a column in the panel.

    Returns
    -------
    pandas.Series
        The benchmark's adjusted closes, missing values dropped.

    Raises
    ------
    ValueError
        If the benchmark is not a column in the price panel.
    """
    if benchmark not in prices.columns:
        raise ValueError(
            f"Benchmark '{benchmark}' is not a column in the price data. "
            f'Available columns: {list(prices.columns)}'
        )
    return prices[benchmark].dropna()


def place_on_calendar(weights, price_index):
    """
    Place the sparse weight rows onto the trading calendar.

    Each trade date is snapped to a real trading day (the next available day if
    the dated row falls on a non-trading day, so no trade is dropped on a
    holiday), then the weights are reindexed onto the full price calendar.
    Non-trade days become NaN, which VectorBT reads as "no order, hold". The
    resulting sparse matrix is what the portfolio consumes.

    Parameters
    ----------
    weights : pandas.DataFrame
        Sparse weights, one row per trade date.
    price_index : pandas.DatetimeIndex
        The trading calendar from the price data.

    Returns
    -------
    pandas.DataFrame
        Weights reindexed onto the price calendar, NaN on non-trade days.
    """
    snapped = []
    for d in weights.index:
        if d in price_index:
            snapped.append(d)
        else:
            after = price_index[price_index >= d]
            snapped.append(after[0] if len(after) else price_index[-1])

    weights_on_cal = weights.copy()
    weights_on_cal.index = pd.DatetimeIndex(snapped)
    weights_sparse = weights_on_cal.reindex(price_index)
    return weights_sparse


def trim_to_window(prices, weights_sparse, bench_px, end_at_last_trade=False,
                   verbose=True):
    """
    Trim prices, weights, and benchmark to a common backtest window.

    The window starts on the later of two dates: the strategy's first trade
    (the first row carrying a nonzero weight) and the benchmark's first
    available day. If the benchmark has less history than the strategy, the
    strategy is truncated to the benchmark start so the comparison is fair, and
    a warning is printed. If that truncation lands between trade dates, the
    first row of the trimmed weights would be NaN and would start the backtest
    in cash; it is seeded with the allocation held as of the start date so the
    held position carries into the window.

    The window ends on the last day of the price data. A position held on the
    final trade date keeps earning its return until the data runs out, which is
    what a live portfolio would do. The downloading harness ends roughly a week
    after the last trade date instead, because it only requested prices that far
    ahead; set end_at_last_trade to reproduce that shorter window when comparing
    results against a backtest run under the older harness.

    Parameters
    ----------
    prices : pandas.DataFrame
        Adjusted closes, columns aligned to the weights.
    weights_sparse : pandas.DataFrame
        Sparse weights placed on the price calendar.
    bench_px : pandas.Series
        Benchmark adjusted closes, taken from the same price panel.
    end_at_last_trade : bool, optional
        If True, end the window one week after the final trade date rather than
        at the end of the price data (default False). This exists only to
        reproduce the downloading harness's window.
    verbose : bool, optional
        If True (default), print the truncation warning and window summary.

    Returns
    -------
    prices_t : pandas.DataFrame
        Prices trimmed to the common window.
    weights_t : pandas.DataFrame
        Weights trimmed to the common window, first row seeded if needed.
    bench_t : pandas.Series
        Benchmark daily returns over the common window.
    start_day : pandas.Timestamp
        The first day of the backtest window.

    Raises
    ------
    ValueError
        If the weights contain no nonzero rows (nothing to backtest).
    """
    # First trade date: the first row that carries a nonzero target weight.
    trade_rows = weights_sparse.dropna(how='all')
    nonzero = trade_rows[(trade_rows.fillna(0) != 0).any(axis=1)]
    if nonzero.empty:
        raise ValueError('No nonzero weights found: nothing to backtest.')
    first_trade = nonzero.index.min()

    # Benchmark first available day.
    bench_first = bench_px.dropna().index.min()

    # Start on the later of the first trade and the benchmark's first day.
    start_day = max(first_trade, bench_first)

    if bench_first > first_trade and verbose:
        gap = (bench_first - first_trade).days
        print('WARNING: benchmark has less history than the strategy.')
        print(f'  Strategy first trade date:     {first_trade.date()}')
        print(f'  Benchmark first available day: {bench_first.date()}')
        print(f'  Truncating the strategy to the benchmark start; the first '
              f'{gap} calendar days are dropped from both.\n')

    # End of window: the full price history, or the older harness's shorter
    # window if the caller asked for a like-for-like comparison.
    end_day = prices.index.max()
    if end_at_last_trade:
        last_trade = weights_sparse.dropna(how='all').index.max()
        end_day = min(end_day, last_trade + pd.Timedelta(days=7))

    # Trim prices, weights, and benchmark to the common window.
    prices_t = prices.loc[start_day:end_day].copy()
    weights_t = weights_sparse.loc[start_day:end_day].copy()
    bench_t = bench_px.loc[start_day:end_day].pct_change().dropna()

    # If truncation landed between trade dates, seed the first trimmed row with
    # the allocation held as of start_day so the backtest does not start in cash.
    if weights_t.iloc[0].isna().all():
        held = weights_sparse.loc[:start_day].dropna(how='all')
        if not held.empty:
            weights_t.iloc[0] = held.iloc[-1].values

    if verbose:
        print(f"Backtest start: {start_day.date()}")
        print(f"Backtest end:   {prices_t.index.max().date()}")
        print(f"Trading days:   {len(prices_t)}")
        print(f"Trades in window: {int(weights_t.notna().any(axis=1).sum())}")

    return prices_t, weights_t, bench_t, start_day


# --------------------------------------------------------------------------- #
# Backtest and scoring
# --------------------------------------------------------------------------- #
def run_portfolio(prices, weights, fees, slippage, init_cash):
    """
    Run a VectorBT portfolio from a sparse weight matrix.

    Uses Portfolio.from_orders with the Notebook 8.2 convention:
    size_type='targetpercent' so each non-NaN row sets the target fraction of
    portfolio value per asset and NaN rows place no order; cash_sharing=True
    with group_by=True pools the assets into one portfolio so a sale funds a
    purchase; call_seq='auto' settles sells before buys on a trade date;
    size_granularity=1 forces whole-share trades. Orders fill at the close of
    the dated row.

    Parameters
    ----------
    prices : pandas.DataFrame
        Adjusted closes, columns aligned to the weights.
    weights : pandas.DataFrame
        Sparse weight matrix on the trading calendar (NaN means hold).
    fees : float
        Commission as a fraction of trade value (e.g. 0.0010 for 10 bps).
    slippage : float
        Slippage as a fraction of trade value (e.g. 0.0005 for 5 bps).
    init_cash : float
        Starting capital.

    Returns
    -------
    vectorbt.portfolio.base.Portfolio
        The constructed portfolio object.
    """
    return vbt.Portfolio.from_orders(
        close=prices,
        size=weights,              # sparse: trade on non-NaN rows, hold on NaN
        size_type='targetpercent',
        cash_sharing=True,
        group_by=True,
        call_seq='auto',
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        size_granularity=1,        # whole shares only, no fractional trades
        freq='1D',
    )


def to_monthly(daily):
    """
    Compound a daily return series into monthly returns.

    Parameters
    ----------
    daily : pandas.Series
        Daily returns in decimal form.

    Returns
    -------
    pandas.Series
        Monthly returns, one observation per calendar month end.
    """
    return daily.resample('ME').apply(lambda x: (1 + x).prod() - 1)


def score_portfolio(pf, bench_daily, periods_per_year=12):
    """
    Score a portfolio on monthly returns against a benchmark.

    Pulls the portfolio's daily returns, aligns them to the benchmark's dates,
    resamples both to monthly by compounding, and runs the Chapter 7
    calculate_performance_metrics. The benchmark is aligned to the strategy so
    the metrics are computed on the overlapping window.

    Parameters
    ----------
    pf : vectorbt.portfolio.base.Portfolio
        The portfolio to score.
    bench_daily : pandas.Series
        Benchmark daily returns.
    periods_per_year : int, optional
        Annualization factor for the monthly metrics (default 12).

    Returns
    -------
    dict
        The metric set from calculate_performance_metrics, including the
        information ratio (benchmark provided).
    """
    strat_daily = pf.returns()
    common = strat_daily.index.intersection(bench_daily.index)
    strat_m = to_monthly(strat_daily.loc[common])
    bench_m = to_monthly(bench_daily.loc[common])
    return calculate_performance_metrics(
        strat_m, benchmark_returns=bench_m,
        risk_free_rate=0.0, periods_per_year=periods_per_year,
    )


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #
class BacktestResult:
    """
    Container for the output of run_backtest.

    Holds the scored metric table, the uncosted and costed portfolio objects,
    the benchmark returns, and the backtest window, and provides a plot of the
    equity curve and drawdown.

    Attributes
    ----------
    metrics : pandas.DataFrame
        Columns for the uncosted strategy, the costed strategy, and the
        benchmark buy and hold, each scored on monthly returns.
    pf_no_cost : vectorbt.portfolio.base.Portfolio
        The portfolio run with zero fees and slippage.
    pf_costs : vectorbt.portfolio.base.Portfolio
        The portfolio run with the supplied fees and slippage.
    bench_returns : pandas.Series
        Benchmark daily returns over the backtest window.
    benchmark : str
        The benchmark ticker.
    start_day : pandas.Timestamp
        First day of the backtest window.
    prices : pandas.DataFrame
        Prices used in the backtest (trimmed, aligned).
    weights : pandas.DataFrame
        Weights used in the backtest (trimmed, on the calendar).
    periods_per_year : int
        Annualization factor used for scoring.
    """

    def __init__(self, metrics, pf_no_cost, pf_costs, bench_returns,
                 benchmark, start_day, prices, weights, periods_per_year):
        """
        Store the backtest outputs. See the class docstring for the meaning of
        each attribute; the parameters here correspond one to one with them.
        """
        self.metrics = metrics
        self.pf_no_cost = pf_no_cost
        self.pf_costs = pf_costs
        self.bench_returns = bench_returns
        self.benchmark = benchmark
        self.start_day = start_day
        self.prices = prices
        self.weights = weights
        self.periods_per_year = periods_per_year

    def plot(self, use_costs=False, figsize=(14, 8)):
        """
        Plot the monthly equity curve against the benchmark, with drawdown.

        Draws the strategy's growth of capital and the benchmark buy and hold
        in a top panel, and the strategy's drawdown in a lower panel, on the
        monthly basis used for scoring.

        Parameters
        ----------
        use_costs : bool, optional
            If True, plot the costed portfolio; otherwise the uncosted one
            (default False).
        figsize : tuple, optional
            Figure size (default (14, 8)).

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        pf = self.pf_costs if use_costs else self.pf_no_cost
        strat_daily = pf.returns()
        common = strat_daily.index.intersection(self.bench_returns.index)
        strat_curve = (1 + to_monthly(strat_daily.loc[common])).cumprod()
        bench_curve = (1 + to_monthly(self.bench_returns.loc[common])).cumprod()
        dd = strat_curve / strat_curve.cummax() - 1

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={'height_ratios': [3, 1]},
        )
        label = 'Strategy w/ Costs' if use_costs else 'Strategy'
        ax1.plot(strat_curve.index, strat_curve.values, label=label)
        ax1.plot(bench_curve.index, bench_curve.values,
                 label=f'{self.benchmark} Buy & Hold', alpha=0.8)
        ax1.set_ylabel('Growth of $1')
        ax1.set_title(f'Strategy vs {self.benchmark} Buy & Hold')
        ax1.legend(loc='upper left')

        ax2.fill_between(dd.index, dd.values, 0, color='tab:red', alpha=0.4)
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        plt.tight_layout()
        plt.show()
        return fig


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def run_backtest(weights_csv, prices, benchmark='SPY', fees=0.0010,
                 slippage=0.0005, init_cash=10_000, periods_per_year=12,
                 end_at_last_trade=False, verbose=True):
    """
    Run a full backtest for a strategy weights file and score it.

    This is the single entry point a strategy notebook calls. It loads the
    sparse weights file and the supplied price data, aligns the price columns to
    the weights, takes the benchmark from the same price data, places the
    weights on the trading calendar, trims to the common window (truncating to
    the benchmark start if the benchmark is younger), runs an uncosted and a
    costed VectorBT portfolio with whole-share trades, and scores both plus the
    benchmark on monthly returns.

    Parameters
    ----------
    weights_csv : str
        Path to the sparse weights CSV (trade dates as rows, tickers as
        columns, target weights as decimals).
    prices : str or pathlib.Path or pandas.DataFrame
        Dividend-adjusted daily closes: a path to a .parquet or .csv file, or a
        DataFrame already in memory. Must contain a column for every ticker in
        the weights file and a column for the benchmark. Extra columns are
        ignored, so one panel can serve many strategies.
    benchmark : str, optional
        Buy-and-hold benchmark ticker, which must be a column in the price data
        (default 'SPY').
    fees : float, optional
        Commission as a fraction of trade value for the costed run
        (default 0.0010, i.e. 10 bps).
    slippage : float, optional
        Slippage as a fraction of trade value for the costed run
        (default 0.0005, i.e. 5 bps).
    init_cash : float, optional
        Starting capital (default 10_000).
    periods_per_year : int, optional
        Annualization factor for the monthly metrics (default 12).
    end_at_last_trade : bool, optional
        If True, end the backtest one week after the final trade date instead of
        at the end of the price data (default False). The default lets a held
        position keep earning until the data runs out, which is what a live
        portfolio does; set this only to reproduce the window produced by the
        downloading harness.
    verbose : bool, optional
        If True (default), print progress and the metric table.

    Returns
    -------
    BacktestResult
        The scored metrics, both portfolio objects, the benchmark returns, and
        the backtest window, with a plot() method for the equity curve.
    """
    # 1. Load weights and prices, take the benchmark, align columns.
    weights = load_weights(weights_csv)
    prices_all = load_prices(prices)
    bench_px = extract_benchmark(prices_all, benchmark)
    prices = align_price_columns(prices_all, weights)

    # 2. Place weights on the calendar and trim to the common window.
    weights_sparse = place_on_calendar(weights, prices.index)
    prices_t, weights_t, bench_t, start_day = trim_to_window(
        prices, weights_sparse, bench_px,
        end_at_last_trade=end_at_last_trade, verbose=verbose
    )

    # 3. Run the portfolio uncosted and costed.
    pf_no_cost = run_portfolio(prices_t, weights_t, 0.0, 0.0, init_cash)
    pf_costs = run_portfolio(prices_t, weights_t, fees, slippage, init_cash)

    # 4. Score both, plus the benchmark on the same window.
    metrics_no_cost = score_portfolio(pf_no_cost, bench_t, periods_per_year)
    metrics_costs = score_portfolio(pf_costs, bench_t, periods_per_year)
    common = pf_no_cost.returns().index.intersection(bench_t.index)
    bench_m_self = to_monthly(bench_t.loc[common])
    metrics_bench = calculate_performance_metrics(
        bench_m_self, risk_free_rate=0.0, periods_per_year=periods_per_year,
    )
    metrics = pd.DataFrame({
        'Strategy No Cost':  metrics_no_cost,
        'Strategy w/ Costs': metrics_costs,
        f'{benchmark} Buy & Hold': metrics_bench,
    })

    if verbose:
        print('\nHarness evaluation (monthly metrics)')
        print('=' * 60)
        print(metrics.to_string())

    return BacktestResult(
        metrics=metrics, pf_no_cost=pf_no_cost, pf_costs=pf_costs,
        bench_returns=bench_t, benchmark=benchmark, start_day=start_day,
        prices=prices_t, weights=weights_t, periods_per_year=periods_per_year,
    )
