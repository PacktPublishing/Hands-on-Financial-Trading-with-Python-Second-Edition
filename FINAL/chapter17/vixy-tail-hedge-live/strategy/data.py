"""
strategy/data.py

Pulls recent daily prices for the four VIXY tail hedge tickers from
yfinance, checks whether a given date is a valid NYSE trading day, and
works out the most recent NYSE session that has closed.
"""
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

TICKERS = ['SPY', 'BIL', 'VIXY', '^VIX']

# A session counts as closed this long after its scheduled close, which
# gives Yahoo time to publish the final daily bar.
SESSION_BUFFER = pd.Timedelta(minutes=15)


def is_trading_day(check_date=None):
    """
    Check whether a date is a valid NYSE trading day.

    Weekends and market holidays return False. This checks the date
    itself against the NYSE calendar, not whether the market happens
    to be open right now.

    Parameters
    ----------
    check_date : date or str, optional
        The date to check. Defaults to today.

    Returns
    -------
    bool
    """
    if check_date is None:
        check_date = pd.Timestamp.today().normalize()
    else:
        check_date = pd.Timestamp(check_date).normalize()

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=check_date, end_date=check_date)
    return not schedule.empty


def last_completed_session(now=None):
    """
    Return the date of the most recent NYSE session that has closed.

    Run after the close on a trading day and this is today. Run before the
    close, on a weekend, or on a market holiday and it is the previous
    trading day. Early-close days are handled by the NYSE calendar.

    Parameters
    ----------
    now : Timestamp, optional
        The moment to evaluate, timezone-aware. Defaults to now in UTC.

    Returns
    -------
    date
    """
    if now is None:
        now = pd.Timestamp.now(tz='UTC')

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=(now - pd.Timedelta(days=10)).date(),
                             end_date=now.date())
    closed = schedule[schedule['market_close'] + SESSION_BUFFER <= now]
    return closed.index[-1].date()


def get_price_data(lookback_days=250):
    """
    Download recent daily prices for the strategy tickers.

    lookback_days is calendar days, not trading days. 250 calendar days
    comfortably covers the 90 trading days the signal needs to warm up,
    even after weekends and holidays are subtracted out.

    Parameters
    ----------
    lookback_days : int
        How many calendar days of history to request. Default 250.

    Returns
    -------
    DataFrame indexed by date with columns SPY, BIL, VIXY, ^VIX, ending at
    the most recent completed session.
    """
    # yfinance treats end as exclusive, so ask for data through tomorrow;
    # otherwise today's close is never included, even after the close.
    end = pd.Timestamp.today() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)

    raw = yf.download(
        TICKERS,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        auto_adjust=True,
        progress=False,
    )

    # auto_adjust=True means 'Close' is already split/dividend adjusted,
    # matching the book's data convention.
    df_prices = raw['Close'][TICKERS].dropna()

    # Keep completed sessions only. During market hours yfinance returns a
    # row for today holding the latest intraday price, which is not a close.
    df_prices = df_prices.loc[:pd.Timestamp(last_completed_session())]

    return df_prices
