"""
strategy/data.py

Pulls recent daily prices for the four VIXY tail hedge tickers from
yfinance, and checks whether a given date is a valid NYSE trading day.
"""
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

TICKERS = ['SPY', 'BIL', 'VIXY', '^VIX']


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
    DataFrame indexed by date with columns SPY, BIL, VIXY, ^VIX.
    """
    end = pd.Timestamp.today()
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

    return df_prices
