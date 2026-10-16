"""
run_strategy.py

Entry point for the VIXY tail hedge live strategy. This is the one file
PythonAnywhere's scheduled task and PyCharm's run configuration both call
directly.

A Telegram failure prints to the console (and to PA's task log) but does
not stop the script. The signal, weights, and log row are already built
before the Telegram attempt, so they are unaffected either way.

The script only proceeds on a real NYSE trading day (checked via
pandas_market_calendars, weekends and market holidays are both caught).
--force overrides both this and the same-day guard, so that a run can be
triggered manually on a weekend or holiday for testing. On such a run the
signal is computed from the most recent close available, which will be the
prior trading day. --dry-run bypasses both as well, since it never sends or
logs anything.

Usage:
    python run_strategy.py
    python run_strategy.py --force     # bypass the same-day guard, for manual testing
    python run_strategy.py --dry-run   # preview today's signal, no Telegram, no log write

--dry-run always runs regardless of the same-day guard, since it never
touches the log or sends anything. It is not something the PA scheduled
task command should ever use; it is a local preview tool.
"""
import argparse
import sys
from datetime import date

from dotenv import load_dotenv

from strategy.data import get_price_data, is_trading_day, last_completed_session
from strategy.logging_utils import already_ran_today, append_log_row
from strategy.notify import send_telegram_message
from strategy.signal import get_today_snapshot

LOG_PATH = 'data/signal_log.csv'


def build_message(snapshot):
    status = "HEDGE ON" if snapshot['signal_on'] else "hedge off"
    return (
        f"VIXY Tail Hedge -- {snapshot['date'].strftime('%Y-%m-%d')}\n"
        f"Signal: {status}\n"
        f"VIX: {snapshot['vix_level']}  (90d avg: {snapshot['vix_sma_90']})\n"
        f"SPY realized vol: {snapshot['spy_realized_vol']}%\n"
        f"Weights -- SPY: {snapshot['spy_weight']:.0%}  "
        f"BIL: {snapshot['bil_weight']:.0%}  "
        f"VIXY: {snapshot['vixy_weight']:.0%}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--force', action='store_true',
        help="Run even if today's date is already logged, or if today is not an NYSE trading day. For manual testing."
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Compute and print today's signal only. No Telegram message, no log write."
    )
    args = parser.parse_args()

    load_dotenv()

    today_str = date.today().strftime('%Y-%m-%d')

    if args.dry_run:
        print(f"[{today_str}] Dry run: no Telegram message will be sent, nothing will be logged.")
        df_prices = get_price_data()
        snapshot = get_today_snapshot(df_prices)
        print(build_message(snapshot))
        return

    if not args.force and not is_trading_day():
        print(f"[{today_str}] Not a trading day on the NYSE calendar (weekend or market holiday). "
              f"Skipping. Use --force to override.")
        sys.exit(0)

    # The log is keyed by the date of the close the signal used, so check for
    # the most recent completed session rather than the calendar date.
    latest_close_str = last_completed_session().strftime('%Y-%m-%d')
    if not args.force and already_ran_today(LOG_PATH, latest_close_str):
        print(f"[{today_str}] Already ran today. Skipping. Use --force to override.")
        sys.exit(0)

    print(f"[{today_str}] Starting VIXY tail hedge run...")

    df_prices = get_price_data()
    snapshot = get_today_snapshot(df_prices)
    message = build_message(snapshot)

    print(message)

    telegram_sent = False
    try:
        send_telegram_message(message)
        telegram_sent = True
        print(f"[{today_str}] Telegram alert sent.")
    except Exception as exc:
        print(f"[{today_str}] Telegram send failed: {exc}")
        print(f"[{today_str}] Continuing. Signal and weights are still logged below.")

    row = {
        'date': snapshot['date'].strftime('%Y-%m-%d'),
        'signal_on': snapshot['signal_on'],
        'vix_level': snapshot['vix_level'],
        'vix_sma_90': snapshot['vix_sma_90'],
        'spy_realized_vol': snapshot['spy_realized_vol'],
        'spy_weight': snapshot['spy_weight'],
        'bil_weight': snapshot['bil_weight'],
        'vixy_weight': snapshot['vixy_weight'],
        'telegram_sent': telegram_sent,
        'forced': args.force,
    }
    append_log_row(LOG_PATH, row)
    print(f"[{today_str}] Logged to {LOG_PATH}.")


if __name__ == '__main__':
    main()
