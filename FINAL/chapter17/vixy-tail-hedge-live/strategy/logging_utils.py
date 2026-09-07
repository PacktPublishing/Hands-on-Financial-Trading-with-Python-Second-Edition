"""
strategy/logging_utils.py

Reads and writes data/signal_log.csv. This one file serves two purposes:
a running record of every day the script has run, and the guard that
stops a second run on the same day from sending a duplicate Telegram
alert.
"""
import csv
import os

LOG_COLUMNS = [
    'date', 'signal_on', 'vix_level', 'vix_sma_90', 'spy_realized_vol',
    'spy_weight', 'bil_weight', 'vixy_weight', 'telegram_sent', 'forced',
]


def already_ran_today(log_path, today_str):
    """
    Check whether today's date already has a row in the log.

    Parameters
    ----------
    log_path : str
        Path to signal_log.csv.
    today_str : str
        Today's date as 'YYYY-MM-DD', matching the format written by
        append_log_row().

    Returns
    -------
    bool, True if today's date is already logged.
    """
    if not os.path.exists(log_path):
        return False

    with open(log_path, newline='') as f:
        reader = csv.DictReader(f)
        return any(row['date'] == today_str for row in reader)


def append_log_row(log_path, row):
    """
    Append one row to the CSV log, writing the header first if the file
    does not exist yet.

    Parameters
    ----------
    log_path : str
        Path to signal_log.csv.
    row : dict
        Must contain every key in LOG_COLUMNS.
    """
    file_exists = os.path.exists(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
