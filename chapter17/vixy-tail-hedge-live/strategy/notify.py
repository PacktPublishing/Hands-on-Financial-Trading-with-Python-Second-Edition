"""
strategy/notify.py

Sends the daily signal message to Telegram. TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID are read from the environment, which the caller loads
from .env before calling send_telegram_message().
"""
import os

import requests

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram_message(text):
    """
    Send a message to the configured Telegram chat.

    Raises if the token or chat ID is missing, or if Telegram returns a
    non-200 response. The caller is expected to catch this, print it,
    and keep running rather than let a Telegram failure stop the script.

    Parameters
    ----------
    text : str
        The message body to send.

    Returns
    -------
    dict, Telegram's JSON response.
    """
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing from the environment."
        )

    url = TELEGRAM_API_URL.format(token=token)
    response = requests.post(
        url,
        data={'chat_id': chat_id, 'text': text},
        timeout=10,
    )
    response.raise_for_status()

    return response.json()
