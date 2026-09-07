# VIXY Tail Hedge, Live Strategy

Automated daily signal for the VIXY tail hedge strategy from *Financial
Trading with Python, 2nd Edition*. Pulls SPY, BIL, VIXY, and VIX prices,
computes the day's target weights, logs the result, and sends a
Telegram alert. Runs only on NYSE trading days unless forced.

## Setup

1. Clone this repository.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your Telegram bot token
   and chat ID.
4. Run it:
   ```
   python run_strategy.py
   ```
   Use `python run_strategy.py --force` to run on a day the script
   would otherwise skip, either because today is already logged or
   because it is not an NYSE trading day. Use `python run_strategy.py
   --dry-run` to preview today's signal without sending Telegram or
   touching the log at all.

## Files

- `run_strategy.py` -- entry point, run this
- `strategy/data.py` -- price download
- `strategy/signal.py` -- the VIXY tail hedge signal and weight calculation
- `strategy/notify.py` -- Telegram messaging
- `strategy/logging_utils.py` -- CSV logging and the duplicate-run guard
- `strategy/__init__.py` -- marks `strategy/` as a package, intentionally empty
- `requirements.txt` -- pinned dependencies, rebuilt on the server from this file
- `.env.example` -- template for `.env`, safe to commit
- `data/signal_log.csv` -- running record, created automatically on first run

## Deployment

Deployed on PythonAnywhere with a daily scheduled task. See Chapter 18
for the clone, environment, and Tasks setup, and the note on keeping
this repo in sync between local and PythonAnywhere going forward.
