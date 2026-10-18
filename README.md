# Hands-On Financial Trading with Python, Second Edition

This is the code repository for *Hands-On Financial Trading with Python, Second Edition*, published by Packt (book code B37124) and written by Cordell L. Tanny, CFA, FRM, FDP.

The book teaches you how to build a quantitative trading strategy from start to finish. It follows the order a professional works in: idea, data, exploration, preparation, signals, evaluation, backtesting and deployment. One case study, the VIXY tail hedge, runs through the book so you can see each step applied to the same strategy. This repository holds every notebook, data file and helper script that the chapters refer to.

## What is in this repository

```
chapter02/ ... chapter17/   Notebooks, plus the small data files they read
data/                       Price data for the VIXY tail hedge case study
python_scripts/             Helper code shared by some notebooks
PYTHON_TUTORIALS/           Optional tutorials if you are new to Python
requirements-ch13.txt       Package versions that reproduce Chapter 13
```

Notebook file names start with the book code, then the chapter number, then the notebook number, for example `B37124_2_01` or `B37124_16_01`.

| Chapter | Title | Notebooks |
|---|---|---|
| 1 | Introducing Algorithmic Trading | None |
| 2 | Setting up a Python Quantitative Workflow | pandas, NumPy, and functions for finance |
| 3 | Data Retrieval | Data concepts, data retrieval, and the case study download |
| 4 | Understanding Exploratory Data Analysis | EDA fundamentals, correlations, and the case study hypothesis |
| 5 | Data Preparation | Data preparation, and the case study application |
| 6 | Removing Noise from Financial Data | Statistical foundations, moving averages and ARIMA/SARIMAX, advanced signal processing, and a comparison of the techniques |
| 7 | Evaluating Strategy Performance | Performance metrics, evaluation libraries, and statistical significance |
| 8 | Understanding Backtesting | Backtesting with bt, and backtesting with vectorbt |
| 9 | Out-of-Sample Testing | Out-of-sample testing |
| 10 | Walk-Forward Analysis | Walk-forward analysis |
| 11 | How to Spot a Bad Backtest | Lookahead bias, overfitting and parameter sensitivity, and a forensic audit |
| 12 | Time-Series Models | ARIMA, logistic regression, and Prophet |
| 13 | Algorithmic Trading Strategies | Seven strategy notebooks, from Bollinger Bands to a bitcoin moving average strategy |
| 14 | Climbing the Machine Learning Ladder | Machine learning concepts |
| 15 | Machine Learning in Practice | Machine learning in practice |
| 16 | AI as an Alpha Signal Generator | FOMC statement corpus, and large language model determinism |
| 17 | Building a Deployable Strategy Repository | None. The `chapter17` folder holds the strategy files you copy into your own repository |
| 18 | Running Your Strategy Without You | None. It uses the repository you build in Chapter 17 |

## Getting started

### 1. Clone this repository

1. Install GitHub Desktop and sign in with your GitHub account.
2. In GitHub Desktop, choose File, then Clone Repository, then the URL tab.
3. Paste the repository URL from the book, choose a folder on your computer, and click Clone.

### 2. Open a notebook in Google Colab

The notebooks are written to run in Google Colab, which needs nothing installed on your computer.

1. Go to colab.research.google.com and sign in with a Google account.
2. Choose File, then Upload notebook, and pick the notebook from the folder you cloned.
3. Click Connect, then run the cells from the top.

Each notebook opens with the question it answers, so you always know why you are running the code. Notebooks that need a file from this repository download it for you.

### 3. Keep your copy up to date

If a notebook is fixed or a file is added, open GitHub Desktop, click Fetch origin, then click Pull origin. Upload the changed notebook to Colab again. If you edited your earlier copy, rename it first so you can keep it.

## Data and API keys

- Most market data comes from yfinance, which needs no key.
- Chapter 12 uses economic data from FRED and needs a free FRED API key.
- Chapter 16 uses a hosted language model and needs an API key from that provider. The provider may bill for usage.

Each chapter's Technical Requirements section lists exactly what you need before you start.

## If a notebook stops working

Google Colab updates its preinstalled packages, and sometimes Python itself. A notebook that ran last month can fail after an update. Work through these steps in order:

1. Print your package versions and compare them with the book. The Chapter 13 notebooks print a version report, and `requirements-ch13.txt` lists the versions that reproduced the chapter's results.
2. Pull the latest version of this repository and upload the notebook again, in case it has already been fixed.
3. Pin only the package that breaks, for example `!pip install package==1.2.3`. Leave the rest of Colab's setup alone.
4. Ask an AI assistant for help. Give it the full error message, the cell that failed and your package versions, and ask for the smallest change that fixes the problem. Then compare your results with the book.

Small differences from the printed numbers can also come from downloaded market data. Data providers revise prices, for example for dividends and splits, after the book was written.

## Disclaimer

The code and examples in this repository are for education. They are not investment advice, and past performance does not predict future results. Test any strategy yourself before you risk money on it.

## About the author

Cordell L. Tanny, CFA, FRM, FDP, is a quantitative finance professional and a part-time lecturer at Concordia University's John Molson School of Business.

## License

This repository is released under the MIT License. See the `LICENSE` file for details.
