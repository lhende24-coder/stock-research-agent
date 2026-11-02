"""
Stock Research Terminal — Local Web App
=========================================
Sections: Research | Paper Trading | Charts & Gaps

WHY LOCAL: browsers block direct calls to Yahoo Finance / Alpha Vantage,
so this runs a small server on YOUR machine (via Python) and your browser
talks to it. Nothing leaves your computer except the same requests your
scripts already made.

SETUP: keep this file in the same folder as stock_research_agent.py.
RUN:   py app.py        (opens your browser automatically)

10K / 10Q TERMINAL COMMANDS: these read filings from SEC EDGAR, which asks every program to identify itself with a
contact email. Set SEC_USER_AGENT (search for it in this file) to "Your Name you@email.com" before using them.

NOTES ON A FEW FEATURES:
- Paper trading (buy/short/puts) uses REAL live prices, but is 100%
  simulated — no real money, no real broker, ever. It's not investment
  advice and doesn't reflect real-world costs like margin interest,
  assignment risk, or slippage.
- "Gaps" are computed from Yahoo Finance's own intraday data (resampled
  into 10-minute candles), not from TradingView's data feed — TradingView's
  free embed only displays a chart, it doesn't expose raw prices to code.
  The TradingView chart is shown alongside for your own visual reference.
"""

from __future__ import annotations

import io
import json
import math
import operator
import os
import re
import time
import xml.etree.ElementTree as ET
from statistics import NormalDist
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from stock_research_agent import StockResearchAgent
except ImportError:
    raise SystemExit(
        "\nCouldn't find stock_research_agent.py.\n"
        "Make sure app.py is saved in the SAME FOLDER as stock_research_agent.py.\n"
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")

STARTING_CASH = 10_000.0


# --------------------------------------------------------------------------- #
# Market data helpers
# --------------------------------------------------------------------------- #

def get_last_price(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
    if hist.empty:
        raise ValueError(f"No price data found for '{ticker}'. Check the ticker symbol.")
    return float(hist.iloc[-1])


def get_option_expirations(ticker: str) -> list[str]:
    tk = yf.Ticker(ticker)
    return list(tk.options or [])


def get_put_chain(ticker: str, expiration: str) -> list[dict]:
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiration)
    puts = chain.puts.copy()
    cols = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest"]
    cols = [c for c in cols if c in puts.columns]
    puts = puts[cols].replace({np.nan: None})
    return puts.to_dict(orient="records")


def get_put_quote(ticker: str, expiration: str, strike: float) -> dict:
    for p in get_put_chain(ticker, expiration):
        if abs(float(p["strike"]) - float(strike)) < 1e-6:
            return p
    raise ValueError("Couldn't find that put contract anymore (it may have expired or delisted).")


# --------------------------------------------------------------------------- #
# Paper trading portfolio
# --------------------------------------------------------------------------- #

def _blank_portfolio() -> dict:
    return {"cash": STARTING_CASH, "long": {}, "short": {}, "options": [], "log": []}


def load_portfolio() -> dict:
    if not os.path.exists(PORTFOLIO_FILE):
        p = _blank_portfolio()
        save_portfolio(p)
        return p
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)


def save_portfolio(p: dict):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)


def _log(p: dict, action: str, ticker: str, qty: float, price: float, details: str = ""):
    p["log"].insert(0, {
        "time": datetime.now().isoformat(timespec="seconds"),
        "action": action, "ticker": ticker, "qty": qty, "price": price, "details": details,
    })
    p["log"] = p["log"][:200]  # keep the log from growing forever


def execute_trade(action: str, ticker: str, qty: float, opt: dict | None = None) -> dict:
    p = load_portfolio()
    ticker = ticker.upper().strip()
    qty = float(qty)

    if action in ("buy", "sell", "short", "cover"):
        price = get_last_price(ticker)

        if action == "buy":
            cost = price * qty
            if cost > p["cash"] + 1e-6:
                raise ValueError(f"Not enough cash. This costs ${cost:,.2f}, you have ${p['cash']:,.2f}.")
            pos = p["long"].get(ticker, {"shares": 0, "avg_cost": 0.0})
            new_shares = pos["shares"] + qty
            pos["avg_cost"] = (pos["avg_cost"] * pos["shares"] + cost) / new_shares
            pos["shares"] = new_shares
            p["long"][ticker] = pos
            p["cash"] -= cost
            _log(p, "buy", ticker, qty, price)

        elif action == "sell":
            pos = p["long"].get(ticker)
            if not pos or pos["shares"] < qty - 1e-6:
                raise ValueError(f"You don't own {qty} shares of {ticker} to sell.")
            proceeds = price * qty
            realized = (price - pos["avg_cost"]) * qty
            pos["shares"] -= qty
            if pos["shares"] <= 1e-9:
                del p["long"][ticker]
            else:
                p["long"][ticker] = pos
            p["cash"] += proceeds
            _log(p, "sell", ticker, qty, price, f"realized P&L ${realized:,.2f}")

        elif action == "short":
            proceeds = price * qty
            pos = p["short"].get(ticker, {"shares": 0, "avg_price": 0.0})
            new_shares = pos["shares"] + qty
            pos["avg_price"] = (pos["avg_price"] * pos["shares"] + proceeds) / new_shares
            pos["shares"] = new_shares
            p["short"][ticker] = pos
            p["cash"] += proceeds
            _log(p, "short", ticker, qty, price)

        elif action == "cover":
            pos = p["short"].get(ticker)
            if not pos or pos["shares"] < qty - 1e-6:
                raise ValueError(f"You don't have a short position of {qty} shares in {ticker} to cover.")
            cost = price * qty
            if cost > p["cash"] + 1e-6:
                raise ValueError(f"Not enough cash to cover. Needs ${cost:,.2f}, you have ${p['cash']:,.2f}.")
            realized = (pos["avg_price"] - price) * qty
            pos["shares"] -= qty
            if pos["shares"] <= 1e-9:
                del p["short"][ticker]
            else:
                p["short"][ticker] = pos
            p["cash"] -= cost
            _log(p, "cover", ticker, qty, price, f"realized P&L ${realized:,.2f}")

    elif action == "buy_put":
        expiration, strike = opt["expiration"], float(opt["strike"])
        quote = get_put_quote(ticker, expiration, strike)
        premium = quote.get("lastPrice") or quote.get("ask") or 0
        cost = premium * 100 * qty
        if cost > p["cash"] + 1e-6:
            raise ValueError(f"Not enough cash. This costs ${cost:,.2f}, you have ${p['cash']:,.2f}.")
        p["cash"] -= cost
        p["options"].append({
            "id": f"{ticker}-{strike}-{expiration}-{datetime.now().timestamp()}",
            "ticker": ticker, "type": "put", "strike": strike, "expiration": expiration,
            "contracts": qty, "entry_premium": premium,
        })
        _log(p, "buy_put", ticker, qty, premium, f"strike ${strike} exp {expiration}")

    elif action == "sell_put":
        opt_id = opt["id"]
        position = next((o for o in p["options"] if o["id"] == opt_id), None)
        if not position:
            raise ValueError("Couldn't find that option position (already closed?).")
        try:
            quote = get_put_quote(position["ticker"], position["expiration"], position["strike"])
            premium = quote.get("lastPrice") or quote.get("bid") or 0
        except Exception:
            premium = 0  # likely expired/delisted; treat as worthless if we can't re-quote it
        proceeds = premium * 100 * position["contracts"]
        realized = (premium - position["entry_premium"]) * 100 * position["contracts"]
        p["cash"] += proceeds
        p["options"] = [o for o in p["options"] if o["id"] != opt_id]
        _log(p, "sell_put", position["ticker"], position["contracts"], premium, f"realized P&L ${realized:,.2f}")

    else:
        raise ValueError(f"Unknown trade action: {action}")

    save_portfolio(p)
    return p


def auto_settle_expired_options(p: dict) -> dict:
    """On load, cash-settle any put whose expiration date has passed."""
    today = date.today()
    remaining = []
    changed = False
    for o in p["options"]:
        try:
            exp_date = datetime.strptime(o["expiration"], "%Y-%m-%d").date()
        except Exception:
            remaining.append(o)
            continue
        if exp_date < today:
            try:
                underlying_price = get_last_price(o["ticker"])
            except Exception:
                underlying_price = o["strike"]  # best-effort fallback
            payoff = max(0.0, o["strike"] - underlying_price) * 100 * o["contracts"]
            p["cash"] += payoff
            realized = payoff - (o["entry_premium"] * 100 * o["contracts"])
            _log(p, "expired_put", o["ticker"], o["contracts"], 0,
                 f"settled at expiration, payoff ${payoff:,.2f}, P&L ${realized:,.2f}")
            changed = True
        else:
            remaining.append(o)
    p["options"] = remaining
    if changed:
        save_portfolio(p)
    return p


def portfolio_snapshot() -> dict:
    p = load_portfolio()
    p = auto_settle_expired_options(p)

    long_out, short_out, opt_out = [], [], []
    total_value = p["cash"]

    for ticker, pos in p["long"].items():
        try:
            price = get_last_price(ticker)
        except Exception:
            price = pos["avg_cost"]
        value = price * pos["shares"]
        total_value += value
        long_out.append({
            "ticker": ticker, "shares": pos["shares"], "avg_cost": pos["avg_cost"],
            "price": price, "value": value, "unrealized_pl": (price - pos["avg_cost"]) * pos["shares"],
        })

    for ticker, pos in p["short"].items():
        try:
            price = get_last_price(ticker)
        except Exception:
            price = pos["avg_price"]
        liability = price * pos["shares"]
        total_value -= liability
        short_out.append({
            "ticker": ticker, "shares": pos["shares"], "avg_price": pos["avg_price"],
            "price": price, "liability": liability,
            "unrealized_pl": (pos["avg_price"] - price) * pos["shares"],
        })

    for o in p["options"]:
        try:
            quote = get_put_quote(o["ticker"], o["expiration"], o["strike"])
            current_premium = quote.get("lastPrice") or quote.get("bid") or 0
        except Exception:
            current_premium = o["entry_premium"]
        value = current_premium * 100 * o["contracts"]
        total_value += value
        opt_out.append({
            **o, "current_premium": current_premium, "value": value,
            "unrealized_pl": (current_premium - o["entry_premium"]) * 100 * o["contracts"],
        })

    return {
        "cash": p["cash"], "long": long_out, "short": short_out, "options": opt_out,
        "log": p["log"][:30], "total_value": total_value,
        "total_return_pct": (total_value - STARTING_CASH) / STARTING_CASH,
    }


# --------------------------------------------------------------------------- #
# Gap detection (Yahoo Finance intraday data, resampled to 10-minute candles)
# --------------------------------------------------------------------------- #

def find_gaps(ticker: str, days: int = 30, min_gap_pct: float = 0.003) -> list[dict]:
    tk = yf.Ticker(ticker)
    hist = tk.history(period=f"{min(days, 59)}d", interval="5m")
    if hist.empty:
        raise ValueError("No intraday data available (Yahoo limits 5-minute data to the last ~60 days).")

    ten_min = hist.resample("10min").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()

    highs, lows = ten_min["High"].to_numpy(dtype=float), ten_min["Low"].to_numpy(dtype=float)
    stamps = list(ten_min.index)
    last_ts = stamps[-1]

    def fill_info(i, prev_close, open_px, up):
        """A gap UP is filled once price trades back down to the prior close (a low <= prev close);
        a gap DOWN is filled once price trades back up to it (a high >= prev close). The gap candle
        itself counts, because price can retrace inside the same candle."""
        if up:
            hit = np.nonzero(lows[i:] <= prev_close)[0]
            best = float(np.min(lows[i:]))
            closed = (open_px - best) / (open_px - prev_close) if open_px != prev_close else 1.0
        else:
            hit = np.nonzero(highs[i:] >= prev_close)[0]
            best = float(np.max(highs[i:]))
            closed = (best - open_px) / (prev_close - open_px) if open_px != prev_close else 1.0
        if len(hit):
            j = i + int(hit[0])
            return {"filled": True, "filled_time": stamps[j].isoformat(),
                    "fill_minutes": (stamps[j] - stamps[i]).total_seconds() / 60,
                    "fill_days": (stamps[j].date() - stamps[i].date()).days, "closed_pct": 1.0}
        return {"filled": False, "filled_time": None, "fill_minutes": None, "fill_days": None,
                "closed_pct": max(0.0, min(float(closed), 0.999)),
                "open_for_days": (last_ts.date() - stamps[i].date()).days}

    gaps = []
    prev_close, prev_day = None, None
    for i, (ts, row) in enumerate(ten_min.iterrows()):
        this_day = ts.date()
        if prev_close is not None:
            gap = row["Open"] - prev_close
            gap_pct = gap / prev_close if prev_close else 0
            if abs(gap_pct) >= min_gap_pct:
                g = {
                    "time": ts.isoformat(),
                    "prev_close": round(float(prev_close), 4),
                    "open": round(float(row["Open"]), 4),
                    "gap_pct": gap_pct,
                    "direction": "up" if gap > 0 else "down",
                    "overnight": this_day != prev_day,
                }
                g.update(fill_info(i, float(prev_close), float(row["Open"]), gap > 0))
                gaps.append(g)
        prev_close, prev_day = row["Close"], this_day

    gaps.sort(key=lambda g: g["time"], reverse=True)
    return gaps[:100]


# --------------------------------------------------------------------------- #
# Command terminal (function codes like EQS / ECOS)
# --------------------------------------------------------------------------- #

FRED_SERIES = {
    "GDP": ("GDPC1", "Real GDP (chained 2017 $, billions, quarterly)"),
    "CPI": ("CPIAUCSL", "Consumer Price Index, all urban consumers"),
    "UNRATE": ("UNRATE", "Unemployment rate (%)"),
    "FEDFUNDS": ("FEDFUNDS", "Effective federal funds rate (%)"),
    "10Y": ("DGS10", "10-year Treasury yield (%)"),
    "2Y": ("DGS2", "2-year Treasury yield (%)"),
    "PAYROLLS": ("PAYEMS", "Total nonfarm payrolls (thousands)"),
    "SENTIMENT": ("UMCSENT", "U. Michigan consumer sentiment"),
    "M2": ("M2SL", "M2 money supply (billions $)"),
    "HOUSING": ("HOUST", "Housing starts (thousands, annualized)"),
    "RETAIL": ("RSAFS", "Retail & food services sales (millions $)"),
    "INDPROD": ("INDPRO", "Industrial production index"),
}

EQS_PRESETS = {
    "GAINERS": "day_gainers",
    "LOSERS": "day_losers",
    "ACTIVE": "most_actives",
    "SHORTED": "most_shorted_stocks",
}


def _f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}"


def _big(x):
    if x is None:
        return "—"
    sign, a = ("-" if x < 0 else ""), abs(x)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if a >= div:
            return f"{sign}${a / div:,.2f}{unit}"
    return f"{sign}${a:,.0f}"


def _parse_num(s: str) -> float:
    s = s.upper().replace(",", "")
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    if s and s[-1] in mult:
        return float(s[:-1]) * mult[s[-1]]
    return float(s)


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# series that also get "% vs prior period" and "% vs year ago" columns (value = periods per year, for reference)
FRED_YOY = {"CPIAUCSL": 12, "PAYEMS": 12, "M2SL": 12, "INDPRO": 12, "RSAFS": 12, "HOUST": 12, "GDPC1": 4}


def _fred_series(series_id: str, years: int = 6) -> pd.DataFrame:
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    last_err = None
    for params in ({"id": series_id, "cosd": start}, {"id": series_id}):   # recent data first, then the full series
        try:
            resp = requests.get(FRED_CSV, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if df.shape[1] < 2:
                raise ValueError("unexpected response format")
            df = df.iloc[:, :2].copy()
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            if df.empty:
                raise ValueError("no observations returned")
            return df.reset_index(drop=True)
        except Exception as e:
            last_err = e
    raise ValueError(f"Couldn't load FRED series '{series_id}' ({type(last_err).__name__}: {last_err}). "
                     "Check your internet connection and the series ID.")


def _fred_table(series_id: str, name: str) -> dict:
    df = _fred_series(series_id)
    s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"]))
    s = s[~s.index.duplicated(keep="last")]
    yoy = series_id in FRED_YOY
    vals, idx = s.tolist(), list(s.index)
    rows = []
    for i in range(len(vals) - 1, max(len(vals) - 13, -1), -1):
        v, prev = vals[i], (vals[i - 1] if i > 0 else None)
        row = [idx[i].strftime("%Y-%m-%d"), _f(v, 3), "—" if prev is None else _f(v - prev, 3)]
        if yoy:
            ago = s.get(idx[i] - pd.DateOffset(years=1))   # same month last year, matched by date (survives missing months)
            row.append("—" if prev in (None, 0) else f"{(v / prev - 1) * 100:+.2f}%")
            row.append("—" if ago is None or pd.isna(ago) or ago == 0 else f"{(v / ago - 1) * 100:+.2f}%")
        rows.append(row)
    cols = ["Date", "Value", "Change vs prior"] + (["% vs prior period", "% vs year ago"] if yoy else [])
    note = "Source: FRED (St. Louis Fed)."
    if yoy:
        note += " The '% vs year ago' column is the annual rate of change (for CPI, the headline inflation rate)."
    return {"title": f"ECOS — {name} ({series_id})", "columns": cols, "rows": rows, "note": note}


def cmd_ecos(args: list[str]) -> dict:
    if not args:
        return {
            "title": "ECOS — Economic indicators",
            "columns": ["Code", "Indicator", "FRED ID"],
            "rows": [[k, v[1], v[0]] for k, v in FRED_SERIES.items()],
            "note": "Type ECOS <code> (e.g. ECOS CPI). For any other FRED series: FRED <series id>.",
        }
    key = args[0].upper()
    if key not in FRED_SERIES:
        raise ValueError(f"Unknown ECOS code '{key}'. Type ECOS alone to see the list.")
    sid, name = FRED_SERIES[key]
    return _fred_table(sid, name)


# code: (label, Yahoo screener field or None, display kind)
EQS_FIELDS = {
    "TPE": ("Trailing P/E", "peratio.lasttwelvemonths", "num"),
    "FPE": ("Forward P/E", None, "num"),
    "PS": ("P/S", "lastclosemarketcaptotalrevenue.lasttwelvemonths", "num"),
    "PB": ("P/B", "lastclosepricebookvalue.lasttwelvemonths", "num"),
    "REV": ("Revenue", "totalrevenues.lasttwelvemonths", "money"),
    "FCF": ("Free cash flow", None, "money"),
    "DEBT": ("Total debt", "totaldebt.lasttwelvemonths", "money"),
    "MCAP": ("Mkt cap", "intradaymarketcap", "money"),
    "PRICE": ("Price", "intradayprice", "num"),
    "CHG": ("Day chg %", "percentchange", "pct"),
    "YTD": ("YTD %", None, "pct"),
    "VOL": ("Volume", "dayvolume", "int"),
    "PEG": ("PEG", None, "num"),
    "GRW3": ("3Y EPS growth (past)", None, "pctu"),
    "GRW5": ("5Y EPS growth (analyst)", None, "pctu"),
    "ROIC": ("Return on capital", None, "pctu"),
    "WACC": ("Cost of capital", None, "pctu"),
}
# fields that need a few extra Yahoo calls per stock (statements, growth estimates): checked last, shown for fewer rows
EQS_COSTLY = {"PEG", "GRW3", "GRW5", "ROIC", "WACC"}
EQS_ALIASES = {"PE": "TPE", "TRAILINGPE": "TPE", "FORWARDPE": "FPE", "FWDPE": "FPE", "PRICESALES": "PS",
               "PRICEBOOK": "PB", "REVENUE": "REV", "SALES": "REV", "FREECASHFLOW": "FCF", "MARKETCAP": "MCAP",
               "CHANGE": "CHG", "VOLUME": "VOL", "PX": "PRICE", "PEGRATIO": "PEG", "GROWTH": "GRW5", "GRW": "GRW5", "LTG": "GRW5", "G3": "GRW3", "G5": "GRW5",
               "COC": "WACC", "COSTOFCAPITAL": "WACC", "ROC": "ROIC", "RETURNONCAPITAL": "ROIC"}
EQS_PHRASES = [
    (r"TRAILING\s*P/?E", "TPE"), (r"(?:FORWARD|FWD)\s*P/?E", "FPE"), (r"PRICE\s*(?:/|TO)\s*SALES|P/S", "PS"),
    (r"PRICE\s*(?:/|TO)\s*BOOK|P/B", "PB"), (r"FREE\s*CASH\s*FLOW", "FCF"), (r"MARKET\s*CAP(?:ITALIZATION)?", "MCAP"),
    (r"TOTAL\s*DEBT", "DEBT"), (r"YEAR\s*TO\s*DATE", "YTD"), (r"P/E", "TPE"),
    (r"PEG\s*RATIO", "PEG"), (r"COST\s*OF\s*CAPITAL", "WACC"), (r"RETURN\s*ON\s*(?:INVESTED\s*)?CAPITAL", "ROIC"),
    (r"3\s*-?\s*5\s*Y(?:EAR|R)?\s*GROWTH", "GRW5"), (r"3\s*-?\s*Y(?:EAR|R)?\s*GROWTH", "GRW3"),
    (r"(?:5\s*-?\s*Y(?:EAR|R)?\s*|ANALYST\s*)GROWTH", "GRW5"),
]
EQS_SERVER_OK = {"MCAP", "PRICE", "CHG", "VOL", "TPE", "PS", "PB", "REV", "DEBT"}
EQS_SERVER_SAFE = {"MCAP", "PRICE", "CHG", "VOL", "TPE"}
EQS_QUOTE_KEYS = {"MCAP": "marketCap", "PRICE": "regularMarketPrice", "CHG": "regularMarketChangePercent",
                  "VOL": "regularMarketVolume", "TPE": "trailingPE", "FPE": "forwardPE", "PB": "priceToBook"}
EQS_INFO_KEYS = {"PS": "priceToSalesTrailing12Months", "REV": "totalRevenue", "FCF": "freeCashflow",
                 "DEBT": "totalDebt", "TPE": "trailingPE", "FPE": "forwardPE", "PB": "priceToBook", "MCAP": "marketCap"}
SECTORS = ["Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive", "Energy",
           "Financial Services", "Healthcare", "Industrials", "Real Estate", "Technology", "Utilities"]
SECTOR_ALIASES = {"materials": "Basic Materials", "mining": "Basic Materials", "telecom": "Communication Services",
                  "communications": "Communication Services", "discretionary": "Consumer Cyclical",
                  "consumerdiscretionary": "Consumer Cyclical", "staples": "Consumer Defensive",
                  "consumerstaples": "Consumer Defensive", "financials": "Financial Services",
                  "financial": "Financial Services", "health": "Healthcare", "tech": "Technology",
                  "industrial": "Industrials"}
EQS_CMP = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge,
           "=": lambda a, b: math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)}
EQS_OPS = {"<": "lt", ">": "gt", "<=": "lte", ">=": "gte", "=": "eq"}


def _eqs_sector(val: str) -> str:
    key = re.sub(r"[^a-z]", "", val.lower())
    for s in SECTORS:
        if re.sub(r"[^a-z]", "", s.lower()) == key:
            return s
    if key in SECTOR_ALIASES:
        return SECTOR_ALIASES[key]
    raise ValueError(f"Unknown sector '{val}'. Use one of: " + ", ".join(SECTORS) +
                     " (shortcuts: MATERIALS, TECH, FINANCIALS, STAPLES, DISCRETIONARY, TELECOM).")


def _eqs_normalize(text: str) -> str:
    t = text.upper()
    for pat, rep in EQS_PHRASES:
        t = re.sub(r"(?<![A-Z0-9_])(?:" + pat + r")(?![A-Z0-9_])", rep, t)
    for name in SECTORS:
        t = re.sub(r"(?<![A-Z0-9_])" + name.upper().replace(" ", r"\s+") + r"(?![A-Z0-9_])", name.upper().replace(" ", "_"), t)
    return re.sub(r"\s*(<=|>=|<|>|=)\s*", r"\1", t)


def _eqs_parse(args: list[str]) -> list[dict]:
    conds = []
    for tok in re.split(r"[\s,]+", _eqs_normalize(" ".join(args)).strip()):
        if not tok or tok == "AND":
            continue
        m = re.match(r"^([A-Z][A-Z0-9_]*)(<=|>=|<|>|=)(.+)$", tok)
        if not m:
            raise ValueError(f"Couldn't read '{tok}'. Examples: TPE<20, TPE<FPE, SECTOR=Technology")
        lhs, op, rhs = m.groups()
        lhs = EQS_ALIASES.get(lhs, lhs)
        if lhs == "SECTOR":
            if op != "=":
                raise ValueError("SECTOR only works with =, e.g. SECTOR=Technology")
            conds.append({"lhs": "SECTOR", "sector": _eqs_sector(rhs)})
            continue
        if lhs not in EQS_FIELDS:
            raise ValueError(f"Unknown filter '{lhs}'. Use: " + ", ".join(EQS_FIELDS) + ", SECTOR.")
        fm = re.match(r"^([A-Z][A-Z0-9_]*?)(?:\*([0-9.]+))?$", rhs)
        other = EQS_ALIASES.get(fm.group(1), fm.group(1)) if fm else None
        if other in EQS_FIELDS:
            try:
                mult = float(fm.group(2) or 1)
            except ValueError:
                raise ValueError(f"Couldn't read the multiplier in '{tok}'.")
            conds.append({"lhs": lhs, "op": op, "rhs": other, "mult": mult})
        else:
            try:
                conds.append({"lhs": lhs, "op": op, "num": _parse_num(rhs)})
            except ValueError:
                raise ValueError(f"Couldn't read the value in '{tok}'. Numbers can end in K, M, B or T.")
    if not conds:
        raise ValueError("Add at least one filter. Type EQS alone to see examples.")
    return conds


def _eqs_num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _eqs_fmt(code: str, v) -> str:
    if v is None:
        return "—"
    kind = EQS_FIELDS[code][2]
    if kind == "money":
        return _big(v)
    if kind == "int":
        return f"{int(v):,}"
    if kind == "pct":
        return f"{v:+.2f}%"
    if kind == "pctu":
        return f"{v:.2f}%"
    return _f(v)


# ---- PEG / analyst growth / cost of capital / return on capital -------------------------------------
EQUITY_RISK_PREMIUM = 5.0          # percent, assumption used for the cost of equity (CAPM)
_EXTRAS_CACHE: dict = {}
_EXTRAS_LOCK = threading.Lock()
_RF_CACHE: dict = {}


def _risk_free() -> float:
    """10-year Treasury yield in percent (Yahoo ^TNX), cached for an hour; 4.0 if unavailable."""
    hit = _RF_CACHE.get("rf")
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    rf = 4.0
    try:
        c = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
        if len(c) and 0 < float(c.iloc[-1]) < 25:
            rf = float(c.iloc[-1])
    except Exception:
        pass
    _RF_CACHE["rf"] = (time.time(), rf)
    return rf


def _analyst_growth(tk) -> dict:
    """Analyst long-term (3-5 year) EPS growth, in percent per year, from Yahoo's growth estimates.
    Returns {"stock": x, "industry": x, "sector": x, "index": x} with None where unavailable."""
    out = {"stock": None, "industry": None, "sector": None, "index": None}
    try:
        df = tk.growth_estimates
    except Exception:
        return out
    if df is None or getattr(df, "empty", True):
        return out
    row = None
    for idx in df.index:
        s = str(idx).lower()
        if "ltg" in s or "5y" in s or "next 5" in s or "long" in s:
            row = df.loc[idx]
            break
    if row is None:
        return out
    for key in out:
        for col in df.columns:
            if key in str(col).lower():
                v = _eqs_num(row[col])
                out[key] = None if v is None else v * 100
                break
    return out


def _stmt_val(df, names, col=0):
    s = _pick(df, *names)
    if s is None or len(s) <= col:
        return None
    return _eqs_num(s.iloc[col])


def _eps_cagr_3y(inc):
    """3-year compound growth in diluted EPS from the annual statements (latest year vs three years earlier),
    in percent per year. None if either EPS is missing or not positive (growth isn't meaningful then)."""
    try:
        if inc is None or inc.empty:
            return None
        df = inc.sort_index(axis=1, ascending=False)
        s = _pick(df, "Diluted EPS", "Basic EPS")
        if s is None or len(s) < 4:
            return None
        now, then = _eqs_num(s.iloc[0]), _eqs_num(s.iloc[3])
        if now is None or then is None or now <= 0 or then <= 0:
            return None
        return ((now / then) ** (1 / 3) - 1) * 100
    except Exception:
        return None


def _eqs_extras(sym: str) -> dict:
    """{"GRW3","GRW5","PEG","ROIC","WACC"} for one stock (percent values; None where it can't be worked out). Cached 30 min.

    GRW5 analyst long-term (next 5 years) EPS growth (Yahoo growth estimates). Yahoo has no analyst 3-year forecast.
    GRW3 3-year EPS growth that has actually happened (CAGR of annual diluted EPS)
    PEG  forward P/E divided by GRW5 (falls back to Yahoo's own PEG when growth is missing)
    ROIC NOPAT / average invested capital, from the last two annual statements: NOPAT = EBIT x (1 - tax rate)
    WACC CAPM cost of equity (10Y Treasury + beta x 5% premium) blended with the after-tax cost of debt
    """
    now = time.time()
    with _EXTRAS_LOCK:
        hit = _EXTRAS_CACHE.get(sym)
        if hit and now - hit[0] < 1800:
            return hit[1]
    out = {"GRW3": None, "GRW5": None, "PEG": None, "ROIC": None, "WACC": None}
    try:
        tk = yf.Ticker(sym)
        try:
            info = tk.get_info() or {}
        except Exception:
            info = {}
        out["GRW5"] = _analyst_growth(tk)["stock"]
        fpe = _eqs_num(info.get("forwardPE"))
        if fpe and fpe > 0 and out["GRW5"] and out["GRW5"] > 0:
            out["PEG"] = fpe / out["GRW5"]
        else:
            peg = _eqs_num(info.get("trailingPegRatio")) or _eqs_num(info.get("pegRatio"))
            out["PEG"] = peg if peg and peg > 0 else None
        try:
            inc, bal = tk.income_stmt, tk.balance_sheet
        except Exception:
            inc = bal = None
        out["GRW3"] = _eps_cagr_3y(inc)
        ebit = _stmt_val(inc, ("EBIT", "Operating Income"))
        pretax = _stmt_val(inc, ("Pretax Income",))
        tax_prov = _stmt_val(inc, ("Tax Provision",))
        tax = _stmt_val(inc, ("Tax Rate For Calcs",))
        if tax is None and pretax and tax_prov is not None and pretax > 0:
            tax = tax_prov / pretax
        tax = 0.21 if tax is None else min(max(tax, 0.0), 0.35)
        ic_now = _stmt_val(bal, ("Invested Capital",))
        ic_prev = _stmt_val(bal, ("Invested Capital",), 1)
        if ic_now is None:
            debt_b = _stmt_val(bal, ("Total Debt",))
            eq_b = _stmt_val(bal, ("Stockholders Equity", "Common Stock Equity"))
            cash_b = _stmt_val(bal, ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")) or 0
            if debt_b is not None and eq_b is not None:
                ic_now = debt_b + eq_b - cash_b
        ic = ic_now if not ic_prev or ic_prev <= 0 or not ic_now else (ic_now + ic_prev) / 2
        if ebit is not None and ic and ic > 0:
            out["ROIC"] = ebit * (1 - tax) / ic * 100
        beta = _eqs_num(info.get("beta"))
        mcap = _eqs_num(info.get("marketCap"))
        debt = _eqs_num(info.get("totalDebt")) or _stmt_val(bal, ("Total Debt",)) or 0.0
        if beta is not None and mcap and mcap > 0:
            rf = _risk_free()
            ke = rf + beta * EQUITY_RISK_PREMIUM
            interest = _stmt_val(inc, ("Interest Expense",))
            kd = rf + 2.0
            if interest and debt > 0:
                kd = min(max(abs(interest) / debt * 100, rf), 15.0)
            wd = debt / (debt + mcap)
            out["WACC"] = (1 - wd) * ke + wd * kd * (1 - tax)
    except Exception:
        pass
    with _EXTRAS_LOCK:
        _EXTRAS_CACHE[sym] = (now, out)
    return out


def _extras_map(symbols) -> dict:
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        return dict(zip(symbols, ex.map(_eqs_extras, symbols)))


def _ytd_pct(sym: str):
    """Year-to-date price change in percent: latest close vs the last close of the prior calendar year."""
    try:
        yr = date.today().year
        h = yf.Ticker(sym).history(start=f"{yr - 1}-12-01", end=(date.today() + timedelta(days=1)).isoformat(),
                                   auto_adjust=False)["Close"].dropna()
        prior = h[h.index.year == yr - 1]
        if h.empty or prior.empty:
            return None
        return (float(h.iloc[-1]) / float(prior.iloc[-1]) - 1) * 100
    except Exception:
        return None


def _ytd_map(symbols) -> dict:
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        return dict(zip(symbols, ex.map(_ytd_pct, symbols)))


def cmd_eqs(args: list[str]) -> dict:
    if not args:
        return {
            "title": "EQS — Equity screener",
            "columns": ["Usage", "Example"],
            "rows": [
                ["Presets: GAINERS, LOSERS, ACTIVE, SHORTED", "EQS GAINERS"],
                ["Valuation: TPE (trailing P/E), FPE (forward P/E), PS (price/sales), PB (price/book)", "EQS TPE<20 PS<5 PB<3"],
                ["Financials: REV (revenue), FCF (free cash flow), DEBT (total debt)", "EQS REV>1B FCF>500M DEBT<2B"],
                ["Growth & returns: PEG, GRW3 (3-yr EPS growth, past %), GRW5 (5-yr EPS growth, analyst estimate %), ROIC (return on capital %), WACC (cost of capital %)", "EQS MCAP>20B PEG<1.5 GRW5>10 GRW3>10 ROIC>WACC"],
                ["Market: MCAP, PRICE, VOL, CHG (today's % change), YTD (year-to-date %)", "EQS MCAP>10B YTD>25"],
                ["Compare two fields (optionally times a number)", "EQS TPE<FPE   or   EQS FPE<TPE*0.8"],
                ["Sector (spaces are fine)", "EQS SECTOR=Basic Materials MCAP>10B"],
                ["Operators: < > <= >= =   Numbers accept K / M / B / T", "EQS FCF>-1B"],
            ],
            "note": "US stocks. Sectors: " + ", ".join(SECTORS) + ". Filters Yahoo can't apply itself (FPE, FCF, "
                    "comparing two fields) are checked against the 250 largest matches, so add MCAP or SECTOR to aim the scan. "
                    "PEG, GRW3, GRW5, ROIC and WACC need extra data per stock, so those results show up to 50 rows and can take a "
                    "little longer. WACC = CAPM cost of equity (10Y Treasury + beta x 5%) blended with after-tax cost of debt; "
                    "ROIC = after-tax operating income / invested capital.",
        }
    first = args[0].upper()
    if not (hasattr(yf, "EquityQuery") and hasattr(yf, "screen")):
        raise ValueError("Your yfinance version is too old for EQS. Run: py -m pip install --upgrade yfinance")

    if first in EQS_PRESETS:
        res = yf.screen(EQS_PRESETS[first])
        quotes = (res or {}).get("quotes", [])
        if not quotes:
            return {"title": f"EQS — {first}", "columns": [], "rows": [], "note": "No matches."}
        ytd = _ytd_map(q.get("symbol") for q in quotes if q.get("symbol"))
        rows = [[q.get("symbol", ""), q.get("shortName", ""), _f(q.get("regularMarketPrice")),
                 _eqs_fmt("YTD", ytd.get(q.get("symbol"))), _big(q.get("marketCap")), _f(q.get("trailingPE"), 1),
                 f"{q['regularMarketVolume']:,}" if q.get("regularMarketVolume") else "—"] for q in quotes]
        return {"title": f"EQS — {first}", "columns": ["Ticker", "Name", "Price", "YTD %", "Mkt Cap", "P/E", "Volume"],
                "rows": rows, "note": f"{len(rows)} result(s)."}

    conds = _eqs_parse(args)

    def run(server_ok):
        q_conds, client = [yf.EquityQuery("eq", ["region", "us"])], []
        for c in conds:
            if c["lhs"] == "SECTOR":
                q_conds.append(yf.EquityQuery("eq", ["sector", c["sector"]]))
                continue
            field = EQS_FIELDS[c["lhs"]][1]
            if "num" in c and field and c["lhs"] in server_ok:
                try:
                    q_conds.append(yf.EquityQuery(EQS_OPS[c["op"]], [field, c["num"]]))
                    continue
                except Exception:
                    pass  # field not accepted by this yfinance version: check it ourselves instead
            client.append(c)
        if len(q_conds) == 1:
            q_conds.append(yf.EquityQuery("gt", ["intradayprice", 0]))
        res = yf.screen(yf.EquityQuery("and", q_conds), sortField="intradaymarketcap", sortAsc=False,
                        size=250 if client else 100)
        return res, client

    try:
        res, client = run(EQS_SERVER_OK)
    except Exception:
        try:
            res, client = run(EQS_SERVER_SAFE)  # Yahoo rejected a field: retry with only the long-standing ones
        except Exception as e:
            raise ValueError(f"Yahoo's screener couldn't run that query ({e}).")

    quotes = (res or {}).get("quotes", []) or []
    refs = []
    for c in conds:
        for code in (c.get("lhs"), c.get("rhs")):
            if code in EQS_FIELDS and code not in refs:
                refs.append(code)

    infos = {}
    need = [q["symbol"] for q in quotes if q.get("symbol") and any(
        _eqs_num(q.get(EQS_QUOTE_KEYS.get(code, ""))) is None for code in refs
        if code != "YTD" and code not in EQS_COSTLY)]
    if need:
        def fetch(sym):
            try:
                return sym, yf.Ticker(sym).get_info() or {}
            except Exception:
                return sym, {}
        with ThreadPoolExecutor(max_workers=12) as ex:
            infos = dict(ex.map(fetch, need))

    ytd = _ytd_map(q.get("symbol") for q in quotes if q.get("symbol")) if "YTD" in refs else {}

    extras: dict = {}

    def val(q, code):
        if code == "YTD":
            return ytd.get(q.get("symbol"))
        if code in EQS_COSTLY:
            return (extras.get(q.get("symbol")) or {}).get(code)
        v = _eqs_num(q.get(EQS_QUOTE_KEYS.get(code, "")))
        if v is None:
            v = _eqs_num(infos.get(q.get("symbol"), {}).get(EQS_INFO_KEYS.get(code, "")))
        return v

    def passes(q, costly=False):
        for c in client:
            uses = {c["lhs"], c.get("rhs")} & EQS_COSTLY
            if bool(uses) != costly:
                continue
            left = val(q, c["lhs"])
            right = c["num"] if "num" in c else None
            if "rhs" in c:
                rv = val(q, c["rhs"])
                right = None if rv is None else rv * c["mult"]
            if left is None or right is None or not EQS_CMP[c["op"]](left, right):
                return False
        return True

    hits = [q for q in quotes if passes(q)]
    scanned_note = ""
    if any({c["lhs"], c.get("rhs")} & EQS_COSTLY for c in client):
        pool = hits[:120]          # the costly fields need several calls per stock, so check the 120 largest survivors
        extras.update(_extras_map(q.get("symbol") for q in pool if q.get("symbol")))
        if len(hits) > len(pool):
            scanned_note = f" PEG/GRW3/GRW5/ROIC/WACC were checked on the {len(pool)} largest of {len(hits)} stocks that passed the other filters."
        hits = [q for q in pool if passes(q, costly=True)]
    title = "EQS — " + " ".join(args).upper()
    if not hits:
        return {"title": title, "columns": [], "rows": [],
                "note": "No matches." + (f" Checked {len(quotes)} stocks." if client else "")}
    extra = [c for c in refs if c not in ("PRICE", "MCAP", "YTD")]
    for c in ("PEG", "GRW3", "GRW5", "ROIC", "WACC"):        # always show these five
        if c not in extra:
            extra.append(c)
    shown = hits[:50]
    extras.update(_extras_map(q.get("symbol") for q in shown if q.get("symbol") and q.get("symbol") not in extras))
    ytd.update(_ytd_map(q.get("symbol") for q in shown if q.get("symbol") and q.get("symbol") not in ytd))
    rows = []
    for q in shown:
        rows.append([q.get("symbol", ""), q.get("shortName", ""), _f(_eqs_num(q.get("regularMarketPrice"))),
                     _eqs_fmt("YTD", ytd.get(q.get("symbol"))), _big(_eqs_num(q.get("marketCap")))]
                    + [_eqs_fmt(c, val(q, c)) for c in extra])
    note = f"{len(hits)} match(es)" + (f", showing the largest {len(rows)}" if len(hits) > len(rows) else "") + "." + scanned_note
    if client:
        names = ", ".join(sorted({c["lhs"] for c in client}))
        note += (f" Checked {len(quotes)} stocks for {names}" +
                 (" (the 250 largest; add MCAP or SECTOR to narrow the pool)." if len(quotes) >= 250 else "."))
    elif res and res.get("total") and res["total"] > len(rows):
        note += f" Yahoo reports {res['total']} in total."
    return {"title": title, "columns": ["Ticker", "Name", "Price", "YTD %", "Mkt Cap"] + [EQS_FIELDS[c][0] for c in extra],
            "rows": rows, "note": note}


def cmd_des(ticker: str) -> dict:
    info = yf.Ticker(ticker).get_info() or {}
    if not (info.get("longName") or info.get("shortName")):
        raise ValueError(f"No description found for '{ticker}'.")
    summary = (info.get("longBusinessSummary") or "—")
    rows = [
        ["Name", info.get("longName") or info.get("shortName")],
        ["Sector / Industry", f"{info.get('sector', '—')} / {info.get('industry', '—')}"],
        ["Country", info.get("country", "—")],
        ["Employees", f"{info['fullTimeEmployees']:,}" if info.get("fullTimeEmployees") else "—"],
        ["Market cap", _big(info.get("marketCap"))],
        ["Website", info.get("website", "—")],
        ["Business", summary[:700] + ("…" if len(summary) > 700 else "")],
    ]
    return {"title": f"{ticker} DES — Description", "columns": ["Field", "Value"], "rows": rows, "note": ""}


def cmd_gp(ticker: str) -> dict:
    close = yf.Ticker(ticker).history(period="1y")["Close"].dropna()
    if close.empty:
        raise ValueError(f"No price data for '{ticker}'.")
    last = float(close.iloc[-1])
    rows = [["Last close", f"${_f(last)}"]]
    for label, n in (("1 week", 5), ("1 month", 21), ("3 months", 63), ("6 months", 126), ("1 year", 252)):
        if len(close) > n:
            base = float(close.iloc[-n - 1])
            rows.append([f"Return, {label}", f"{(last / base - 1) * 100:+.2f}%"])
    ytd = close[close.index.year == close.index[-1].year]
    if len(ytd) > 1:
        rows.append(["Return, year to date", f"{(last / float(ytd.iloc[0]) - 1) * 100:+.2f}%"])
    rows.append(["52-week high / low", f"${_f(float(close.max()))} / ${_f(float(close.min()))}"])
    return {"title": f"{ticker} GP — Price performance", "columns": ["Metric", "Value"], "rows": rows,
            "note": "For the full chart, use the Charts & Gaps tab."}


# ---- Calendar-driven "top topics" (used by the Barron's tab and EVTS) ---- #

KEY_ECON = [
    (r"\bFOMC\b|FEDERAL FUNDS|INTEREST RATE DECISION|FED INTEREST", 100),
    (r"\bCPI\b|CONSUMER PRICE", 95),
    (r"NONFARM|\bPAYROLL", 95),
    (r"\bGDP\b", 90),
    (r"\bPCE\b|PERSONAL CONSUMPTION", 90),
    (r"POWELL|FED CHAIR|\bMINUTES\b", 85),
    (r"UNEMPLOYMENT RATE|EMPLOYMENT CHANGE|\bADP\b", 85),
    (r"RETAIL SALES", 80),
    (r"\bPPI\b|PRODUCER PRICE", 80),
    (r"\bISM\b|\bPMI\b", 75),
    (r"JOLTS|JOB OPENINGS", 70),
    (r"CONSUMER CONFIDENCE|CONSUMER SENTIMENT|MICHIGAN", 65),
    (r"DURABLE GOODS|HOUSING STARTS|EXISTING HOME|NEW HOME", 55),
    (r"JOBLESS CLAIMS", 50),
]
TIMING = {"AMC": "after close", "BMO": "before open", "TAS": "time not supplied", "TNS": "time not supplied"}
_TOPICS_CACHE = {"t": 0.0, "data": None}


def _s(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def _find_col(df, *keys, exclude=()):
    for c in df.columns:
        lc = str(c).lower()
        if all(k in lc for k in keys) and not any(x in lc for x in exclude):
            return c
    return None


def _clean_co(name: str) -> str:
    return re.sub(r"[,\s]+(Inc|Corp|Corporation|Co|Ltd|plc|PLC|Limited|LLC)\.?$", "", name.strip())


def _as_ny(series):
    try:
        s = pd.to_datetime(series, errors="coerce")
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("America/New_York")
    except Exception:
        s = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("America/New_York")
    return s


def _paged(fn, limit=100, max_pages=5, **kw) -> pd.DataFrame:
    frames = []
    for p in range(max_pages):
        df = fn(limit=limit, offset=p * limit, **kw)
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < limit:
            break
    return pd.concat(frames) if frames else pd.DataFrame()


def _calendars(days: int):
    if not hasattr(yf, "Calendars"):
        raise ValueError("Your yfinance version has no calendar support. Run: py -m pip install --upgrade yfinance")
    start = date.today()
    return yf.Calendars(start=start, end=start + timedelta(days=days))


def _econ_events(cal) -> list[dict]:
    df = _paged(cal.get_economic_events_calendar)
    if df.empty:
        return []
    ev_c = _find_col(df, "event", exclude=("time", "date"))
    t_c = _find_col(df, "time") or _find_col(df, "date")
    reg_c = _find_col(df, "region")
    exp_c, pri_c, act_c = _find_col(df, "expected"), (_find_col(df, "last") or _find_col(df, "prior")), _find_col(df, "actual")
    if t_c is None:
        return []
    stamps = _as_ny(df[t_c])
    out = []
    for i in range(len(df)):
        ts = stamps.iloc[i]
        if pd.isna(ts):
            continue
        row = df.iloc[i]
        name = _s(row[ev_c]) if ev_c else _s(df.index[i])
        if not name:
            continue
        out.append({
            "dt": ts, "date": ts.date(), "event": name,
            "region": _s(row[reg_c]) if reg_c else "",
            "expected": _s(row[exp_c]) if exp_c else "", "prior": _s(row[pri_c]) if pri_c else "",
            "actual": _s(row[act_c]) if act_c else "",
        })
    return out


def _earnings_events(cal) -> list[dict]:
    df = _paged(cal.get_earnings_calendar, market_cap=2e9)
    if df.empty:
        return []
    name_c, cap_c = _find_col(df, "company"), _find_col(df, "cap")
    date_c, time_c, est_c = _find_col(df, "date"), _find_col(df, "timing"), _find_col(df, "estimate")
    if date_c is None:
        return []
    stamps = _as_ny(df[date_c])
    out = []
    for i in range(len(df)):
        ts = stamps.iloc[i]
        if pd.isna(ts):
            continue
        row = df.iloc[i]
        sym = _s(df.index[i])
        name = _clean_co(_s(row[name_c])) if name_c else sym
        cap = pd.to_numeric(row[cap_c], errors="coerce") if cap_c else np.nan
        score = 45 + 12 * math.log10(cap / 1e9) if cap and not pd.isna(cap) and cap > 0 else 40
        bits = [sym, TIMING.get(_s(row[time_c]).upper(), _s(row[time_c])) if time_c else ""]
        if est_c and _s(row[est_c]):
            bits.append(f"EPS est {_s(row[est_c])}")
        out.append({"date": ts.date(), "title": f"{name or sym} earnings", "kind": "earnings",
                    "detail": " · ".join(b for b in bits if b), "score": score})
    return out


def _ipo_events(cal) -> list[dict]:
    df = _paged(cal.get_ipo_info_calendar, max_pages=1)
    if df.empty:
        return []
    name_c, date_c = _find_col(df, "company") or _find_col(df, "name"), _find_col(df, "date")
    if date_c is None:
        return []
    stamps = _as_ny(df[date_c])
    out = []
    for i in range(len(df)):
        ts = stamps.iloc[i]
        if pd.isna(ts):
            continue
        sym = _s(df.index[i])
        name = _clean_co(_s(df.iloc[i][name_c])) if name_c else sym
        out.append({"date": ts.date(), "title": f"{name or sym} IPO", "kind": "ipo", "detail": sym, "score": 55})
    return out


def build_finance_topics(days: int = 7) -> list[dict]:
    cal = _calendars(days)
    items, errors = [], []
    for label, fn in (("earnings", _earnings_events), ("IPOs", _ipo_events)):
        try:
            items += fn(cal)
        except Exception as e:
            errors.append(f"{label}: {e}")
    try:
        for ev in _econ_events(cal):
            if ev["region"] and ev["region"].upper() not in ("US", "USA", "UNITED STATES"):
                continue
            score = max([w for pat, w in KEY_ECON if re.search(pat, ev["event"].upper())] or [0])
            if score:
                detail = f"Expected {ev['expected']}" if ev["expected"] else ""
                items.append({"date": ev["date"], "title": ev["event"], "kind": "economic",
                              "detail": detail, "score": score})
    except Exception as e:
        errors.append(f"economic events: {e}")
    if not items and errors:
        raise ValueError("Couldn't load Yahoo Finance's calendars (" + "; ".join(errors) + ")")

    start = date.today()
    result = []
    for i in range(days):
        d = start + timedelta(days=i)
        seen, picks = set(), []
        for it in sorted((x for x in items if x["date"] == d), key=lambda x: -x["score"]):
            if it["title"].lower() in seen:
                continue
            seen.add(it["title"].lower())
            picks.append({k: it[k] for k in ("title", "kind", "detail")})
            if len(picks) == 5:
                break
        result.append({"date": d.isoformat(), "topics": picks})
    return result


def get_finance_topics(force: bool = False) -> dict:
    if not force and _TOPICS_CACHE["data"] and time.time() - _TOPICS_CACHE["t"] < 1800:
        return _TOPICS_CACHE["data"]
    data = {"days": build_finance_topics(7)}
    _TOPICS_CACHE.update(t=time.time(), data=data)
    return data


# ---- Bloomberg-style functions ---- #

NEWS_FEEDS = [
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://feeds.marketwatch.com/marketwatch/topstories",
]
INDICES = [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "Nasdaq Composite"), ("^RUT", "Russell 2000"),
           ("^VIX", "VIX"), ("^FTSE", "FTSE 100"), ("^GDAXI", "DAX"), ("^FCHI", "CAC 40"),
           ("^N225", "Nikkei 225"), ("^HSI", "Hang Seng"), ("000001.SS", "Shanghai Composite"), ("^BSESN", "BSE Sensex")]
FX_PAIRS = [("EURUSD=X", "EUR/USD"), ("GBPUSD=X", "GBP/USD"), ("USDJPY=X", "USD/JPY"), ("USDCAD=X", "USD/CAD"),
            ("AUDUSD=X", "AUD/USD"), ("USDCHF=X", "USD/CHF"), ("USDMXN=X", "USD/MXN"), ("USDCNY=X", "USD/CNY"),
            ("DX-Y.NYB", "US Dollar Index")]
COMMODITIES = [("CL=F", "WTI crude"), ("BZ=F", "Brent crude"), ("NG=F", "Natural gas"), ("GC=F", "Gold"),
               ("SI=F", "Silver"), ("HG=F", "Copper"), ("ZC=F", "Corn"), ("ZW=F", "Wheat")]
YIELD_TENORS = [("1M", "DGS1MO"), ("3M", "DGS3MO"), ("6M", "DGS6MO"), ("1Y", "DGS1"), ("2Y", "DGS2"), ("3Y", "DGS3"),
                ("5Y", "DGS5"), ("7Y", "DGS7"), ("10Y", "DGS10"), ("20Y", "DGS20"), ("30Y", "DGS30")]


def _table(title, columns, rows, note="", **extra):
    return {"title": title, "columns": columns, "rows": rows, "note": note, **extra}


def _px_table(title, pairs, digits=2):
    def one(p):
        sym, name = p
        try:
            c = yf.Ticker(sym).history(period="5d")["Close"].dropna()
            last = float(c.iloc[-1])
            if len(c) < 2:
                return [name, sym, _f(last, digits), "—", "—"]
            prev = float(c.iloc[-2])
            return [name, sym, _f(last, digits), _f(last - prev, digits), f"{(last / prev - 1) * 100:+.2f}%"]
        except Exception:
            return [name, sym, "—", "—", "—"]
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, pairs))
    return _table(title, ["Name", "Symbol", "Last", "Chg", "Chg %"], rows, "Prices from Yahoo Finance (may be delayed).")


def _quote_rows(quotes):
    return [[q.get("symbol", ""), q.get("shortName", ""), _f(q.get("regularMarketPrice")),
             (_f(q.get("regularMarketChangePercent")) + "%") if q.get("regularMarketChangePercent") is not None else "—",
             _big(q.get("marketCap")),
             f"{q['regularMarketVolume']:,}" if q.get("regularMarketVolume") else "—"] for q in quotes]


def cmd_most(args):
    if args and args[0] == "ACTIVE" or not args:
        res = yf.screen("most_actives")
        return _table("MOST — Most active", ["Ticker", "Name", "Price", "Chg %", "Mkt Cap", "Volume"],
                      _quote_rows((res or {}).get("quotes", [])[:25]))
    raise ValueError("Usage: MOST")


def cmd_mov(args):
    rows = []
    for side, name in (("Gainer", "day_gainers"), ("Loser", "day_losers")):
        for r in _quote_rows((yf.screen(name) or {}).get("quotes", [])[:10]):
            rows.append([side] + r)
    return _table("MOV — Market movers", ["Side", "Ticker", "Name", "Price", "Chg %", "Mkt Cap", "Volume"], rows)


def cmd_news(args):
    if args:
        return cmd_cn(args[0])
    items = []
    for url in NEWS_FEEDS:
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            items = ET.fromstring(r.content).findall("./channel/item")
            if items:
                break
        except Exception:
            items = []
    if not items:
        raise ValueError("Couldn't load the MarketWatch headlines feed right now. Try again in a minute.")
    rows = []
    for it in items[:20]:
        pub = it.findtext("pubDate") or ""
        try:
            pub = parsedate_to_datetime(pub).astimezone().strftime("%b %d %I:%M %p")
        except Exception:
            pass
        rows.append([pub, (it.findtext("title") or "").strip(), (it.findtext("link") or "").strip()])
    return _table("NEWS — Top stories", ["Time", "Headline", "Link"], rows,
                  "Headlines from MarketWatch's public RSS feed (Dow Jones). NEWS <TICKER> shows company news.")


def cmd_cn(ticker):
    news = yf.Ticker(ticker).news or []
    rows = []
    for e in news[:15]:
        c = e.get("content", e) if isinstance(e, dict) else {}
        prov = c.get("provider")
        url = c.get("canonicalUrl") or c.get("clickThroughUrl")
        rows.append([
            _s(c.get("pubDate"))[:16].replace("T", " "), _s(c.get("title") or e.get("title")),
            _s(prov.get("displayName") if isinstance(prov, dict) else e.get("publisher")),
            _s(url.get("url") if isinstance(url, dict) else e.get("link")),
        ])
    if not rows:
        raise ValueError(f"No recent news found for '{ticker}'.")
    return _table(f"{ticker} CN — Company news", ["Time", "Headline", "Source", "Link"], rows)


def cmd_hp(ticker, args):
    n = min(max(int(args[0]), 1), 250) if args and args[0].isdigit() else 20
    h = yf.Ticker(ticker).history(period="1y").dropna(subset=["Close"]).tail(n)
    if h.empty:
        raise ValueError(f"No price data for '{ticker}'.")
    rows = [[ts.strftime("%Y-%m-%d"), _f(r["Open"]), _f(r["High"]), _f(r["Low"]), _f(r["Close"]), f"{int(r['Volume']):,}"]
            for ts, r in h.iloc[::-1].iterrows()]
    return _table(f"{ticker} HP — Historical prices (last {len(rows)} sessions)",
                  ["Date", "Open", "High", "Low", "Close", "Volume"], rows, "Add a number for more rows, e.g. AAPL HP 60.")


def cmd_qr(ticker):
    h = yf.Ticker(ticker).history(period="1y").dropna(subset=["Close"])
    if h.empty:
        raise ValueError(f"No price data for '{ticker}'.")
    last = h.iloc[-1]
    rows = [["As of", h.index[-1].strftime("%Y-%m-%d")], ["Last", f"${_f(last['Close'])}"]]
    if len(h) > 1:
        prev = float(h.iloc[-2]["Close"])
        rows += [["Change", f"{last['Close'] - prev:+.2f}"], ["Change %", f"{(last['Close'] / prev - 1) * 100:+.2f}%"]]
    rows += [["Day high / low", f"${_f(last['High'])} / ${_f(last['Low'])}"], ["Volume", f"{int(last['Volume']):,}"],
             ["52-week high / low", f"${_f(float(h['High'].max()))} / ${_f(float(h['Low'].min()))}"]]
    return _table(f"{ticker} QR — Quote", ["Field", "Value"], rows)


def _pick(df, *names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def cmd_fa(ticker):
    tk = yf.Ticker(ticker)
    inc, bal, cf = tk.income_stmt, tk.balance_sheet, tk.cashflow
    if inc is None or inc.empty:
        raise ValueError(f"No financial statements found for '{ticker}'.")
    spec = [
        (inc, ("Total Revenue", "Operating Revenue"), "Revenue", _big), (inc, ("Gross Profit",), "Gross profit", _big),
        (inc, ("Operating Income",), "Operating income", _big), (inc, ("Net Income",), "Net income", _big),
        (inc, ("Diluted EPS",), "Diluted EPS", _f),
        (bal, ("Total Assets",), "Total assets", _big), (bal, ("Total Debt",), "Total debt", _big),
        (bal, ("Stockholders Equity", "Common Stock Equity"), "Shareholders' equity", _big),
        (cf, ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"), "Operating cash flow", _big),
        (cf, ("Free Cash Flow",), "Free cash flow", _big),
    ]
    years = [f"FY{str(c)[:4]}" for c in inc.columns[:4]]
    rows = []
    for df, names, label, fmtf in spec:
        s = _pick(df, *names)
        if s is None:
            continue
        vals = [fmtf(None if pd.isna(v) else float(v)) for v in s.iloc[:4]]
        rows.append([label] + vals + ["—"] * (len(years) - len(vals)))
    return _table(f"{ticker} FA — Financial analysis (annual)", ["Metric"] + years, rows, "Source: Yahoo Finance statements.")


def cmd_anr(ticker):
    tk = yf.Ticker(ticker)
    rows = []
    try:
        t = tk.get_analyst_price_targets() or {}
        for k, label in (("current", "Current price"), ("low", "Target low"), ("mean", "Target mean"),
                         ("median", "Target median"), ("high", "Target high")):
            if t.get(k) is not None:
                rows.append([label, f"${_f(float(t[k]))}"])
    except Exception:
        pass
    try:
        r = tk.get_recommendations_summary()
        if r is not None and not r.empty:
            row = r.iloc[0]
            for k, label in (("strongBuy", "Strong buy"), ("buy", "Buy"), ("hold", "Hold"), ("sell", "Sell"), ("strongSell", "Strong sell")):
                if k in r.columns:
                    rows.append([f"Analysts: {label}", _s(row[k])])
    except Exception:
        pass
    try:
        g = _analyst_growth(tk)
        for k, label in (("stock", f"Analyst 5Y EPS growth, forecast ({ticker})"), ("industry", "  Industry average"),
                         ("sector", "  Sector average"), ("index", "  S&P 500")):
            if g.get(k) is not None:
                rows.append([label, f"{g[k]:.2f}% per year"])
    except Exception:
        pass
    try:
        g3 = _eps_cagr_3y(tk.income_stmt)
        if g3 is not None:
            rows.append([f"EPS growth, past 3Y ({ticker})", f"{g3:.2f}% per year"])
    except Exception:
        pass
    if not rows:
        raise ValueError(f"No analyst data found for '{ticker}'.")
    return _table(f"{ticker} ANR — Analyst recommendations", ["Field", "Value"], rows)


def cmd_ern(ticker):
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=12)
    except Exception:
        raise ValueError(f"Couldn't load earnings dates for '{ticker}'.")
    if df is None or df.empty:
        raise ValueError(f"No earnings history found for '{ticker}'.")
    est_c, rep_c, sur_c = _find_col(df, "estimate"), _find_col(df, "reported"), _find_col(df, "surprise")
    rows = [[str(df.index[i])[:10], _s(df.iloc[i][est_c]) if est_c else "", _s(df.iloc[i][rep_c]) if rep_c else "",
             _s(df.iloc[i][sur_c]) if sur_c else ""] for i in range(len(df))]
    return _table(f"{ticker} ERN — Earnings", ["Date", "EPS estimate", "Reported EPS", "Surprise %"], rows,
                  "Blank reported EPS = upcoming.")


def cmd_dvd(ticker):
    s = yf.Ticker(ticker).dividends
    if s is None or s.empty:
        raise ValueError(f"No dividend history found for '{ticker}'.")
    rows = [[ts.strftime("%Y-%m-%d"), f"${v:.4f}"] for ts, v in s.tail(12).iloc[::-1].items()]
    return _table(f"{ticker} DVD — Dividends (ex-dates)", ["Ex-date", "Amount / share"], rows)


def cmd_hds(ticker):
    df = yf.Ticker(ticker).institutional_holders
    if df is None or df.empty:
        raise ValueError(f"No holder data found for '{ticker}'.")
    h_c, s_c, p_c, v_c = _find_col(df, "holder"), _find_col(df, "shares"), _find_col(df, "pct", "held") or _find_col(df, "out"), _find_col(df, "value")
    rows = []
    for i in range(min(10, len(df))):
        r = df.iloc[i]
        pct = pd.to_numeric(r[p_c], errors="coerce") if p_c else np.nan
        pct_s = "—" if pd.isna(pct) else (f"{pct * 100:.2f}%" if pct <= 1 else f"{pct:.2f}%")
        sh = pd.to_numeric(r[s_c], errors="coerce") if s_c else np.nan
        val = pd.to_numeric(r[v_c], errors="coerce") if v_c else np.nan
        rows.append([_s(r[h_c]) if h_c else "", "—" if pd.isna(sh) else f"{int(sh):,}", pct_s, "—" if pd.isna(val) else _big(float(val))])
    return _table(f"{ticker} HDS — Top institutional holders", ["Holder", "Shares", "% held", "Value"], rows)


def cmd_omon(ticker, args):
    tk = yf.Ticker(ticker)
    exps = list(tk.options or [])
    if not exps:
        raise ValueError(f"No listed options found for '{ticker}'.")
    exp = args[0] if args and re.match(r"^\d{4}-\d{2}-\d{2}$", args[0]) else exps[0]
    if exp not in exps:
        raise ValueError("That expiration isn't listed. Available: " + ", ".join(exps[:8]))
    px = float(tk.history(period="5d")["Close"].dropna().iloc[-1])
    ch = tk.option_chain(exp)
    cols = ["strike", "lastPrice", "bid", "ask", "volume"]
    m = ch.calls[cols].merge(ch.puts[cols], on="strike", how="outer", suffixes=("_c", "_p")).sort_values("strike")
    m = m.iloc[(m["strike"] - px).abs().argsort()[:12]].sort_values("strike")
    rows = [[_f(r["strike"]), _f(r["lastPrice_c"]), _f(r["bid_c"]), _f(r["ask_c"]), _f(r["volume_c"], 0),
             _f(r["lastPrice_p"]), _f(r["bid_p"]), _f(r["ask_p"]), _f(r["volume_p"], 0)] for _, r in m.iterrows()]
    return _table(f"{ticker} OMON — Options monitor, expiring {exp} (stock ${_f(px)})",
                  ["Strike", "Call last", "Call bid", "Call ask", "Call vol", "Put last", "Put bid", "Put ask", "Put vol"], rows,
                  "Strikes nearest the current price. Add a date for another expiry, e.g. AAPL OMON 2026-12-18.")


def cmd_eco(args):
    evs = _econ_events(_calendars(7))
    if not (args and args[0] == "ALL"):
        evs = [e for e in evs if not e["region"] or e["region"].upper() in ("US", "USA", "UNITED STATES")]
    evs.sort(key=lambda e: e["dt"])
    if not evs:
        return _table("ECO — Economic calendar", [], [], "No events found for the next 7 days.")
    rows = [[e["dt"].strftime("%a %b %d %I:%M %p"), e["region"], e["event"], e["expected"], e["prior"], e["actual"]]
            for e in evs[:60]]
    return _table("ECO — Economic calendar (next 7 days, ET)", ["When", "Region", "Event", "Expected", "Prior", "Actual"], rows,
                  "US events by default; ECO ALL for every region.")


def cmd_evts():
    rows = []
    for day in get_finance_topics()["days"]:
        for t in day["topics"]:
            rows.append([day["date"], t["kind"], t["title"], t["detail"]])
    return _table("EVTS — Top events, next 7 days", ["Date", "Type", "Event", "Detail"], rows,
                  "Same list as the Barron's tab: earnings, IPOs and key US economic releases.")


def cmd_gc():
    def one(t):
        tenor, sid = t
        try:
            v = _fred_series(sid)["value"].tolist()
            last, prev = v[-1], v[-2]
            month = v[-22] if len(v) > 22 else None
            return [tenor, f"{last:.2f}", f"{(last - prev) * 100:+.0f}", f"{month:.2f}" if month is not None else "—"]
        except Exception:
            return [tenor, "—", "—", "—"]
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one, YIELD_TENORS))
    return _table("GC — US Treasury yield curve", ["Tenor", "Yield %", "Chg (bp)", "~1 month ago %"], rows,
                  "Source: FRED (constant-maturity Treasury yields).")


GF_METRICS = {
    "REV": ("Revenue", "inc", ("Total Revenue", "Operating Revenue"), "money"),
    "GP": ("Gross profit", "inc", ("Gross Profit",), "money"),
    "OI": ("Operating income", "inc", ("Operating Income",), "money"),
    "NI": ("Net income", "inc", ("Net Income",), "money"),
    "EPS": ("Diluted EPS", "inc", ("Diluted EPS",), "eps"),
    "OCF": ("Operating cash flow", "cf", ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"), "money"),
    "FCF": ("Free cash flow", "cf", ("Free Cash Flow",), "money"),
    "ASSETS": ("Total assets", "bal", ("Total Assets",), "money"),
    "DEBT": ("Total debt", "bal", ("Total Debt",), "money"),
    "EQUITY": ("Shareholders' equity", "bal", ("Stockholders Equity", "Common Stock Equity"), "money"),
}
GF_ATTRS = {"inc": ("income_stmt", "quarterly_income_stmt"), "bal": ("balance_sheet", "quarterly_balance_sheet"),
            "cf": ("cashflow", "quarterly_cashflow")}
EE_PERIODS = {"0q": "Current quarter", "+1q": "Next quarter", "0y": "Current year", "+1y": "Next year"}


def cmd_gf(ticker, args):
    quarterly = "Q" in args
    codes = [a for a in args if a not in ("Q", "A")]
    metric = codes[0] if codes else "REV"
    if metric not in GF_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Use one of: {', '.join(GF_METRICS)}. "
                         "Add Q for quarterly, e.g. AAPL GF NI Q.")
    label, stmt, names, kind = GF_METRICS[metric]
    df = getattr(yf.Ticker(ticker), GF_ATTRS[stmt][1 if quarterly else 0])
    s = _pick(df, *names)
    if s is None:
        raise ValueError(f"No {label.lower()} data found for '{ticker}'.")
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"No {label.lower()} data found for '{ticker}'.")
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    labels = [f"{d.strftime('%b')} '{d.strftime('%y')}" if quarterly else f"FY{d.year}" for d in s.index]
    vals = [float(v) for v in s.values]
    if kind == "eps":
        scaled, unit = vals, "$ per share"
    else:
        m = max(abs(v) for v in vals)
        div, unit = (1e9, "$ billions") if m >= 1e9 else (1e6, "$ millions") if m >= 1e6 else (1, "$")
        scaled = [v / div for v in vals]
    rows = []
    for i, (lab, v) in enumerate(zip(labels, vals)):
        chg = "—" if i == 0 or vals[i - 1] <= 0 else f"{(v / vals[i - 1] - 1) * 100:+.1f}%"
        rows.append([lab, _f(v) if kind == "eps" else _big(v), chg])
    period = "quarterly" if quarterly else "annual"
    return _table(f"{ticker} GF — {label} ({period})", ["Period", label, "Change vs prior period"], rows,
                  "Yahoo Finance statements (about 4 years annual, fewer quarters). Metrics: "
                  + ", ".join(GF_METRICS) + ". Add Q for quarterly, e.g. AAPL GF NI Q.",
                  chart={"labels": labels, "values": [round(x, 4) for x in scaled], "unit": unit})


def _gv(row, key):
    if row is None:
        return None
    for k in row.index:
        if str(k).lower() == key.lower():
            return None if pd.isna(row[k]) else row[k]
    return None


def _gnum(v, d=2):
    return "—" if v is None else _f(float(v), d)


def _gfrac(v):
    return "—" if v is None else f"{float(v) * 100:+.1f}%"


def cmd_ee(ticker):
    tk = yf.Ticker(ticker)

    def get(name):
        try:
            df = getattr(tk, name)
            return df if df is not None and not df.empty else None
        except Exception:
            return None

    ee, rev, trend = get("earnings_estimate"), get("revenue_estimate"), get("eps_trend")
    revs, hist = get("eps_revisions"), get("earnings_history")
    if ee is None and rev is None:
        raise ValueError(f"No analyst estimates found for '{ticker}'.")

    def row_of(df, p):
        return df.loc[p] if df is not None and p in df.index else None

    periods = [p for p in EE_PERIODS if row_of(ee, p) is not None or row_of(rev, p) is not None]
    rows = []
    for p in periods:
        e, r = row_of(ee, p), row_of(rev, p)
        rows.append([EE_PERIODS[p], _gnum(_gv(e, "avg")), _gnum(_gv(e, "low")), _gnum(_gv(e, "high")),
                     _gnum(_gv(e, "numberOfAnalysts"), 0), _gnum(_gv(e, "yearAgoEps")), _gfrac(_gv(e, "growth")),
                     _big(_gv(r, "avg")), _gfrac(_gv(r, "growth"))])
    sections = []
    if trend is not None:
        keys = ("current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo")
        sections.append({"title": "EPS trend (how the estimate has moved)",
                         "columns": ["Period", "Current", "7d ago", "30d ago", "60d ago", "90d ago"],
                         "rows": [[EE_PERIODS[p]] + [_gnum(_gv(row_of(trend, p), k)) for k in keys]
                                  for p in EE_PERIODS if row_of(trend, p) is not None]})
    if revs is not None:
        keys = ("upLast7days", "upLast30days", "downLast7Days", "downLast30days")
        sections.append({"title": "Analyst revisions (count)",
                         "columns": ["Period", "Up 7d", "Up 30d", "Down 7d", "Down 30d"],
                         "rows": [[EE_PERIODS[p]] + [_gnum(_gv(row_of(revs, p), k), 0) for k in keys]
                                  for p in EE_PERIODS if row_of(revs, p) is not None]})
    if hist is not None:
        hrows = []
        for i in range(len(hist)):
            r = hist.iloc[i]
            est, act = _gv(r, "epsEstimate"), _gv(r, "epsActual")
            sur = "—" if est in (None, 0) or act is None else f"{(float(act) - float(est)) / abs(float(est)) * 100:+.1f}%"
            hrows.append([str(hist.index[i])[:10], _gnum(est), _gnum(act), sur])
        sections.append({"title": "Earnings history (actual vs estimate)",
                         "columns": ["Quarter", "EPS estimate", "EPS actual", "Surprise"], "rows": hrows})
    return _table(f"{ticker} EE — Earnings estimates",
                  ["Period", "EPS est", "Low", "High", "# analysts", "Year-ago EPS", "EPS growth", "Revenue est", "Rev growth"],
                  rows, "Consensus from Yahoo Finance; analyst coverage varies by company.", sections=sections)


# --------------------------------------------------------------------------- #
# 10K / 10Q: most recent SEC filing, with its reported financials (SEC EDGAR)
# --------------------------------------------------------------------------- #
# The SEC asks every program that downloads from EDGAR to identify itself with a contact email. Put yours here
# (or set the SEC_USER_AGENT environment variable), e.g. "Jane Smith jane@school.edu".
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "StockResearchTerminal contact@example.com")
_SEC_CACHE: dict = {}


def _sec_json(url: str, ttl: int):
    hit = _SEC_CACHE.get(url)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    except requests.RequestException:
        raise ValueError("Couldn't reach SEC EDGAR. Check your internet connection and try again.")
    if r.status_code == 404:
        return None
    if r.status_code in (403, 429):
        raise ValueError("SEC EDGAR refused the request. The SEC requires a contact email in each request: set SEC_USER_AGENT "
                         "near the top of the 10K/10Q section of app.py to something like 'Your Name you@email.com'. "
                         "(If you already did, wait a minute: the SEC also rate-limits heavy use.)")
    r.raise_for_status()
    data = r.json()
    if "companyfacts" in url:                       # these files are several MB: keep only the last few
        for k in [k for k in _SEC_CACHE if "companyfacts" in k][:-3]:
            _SEC_CACHE.pop(k, None)
    _SEC_CACHE[url] = (time.time(), data)
    return data


def _sec_company(ticker: str):
    data = _sec_json("https://www.sec.gov/files/company_tickers.json", 86400) or {}
    t = ticker.upper().replace(".", "-")
    for v in data.values():
        if str(v.get("ticker", "")).upper() == t:
            return int(v["cik_str"]), v.get("title", ticker)
    raise ValueError(f"'{ticker}' isn't in the SEC's company list. 10-K and 10-Q filings exist only for companies that file with "
                     f"the SEC (US-listed companies); ETFs, funds and many foreign issuers don't.")


def _sec_latest(cik: int, form: str) -> dict:
    sub = _sec_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", 900)
    if not sub:
        raise ValueError("The SEC has no filing index for this company.")
    rec = sub["filings"]["recent"]
    for i, f in enumerate(rec["form"]):
        if f == form:
            accn = rec["accessionNumber"][i]
            flat = accn.replace("-", "")
            return {"accn": accn, "filed": rec["filingDate"][i], "period": rec["reportDate"][i] or None, "form": form,
                    "doc": f"https://www.sec.gov/Archives/edgar/data/{cik}/{flat}/{rec['primaryDocument'][i]}",
                    "index": f"https://www.sec.gov/Archives/edgar/data/{cik}/{flat}/{accn}-index.htm"}
    forms = set(rec["form"])
    if form == "10-K" and "20-F" in forms:
        raise ValueError("This company files 20-F annual reports (a foreign issuer), not 10-Ks.")
    if form == "10-Q" and "6-K" in forms and "10-K" not in forms:
        raise ValueError("This company files 6-K reports (a foreign issuer), not 10-Qs.")
    raise ValueError(f"No {form} found in this company's recent SEC filings.")


def _d(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def _sec_entries(facts, tag, unit, accn):
    node = (facts.get("us-gaap") or {}).get(tag)
    return [e for e in (node or {}).get("units", {}).get(unit, []) if e.get("accn") == accn] if node else []


def _dur_val(es, end_target, lo, hi, tol):
    """Value of a flow (income/cash-flow) fact that ends near end_target and spans lo..hi days; the longest span wins."""
    best = None
    for e in es:
        if "start" not in e:
            continue
        try:
            end, days = _d(e["end"]), (_d(e["end"]) - _d(e["start"])).days
        except Exception:
            continue
        if abs((end - end_target).days) <= tol and lo <= days <= hi and (best is None or days > best[0]):
            best = (days, e["val"])
    return None if best is None else float(best[1])


def _inst_val(es, end_target, tol):
    for e in es:
        if "start" in e:
            continue
        try:
            if abs((_d(e["end"]) - end_target).days) <= tol:
                return float(e["val"])
        except Exception:
            continue
    return None


def _inst_prior_fye(es, pe):
    """For a 10-Q: the balance-sheet value at the previous fiscal year-end that the filing compares against."""
    c = [(e["end"], e["val"]) for e in es if "start" not in e and 60 < (pe - _d(e["end"])).days < 330]
    return float(max(c)[1]) if c else None


def _sec_fmt_usd(v):
    return _big(v)


def _sec_fmt_eps(v):
    return "—" if v is None else f"{v:,.2f}"


def _sec_fmt_shs(v):
    if v is None:
        return "—"
    for unit, div in (("B", 1e9), ("M", 1e6)):
        if abs(v) >= div:
            return f"{v / div:,.2f}{unit}"
    return f"{v:,.0f}"


def _sec_fmt_pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def _sec_chg(cur, prior):
    if cur is None or prior is None or prior == 0:
        return "—"
    return f"{(cur - prior) / abs(prior) * 100:+.1f}%"


# label, tags in priority order, unit, formatter
SEC_INCOME = [
    ("Revenue", ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "RevenueFromContractWithCustomerIncludingAssessedTax",
                 "SalesRevenueNet", "SalesRevenueGoodsNet", "SalesRevenueServicesNet"), "USD", _sec_fmt_usd),
    ("Cost of revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"), "USD", _sec_fmt_usd),
    ("Gross profit", ("GrossProfit",), "USD", _sec_fmt_usd),
    ("Research & development", ("ResearchAndDevelopmentExpense",), "USD", _sec_fmt_usd),
    ("Selling, general & admin", ("SellingGeneralAndAdministrativeExpense",), "USD", _sec_fmt_usd),
    ("Operating expenses", ("OperatingExpenses", "CostsAndExpenses"), "USD", _sec_fmt_usd),
    ("Operating income", ("OperatingIncomeLoss",), "USD", _sec_fmt_usd),
    ("Interest expense", ("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"), "USD", _sec_fmt_usd),
    ("Pretax income", ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                       "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"), "USD", _sec_fmt_usd),
    ("Income tax", ("IncomeTaxExpenseBenefit",), "USD", _sec_fmt_usd),
    ("Net income", ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"), "USD", _sec_fmt_usd),
    ("EPS, basic", ("EarningsPerShareBasic",), "USD/shares", _sec_fmt_eps),
    ("EPS, diluted", ("EarningsPerShareDiluted",), "USD/shares", _sec_fmt_eps),
    ("Diluted shares", ("WeightedAverageNumberOfDilutedSharesOutstanding",), "shares", _sec_fmt_shs),
]
SEC_BALANCE = [
    ("Cash & equivalents", ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")),
    ("Short-term investments", ("MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent", "ShortTermInvestments")),
    ("Receivables", ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent")),
    ("Inventory", ("InventoryNet",)),
    ("Total current assets", ("AssetsCurrent",)),
    ("Property, plant & equipment (net)", ("PropertyPlantAndEquipmentNet",)),
    ("Goodwill", ("Goodwill",)),
    ("Total assets", ("Assets",)),
    ("Total current liabilities", ("LiabilitiesCurrent",)),
    ("Short-term debt", ("DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings", "CommercialPaper")),
    ("Long-term debt", ("LongTermDebtNoncurrent", "LongTermDebt", "LongTermNotesPayable")),
    ("Total liabilities", ("Liabilities",)),
    ("Shareholders' equity", ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
]
SEC_CASHFLOW = [
    ("Operating cash flow", ("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
    ("Investing cash flow", ("NetCashProvidedByUsedInInvestingActivities", "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations")),
    ("Financing cash flow", ("NetCashProvidedByUsedInFinancingActivities", "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations")),
    ("Depreciation & amortization", ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet")),
    ("Stock-based compensation", ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense")),
    ("Capital expenditures", ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")),
    ("Dividends paid", ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock")),
    ("Share repurchases", ("PaymentsForRepurchaseOfCommonStock",)),
]


def cmd_filing(ticker: str, form: str) -> dict:
    """Most recent 10-K or 10-Q from SEC EDGAR: the reported income statement, balance sheet and cash flow."""
    cik, name = _sec_company(ticker)
    fil = _sec_latest(cik, form)
    pe = _d(fil["period"]) if fil["period"] else None
    if pe is None:
        raise ValueError("The SEC index doesn't list a period end date for this filing.")
    annual = form == "10-K"
    facts_all = _sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", 3600)
    facts = (facts_all or {}).get("facts") or {}
    accn, prior_end = fil["accn"], pe - timedelta(days=365)
    kind = "10-K (annual report)" if annual else "10-Q (quarterly report)"

    # ---- income statement ----
    def flow(tags, unit):
        for tag in tags:
            es = _sec_entries(facts, tag, unit, accn)
            if not es:
                continue
            if annual:
                cur = _dur_val(es, pe, 340, 380, 4)
                if cur is not None:
                    return cur, _dur_val(es, prior_end, 340, 380, 10), None, None
            else:
                cur = _dur_val(es, pe, 80, 100, 4)
                if cur is not None:
                    return (cur, _dur_val(es, prior_end, 80, 100, 10),
                            _dur_val(es, pe, 101, 290, 4), _dur_val(es, prior_end, 101, 290, 10))
        return None, None, None, None

    inc = {label: flow(tags, unit) for label, tags, unit, _ in SEC_INCOME}
    if inc["Gross profit"][0] is None and inc["Revenue"][0] is not None and inc["Cost of revenue"][0] is not None:
        inc["Gross profit"] = tuple(None if a is None or b is None else a - b for a, b in zip(inc["Revenue"], inc["Cost of revenue"]))
    has_ytd = any(v[2] is not None for v in inc.values())
    inc_cols = ["Income statement", f"{'Year' if annual else 'Quarter'} to {pe}", "Prior year" if annual else "Year-ago quarter", "Change"]
    if has_ytd:
        inc_cols += [f"YTD to {pe}", "Prior YTD"]
    inc_rows = []
    for label, _, _, fmtf in SEC_INCOME:
        c, p, yc, yp = inc[label]
        if c is None and p is None and yc is None:
            continue
        row = [label, fmtf(c), fmtf(p), _sec_chg(c, p)]
        if has_ytd:
            row += [fmtf(yc), fmtf(yp)]
        inc_rows.append(row)

    # ---- balance sheet ----
    def inst(tags):
        for tag in tags:
            es = _sec_entries(facts, tag, "USD", accn)
            cur = _inst_val(es, pe, 4) if es else None
            if cur is not None:
                return cur, (_inst_val(es, prior_end, 10) if annual else _inst_prior_fye(es, pe))
        return None, None

    bal = {label: inst(tags) for label, tags in SEC_BALANCE}
    if bal["Total liabilities"][0] is None and bal["Total assets"][0] is not None and bal["Shareholders' equity"][0] is not None:
        bal["Total liabilities"] = tuple(None if a is None or b is None else a - b for a, b in
                                         zip(bal["Total assets"], bal["Shareholders' equity"]))
    bal_cols = ["Balance sheet", str(pe), "Prior year" if annual else "Prior fiscal year-end", "Change"]
    bal_rows = [[l, _big(c), _big(p), _sec_chg(c, p)] for l, (c, p) in bal.items() if c is not None or p is not None]

    # ---- cash flow (10-Q cash flow statements are year-to-date) ----
    def cflow(tags):
        for tag in tags:
            es = _sec_entries(facts, tag, "USD", accn)
            if not es:
                continue
            for lo, hi in ((340, 380),) if annual else ((101, 290), (80, 100)):
                cur = _dur_val(es, pe, lo, hi, 4)
                if cur is not None:
                    return cur, _dur_val(es, prior_end, lo, hi, 10)
        return None, None

    cf = {label: cflow(tags) for label, tags in SEC_CASHFLOW}
    cfo, capex = cf["Operating cash flow"], cf["Capital expenditures"]
    fcf = tuple(None if a is None or b is None else a - abs(b) for a, b in zip(cfo, capex))
    cf_rows = [[l, _big(c), _big(p), _sec_chg(c, p)] for l, (c, p) in cf.items() if c is not None or p is not None]
    if fcf[0] is not None:
        cf_rows.append(["Free cash flow (operating cash flow minus capex)", _big(fcf[0]), _big(fcf[1]), _sec_chg(fcf[0], fcf[1])])
    cf_label = "Cash flow (year-to-date)" if not annual else "Cash flow"
    cf_cols = [cf_label, "Current period", "Prior year", "Change"]

    # ---- key ratios, worked out from the numbers above ----
    def ratio(a, b):
        return None if a is None or not b else a / b

    def pair(f):
        return f(0), f(1)

    rev, gp, oi, ni = (inc[k] for k in ("Revenue", "Gross profit", "Operating income", "Net income"))
    ca, cl = bal["Total current assets"], bal["Total current liabilities"]
    debt = pair(lambda i: None if bal["Long-term debt"][i] is None and bal["Short-term debt"][i] is None
                else (bal["Long-term debt"][i] or 0) + (bal["Short-term debt"][i] or 0))
    cash = pair(lambda i: None if bal["Cash & equivalents"][i] is None else bal["Cash & equivalents"][i] + (bal["Short-term investments"][i] or 0))
    eq = bal["Shareholders' equity"]
    metrics = []

    def add(label, cur, prior, fmtf, pts=False, usd=False):
        if cur is None and prior is None:
            return
        if cur is None or prior is None:
            chg = "—"
        elif pts:
            chg = f"{(cur - prior) * 100:+.1f} pts"
        elif usd:
            chg = _sec_chg(cur, prior)
        else:
            chg = f"{cur - prior:+.2f}x"
        metrics.append([label, fmtf(cur), fmtf(prior), chg])
    add("Gross margin", ratio(gp[0], rev[0]), ratio(gp[1], rev[1]), _sec_fmt_pct, True)
    add("Operating margin", ratio(oi[0], rev[0]), ratio(oi[1], rev[1]), _sec_fmt_pct, True)
    add("Net margin", ratio(ni[0], rev[0]), ratio(ni[1], rev[1]), _sec_fmt_pct, True)
    add("Free cash flow margin", ratio(fcf[0], rev[0] if annual else (rev[2] or rev[0])),
        ratio(fcf[1], rev[1] if annual else (rev[3] or rev[1])), _sec_fmt_pct, True)
    add("Current ratio", ratio(ca[0], cl[0]), ratio(ca[1], cl[1]), lambda v: "—" if v is None else f"{v:.2f}x")
    add("Total debt", debt[0], debt[1], _big, usd=True)
    add("Net debt (debt minus cash & short-term investments)", None if debt[0] is None or cash[0] is None else debt[0] - cash[0],
        None if debt[1] is None or cash[1] is None else debt[1] - cash[1], _big, usd=True)
    add("Debt / equity", ratio(debt[0], eq[0]), ratio(debt[1], eq[1]), lambda v: "—" if v is None else f"{v:.2f}x")
    metric_cols = ["Key ratios (worked out from the figures above" + ("" if annual else "; margins use the quarter, cash-flow margin the year-to-date") + ")",
                   "Current", "Prior", "Change"]

    fy_fp = None
    for node in (facts.get("us-gaap") or {}).values():
        for es in node.get("units", {}).values():
            for e in es:
                if e.get("accn") == accn and e.get("fp"):
                    fy_fp = (e.get("fy"), e["fp"])
                    break
            if fy_fp:
                break
        if fy_fp:
            break
    period_txt = f"{fy_fp[1]} {fy_fp[0]}, " if fy_fp and not annual else (f"FY{fy_fp[0]}, " if fy_fp else "")

    links = [["Filing document (full text on sec.gov)", fil["doc"]], ["Filing index (exhibits, financial report)", fil["index"]]]
    sections = []
    for title, cols, rows in ((bal_cols[0], bal_cols, bal_rows), (cf_cols[0], cf_cols, cf_rows), (metric_cols[0], metric_cols, metrics)):
        if rows:
            sections.append({"title": title, "columns": cols, "rows": rows})
    sections.append({"title": "Filing links", "columns": ["Document", "Link"], "rows": links})

    note = (f"{name}: {kind} for the period ending {pe} ({period_txt}filed {fil['filed']}). Figures are as reported in the filing's "
            f"XBRL data from SEC EDGAR, in US dollars; \"—\" means the company didn't report that line under a standard tag "
            f"(banks and insurers label many lines differently), so check the filing itself.")
    if not inc_rows and not bal_rows:
        note = (f"{name}: {kind} filed {fil['filed']} for the period ending {pe}. The SEC hasn't published structured financial data for "
                f"this filing yet (it usually appears within a day), so open the filing itself with the links below.")
        return _table(f"{ticker} {form} — {name}", ["Document", "Link"], links, note)
    return _table(f"{ticker} {form} — Most recent {'annual' if annual else 'quarterly'} report", inc_cols, inc_rows, note, sections=sections)


NO_TICKER_FUNCS = {"PORT", "HELP", "EQS", "ECOS", "ECST", "ECO", "FRED", "NEWS", "TOP", "N", "EVTS", "MOST", "MOV",
                   "WEI", "FX", "FXC", "CMDTY", "GC"}
TICKER_FUNCS = {"DES", "GP", "HP", "FA", "ANR", "ERN", "DVD", "HDS", "OMON", "CN", "QR", "GF", "EE", "10K", "10Q"}
SKIP_TOKENS = {"US", "EQUITY", "EQ", "INDEX", "UN", "UQ", "UW", "UA", "UP", "UR"}


def cmd_help():
    rows = [
        ["Equities (type the ticker first, Bloomberg-style: AAPL US EQUITY DES also works)", "", ""],
        ["<T> DES", "Description", "AAPL DES"], ["<T> QR", "Quote", "AAPL QR"],
        ["<T> GP", "Price performance", "AAPL GP"], ["<T> HP [n]", "Historical prices", "AAPL HP 60"],
        ["<T> 10K", "Most recent annual report (10-K) from the SEC: income statement, balance sheet, cash flow, ratios, filing link", "AAPL 10K"],
        ["<T> 10Q", "Most recent quarterly report (10-Q) from the SEC, same layout", "AAPL 10Q"],
        ["<T> FA", "Financial analysis (annual statements)", "AAPL FA"], ["<T> ANR", "Analyst recommendations, targets, 5Y growth forecast & past 3Y EPS growth", "AAPL ANR"],
        ["<T> EE", "Earnings estimates (consensus, trend, revisions, surprises)", "AAPL EE"],
        ["<T> GF [metric] [Q]", "Graph fundamentals: REV GP OI NI EPS OCF FCF ASSETS DEBT EQUITY", "AAPL GF NI Q"],
        ["<T> ERN", "Earnings dates & history", "AAPL ERN"], ["<T> DVD", "Dividend history", "AAPL DVD"],
        ["<T> HDS", "Top institutional holders", "AAPL HDS"], ["<T> OMON [date]", "Options monitor", "AAPL OMON"],
        ["<T> CN", "Company news", "AAPL CN"],
        ["Markets & news", "", ""],
        ["NEWS (or TOP)", "Top market headlines", "NEWS"], ["NEWS <T>", "Company news", "NEWS AAPL"],
        ["EVTS", "Top events next 7 days", "EVTS"], ["MOST", "Most active stocks", "MOST"],
        ["MOV", "Gainers and losers", "MOV"], ["WEI", "World equity indices", "WEI"],
        ["FX (or FXC)", "Currency rates", "FX"], ["CMDTY", "Commodities", "CMDTY"], ["GC", "US Treasury yield curve", "GC"],
        ["Portfolios", "", ""],
        ["PORT [name]", "Build, value and save portfolios: Mkt Val, Tot Rtn, CTR, benchmark ETF, yield/P/E/D-E/ROE, VaR, crash stress tests", "PORT   or   PORT Growth"],
        ["Screening & economics", "", ""],
        ["EQS", "Equity screener: TPE FPE PS PB PEG GRW3 GRW5 ROIC WACC REV FCF DEBT MCAP YTD SECTOR, or compare two fields", "EQS TPE<FPE SECTOR=Technology"], ["ECO", "Economic calendar", "ECO"],
        ["ECOS (or ECST)", "Economic statistics", "ECOS CPI"], ["FRED <id>", "Any FRED series", "FRED DGS30"],
    ]
    return _table("HELP — Functions", ["Function", "What it does", "Example"], rows,
                  "Not affiliated with any commercial terminal; the function names are just familiar shortcuts.")


def run_terminal_command(cmd: str) -> dict:
    toks = [{"10-K": "10K", "10-Q": "10Q"}.get(t, t) for t in cmd.replace("<GO>", " ").upper().split()]
    if len(toks) >= 2 and toks[-1] == "GO":
        toks = toks[:-1]
    if not toks:
        raise ValueError("Type a function. HELP lists them.")
    head, ticker, args = toks[0], None, []
    if head in NO_TICKER_FUNCS:
        fn, args = head, toks[1:]
    elif head in TICKER_FUNCS and not (len(toks) >= 2 and (toks[1] in TICKER_FUNCS or toks[1] in SKIP_TOKENS)):
        if len(toks) < 2:
            raise ValueError(f"{head} needs a ticker, e.g. AAPL {head}")
        fn, ticker, args = head, toks[1], toks[2:]
    else:
        ticker, rest = head, toks[1:]
        while rest and rest[0] in SKIP_TOKENS:
            rest = rest[1:]
        if not rest or rest[0] not in TICKER_FUNCS:
            raise ValueError(f"Unknown function '{rest[0] if rest else toks[0]}'. Type HELP.")
        fn, args = rest[0], rest[1:]

    simple = {
        "HELP": cmd_help, "EVTS": cmd_evts, "WEI": lambda: _px_table("WEI — World equity indices", INDICES),
        "FX": lambda: _px_table("FX — Currency rates", FX_PAIRS, 4), "FXC": lambda: _px_table("FX — Currency rates", FX_PAIRS, 4),
        "CMDTY": lambda: _px_table("CMDTY — Commodities", COMMODITIES), "GC": cmd_gc,
    }
    if fn == "PORT":
        raise ValueError("PORT opens the portfolio builder on this page. Type PORT in the Terminal tab.")
    if fn in simple:
        return simple[fn]()
    with_args = {
        "EQS": cmd_eqs, "ECOS": cmd_ecos, "ECST": cmd_ecos, "ECO": cmd_eco, "NEWS": cmd_news, "TOP": cmd_news,
        "N": cmd_news, "MOST": cmd_most, "MOV": cmd_mov,
    }
    if fn in with_args:
        return with_args[fn](args)
    if fn == "FRED":
        if not args:
            raise ValueError("Usage: FRED <series id>, e.g. FRED DGS30")
        return _fred_table(args[0], "FRED series")
    by_ticker = {
        "DES": lambda: cmd_des(ticker), "GP": lambda: cmd_gp(ticker), "HP": lambda: cmd_hp(ticker, args),
        "FA": lambda: cmd_fa(ticker), "ANR": lambda: cmd_anr(ticker), "ERN": lambda: cmd_ern(ticker),
        "DVD": lambda: cmd_dvd(ticker), "HDS": lambda: cmd_hds(ticker), "OMON": lambda: cmd_omon(ticker, args),
        "CN": lambda: cmd_cn(ticker), "QR": lambda: cmd_qr(ticker),
        "GF": lambda: cmd_gf(ticker, args), "EE": lambda: cmd_ee(ticker),
        "10K": lambda: cmd_filing(ticker, "10-K"), "10Q": lambda: cmd_filing(ticker, "10-Q"),
    }
    return by_ticker[fn]()


# --------------------------------------------------------------------------- #
# PORT: named portfolios (saved in portfolios.json next to app.py)
# --------------------------------------------------------------------------- #

PORTFOLIOS_FILE = os.path.join(BASE_DIR, "portfolios.json")
_PORT_LOCK = threading.Lock()


def _ports_load() -> dict:
    if not os.path.exists(PORTFOLIOS_FILE):
        return {}
    try:
        with open(PORTFOLIOS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        raise ValueError("portfolios.json couldn't be read (the file may be damaged). "
                         "Rename or fix it, then try again. Your saved portfolios are not overwritten.")
    return data if isinstance(data, dict) else {}


def _ports_write(data: dict):
    tmp = PORTFOLIOS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, PORTFOLIOS_FILE)


def _port_find(data: dict, name: str):
    low = name.strip().lower()
    return next((k for k in data if k.lower() == low), None)


def _port_clean_name(name) -> str:
    name = str(name or "").strip()
    if not re.fullmatch(r"[\w .&'()\-]{1,60}", name):
        raise ValueError("Portfolio names can use letters, numbers, spaces and . & ' ( ) - (up to 60 characters).")
    return name


def _port_clean_holdings(holdings) -> list[dict]:
    if not isinstance(holdings, list) or not holdings:
        raise ValueError("Add at least one holding.")
    if len(holdings) > 200:
        raise ValueError("A portfolio can hold up to 200 positions.")
    out = []
    for h in holdings:
        h = h if isinstance(h, dict) else {}
        t = str(h.get("ticker") or "").strip().upper()
        if not re.fullmatch(r"[A-Z0-9.\-^=]{1,12}", t):
            raise ValueError(f"'{t or '(blank)'}' isn't a valid ticker.")
        try:
            shares = float(h.get("shares"))
        except (TypeError, ValueError):
            raise ValueError(f"Enter the number of shares for {t}.")
        if not math.isfinite(shares) or shares == 0:
            raise ValueError(f"Shares for {t} must be a number other than zero (negative = short).")
        cost = h.get("cost")
        if cost in (None, ""):
            cost = None
        else:
            try:
                cost = float(cost)
            except (TypeError, ValueError):
                raise ValueError(f"Cost per share for {t} must be a number.")
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(f"Cost per share for {t} must be zero or more.")
        out.append({"ticker": t, "shares": shares, "cost": cost})
    return out


def port_list() -> dict:
    with _PORT_LOCK:
        return {"names": sorted(_ports_load(), key=str.lower)}


def port_get(name: str) -> dict:
    with _PORT_LOCK:
        data = _ports_load()
    key = _port_find(data, name or "")
    if key is None:
        raise ValueError(f"No saved portfolio named '{name}'.")
    return {"name": key, "holdings": data[key]["holdings"], "saved": data[key].get("saved"),
            "benchmark": data[key].get("benchmark") or ""}


def _port_clean_bench(b) -> str:
    b = str(b or "").strip().upper()
    if not b:
        return ""
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,12}", b):
        raise ValueError(f"'{b}' isn't a valid benchmark ticker. Enter an ETF such as SPY, QQQ or IWM.")
    return b


def port_save(name, holdings, benchmark="") -> dict:
    name = _port_clean_name(name)
    holdings = _port_clean_holdings(holdings)
    benchmark = _port_clean_bench(benchmark)
    with _PORT_LOCK:
        data = _ports_load()
        existing = _port_find(data, name)
        if existing:
            data.pop(existing)
        data[name] = {"holdings": holdings, "benchmark": benchmark,
                      "saved": datetime.now().isoformat(timespec="seconds")}
        _ports_write(data)
        return {"name": name, "replaced": existing is not None, "saved": data[name]["saved"],
                "names": sorted(data, key=str.lower)}


def port_delete(name) -> dict:
    with _PORT_LOCK:
        data = _ports_load()
        key = _port_find(data, str(name or ""))
        if key is None:
            raise ValueError(f"No saved portfolio named '{name}'.")
        data.pop(key)
        _ports_write(data)
        return {"names": sorted(data, key=str.lower)}


def port_value(holdings) -> dict:
    holdings = _port_clean_holdings(holdings)

    def px(t):
        try:
            c = yf.Ticker(t).history(period="5d")["Close"].dropna()
            return t, float(c.iloc[-1]), (float(c.iloc[-2]) if len(c) > 1 else None)
        except Exception:
            return t, None, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        prices = {t: (p, prev) for t, p, prev in ex.map(px, sorted({h["ticker"] for h in holdings}))}

    rows, missing = [], []
    for h in holdings:
        price, prev = prices[h["ticker"]]
        row = {"ticker": h["ticker"], "shares": h["shares"], "cost": h["cost"], "price": price, "chg_pct": None,
               "value": None, "day_pl": None, "pl": None, "pl_pct": None, "weight": None,
               "tot_rtn": None, "ctr": None}
        if price is None:
            if h["ticker"] not in missing:
                missing.append(h["ticker"])
        else:
            row["value"] = price * h["shares"]
            if prev:
                row["chg_pct"] = price / prev - 1
                row["day_pl"] = (price - prev) * h["shares"]
            if h["cost"] is not None:
                row["pl"] = (price - h["cost"]) * h["shares"]
                row["pl_pct"] = (price / h["cost"] - 1) if h["cost"] > 0 else None
        rows.append(row)

    gross = sum(abs(r["value"]) for r in rows if r["value"] is not None)
    for r in rows:
        if r["value"] is not None and gross:
            r["weight"] = abs(r["value"]) / gross
    valued = [r for r in rows if r["value"] is not None]
    with_cost = [r for r in valued if r["cost"] is not None]
    cost_total = sum(abs(r["cost"] * r["shares"]) for r in with_cost)
    pl_total = sum(r["pl"] for r in with_cost)
    with_day = [r for r in valued if r["day_pl"] is not None]
    day_pl = sum(r["day_pl"] for r in with_day) if with_day else None
    prev_total = sum(r["value"] - r["day_pl"] for r in with_day) if with_day else 0
    value_total = sum(r["value"] for r in valued)
    for r in with_cost:
        # Tot Rtn: price return on cost (long: price/cost - 1; short: gain when the price falls).
        # CTR: this holding's contribution to the portfolio's total return, in return points
        # (its P&L divided by the whole portfolio's cost basis), so the CTRs add up to the portfolio Tot Rtn.
        r["tot_rtn"] = r["pl_pct"] if r["shares"] > 0 else (-r["pl_pct"] if r["pl_pct"] is not None else None)
        r["ctr"] = r["pl"] / cost_total if cost_total else None
    totals = {"value": value_total, "day_pl": day_pl, "cost": cost_total if with_cost else None,
              "pl": pl_total if with_cost else None, "pl_pct": (pl_total / cost_total) if with_cost and cost_total else None,
              "day_pct": (day_pl / abs(prev_total)) if with_day and prev_total else None,
              "tot_rtn": (pl_total / cost_total) if with_cost and cost_total else None}
    return {"rows": rows, "totals": totals, "missing": missing}



# --------------------------------------------------------------------------- #
# PORT analytics: aggregate metrics, VaR, benchmark comparison, historical stress tests
# --------------------------------------------------------------------------- #

_HIST_CACHE: dict = {}
_HIST_LOCK = threading.Lock()


def _px_series(ticker: str) -> pd.Series:
    """Full daily history of dividend- and split-adjusted closes (so returns are total returns), cached 15 min."""
    now = time.time()
    with _HIST_LOCK:
        hit = _HIST_CACHE.get(ticker)
        if hit and now - hit[0] < 900:
            return hit[1]
    try:
        h = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        s = h["Close"].dropna() if h is not None and not h.empty else pd.Series(dtype=float)
    except Exception:
        s = pd.Series(dtype=float)
    if len(s):
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        s = pd.Series(s.to_numpy(dtype=float), index=idx.normalize())
        s = s[~s.index.duplicated(keep="last")].sort_index()
    with _HIST_LOCK:
        _HIST_CACHE[ticker] = (now, s)
    return s


def _px_many(tickers) -> dict:
    tickers = list(dict.fromkeys(tickers))
    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(tickers, ex.map(_px_series, tickers)))


def _asof(s: pd.Series, when):
    """Last close on or before `when` (None if the series starts later)."""
    s2 = s[s.index <= pd.Timestamp(when)]
    return float(s2.iloc[-1]) if len(s2) else None


def _port_positions(holdings, series: dict) -> tuple[dict, list]:
    """Net shares per ticker and the tickers with no price history."""
    shares: dict = {}
    for h in holdings:
        shares[h["ticker"]] = shares.get(h["ticker"], 0.0) + h["shares"]
    shares = {t: n for t, n in shares.items() if abs(n) > 1e-12}
    missing = [t for t in shares if t not in series or series[t].empty]
    return shares, missing


def _base_value(values: dict) -> float:
    """Denominator for portfolio percentages: net market value, or gross exposure if net isn't positive."""
    net = sum(values.values())
    return net if net > 0 else sum(abs(v) for v in values.values())


def _div_yield(info: dict, price):
    rate = _eqs_num(info.get("dividendRate"))
    if rate is not None and price:
        return rate / price
    for k in ("trailingAnnualDividendYield", "yield"):
        v = _eqs_num(info.get(k))
        if v is not None:
            return v
    return None


def _wavg(items, harmonic=False):
    """items: [(weight, value)] -> (average, share of total |weight| covered). Weights are renormalised over
    holdings that have the metric. harmonic=True is used for P/E (average the earnings yields, then invert)."""
    tot_w = sum(abs(w) for w, _ in items)
    use = [(w, v) for w, v in items if v is not None and (v > 0 if harmonic else True)]
    cov_w = sum(abs(w) for w, _ in use)
    if not use or not cov_w:
        return None, 0.0
    sw = sum(w for w, _ in use)
    if abs(sw) < 1e-12:
        return None, 0.0
    if harmonic:
        ey = sum(w / v for w, v in use) / sw
        return (1 / ey if ey > 0 else None), cov_w / tot_w
    return sum(w * v for w, v in use) / sw, cov_w / tot_w


def _portfolio_pnl(shares: dict, series: dict, tickers: list, days: int = 504):
    """Daily P&L (USD) of today's positions replayed over the last `days` trading days."""
    px = pd.concat({t: series[t] for t in tickers}, axis=1).sort_index().ffill().dropna()
    px = px.iloc[-(days + 1):]
    vals = pd.Series({t: shares[t] * float(series[t].iloc[-1]) for t in tickers})
    rets = px.pct_change().dropna()
    return (rets * vals).sum(axis=1), rets, vals


PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]


def _period_start(period: str, end: pd.Timestamp) -> pd.Timestamp:
    if period == "YTD":
        return pd.Timestamp(year=end.year - 1, month=12, day=31)
    n, unit = int(period[:-1]), period[-1]
    return end - (pd.DateOffset(months=n) if unit == "M" else pd.DateOffset(years=n))


def port_analyze(holdings, benchmark="", period="1Y") -> dict:
    holdings = _port_clean_holdings(holdings)
    bench = _port_clean_bench(benchmark)
    period = period if period in PERIODS else "1Y"
    names = sorted({h["ticker"] for h in holdings})
    series = _px_many(names + ([bench] if bench and bench not in names else []))
    shares, missing = _port_positions(holdings, series)
    have = [t for t in shares if t not in missing]
    if not have:
        raise ValueError("No price history found for any holding.")
    last = {t: float(series[t].iloc[-1]) for t in have}
    values = {t: shares[t] * last[t] for t in have}
    base = _base_value(values)
    if not base:
        raise ValueError("Portfolio value is zero.")
    out = {"base": base, "missing": missing, "benchmark": bench}

    # ---- aggregate metrics (weighted by market value) ----
    with ThreadPoolExecutor(max_workers=8) as ex:
        infos = dict(zip(have, ex.map(_info_cached, have)))
    wts = {t: values[t] / base for t in have}
    per, series_items = [], {"yield": [], "pe": [], "de": [], "cr": [], "roe": []}
    for t in have:
        i = infos[t]
        dy = _div_yield(i, last[t])
        pe = _eqs_num(i.get("trailingPE"))
        de = _eqs_num(i.get("debtToEquity"))
        de = de / 100 if de is not None else None          # Yahoo reports D/E in percent
        cr = _eqs_num(i.get("currentRatio"))
        roe = _eqs_num(i.get("returnOnEquity"))
        per.append({"ticker": t, "weight": wts[t], "yield": dy, "pe": pe, "de": de, "cr": cr, "roe": roe})
        for k, v in (("yield", dy), ("pe", pe), ("de", de), ("cr", cr), ("roe", roe)):
            series_items[k].append((wts[t], v))
    m = {}
    for k in series_items:
        v, cov = _wavg(series_items[k], harmonic=(k == "pe"))
        m[k] = {"value": v, "coverage": cov}
    out["metrics"], out["holdings_metrics"] = m, per

    # ---- value at risk (historical + parametric, 1 and 10 days) ----
    long_enough = [t for t in have if len(series[t]) >= 120]
    var = None
    if long_enough:
        pnl, rets, vals = _portfolio_pnl(shares, series, long_enough)
        cov_base = _base_value(dict(vals))
        scale = base / cov_base if cov_base else 1.0       # scale up if some holdings lacked history
        pnl = pnl * scale
        nd = NormalDist()
        mu, sd = float(pnl.mean()), float(pnl.std(ddof=1))
        rows = []
        for conf in (0.95, 0.99):
            q = float(np.percentile(pnl, (1 - conf) * 100))
            tail = pnl[pnl <= q]
            h1 = -q
            cvar1 = -float(tail.mean()) if len(tail) else h1
            p1 = -(mu + nd.inv_cdf(1 - conf) * sd)
            rows.append({"conf": conf, "hist_1d": h1, "hist_10d": h1 * math.sqrt(10), "cvar_1d": cvar1,
                         "param_1d": p1, "param_10d": p1 * math.sqrt(10)})
        var = {"rows": rows, "days": int(len(pnl)), "from": str(pnl.index[0].date()), "to": str(pnl.index[-1].date()),
               "excluded": [t for t in have if t not in long_enough]}
        if bench and not series[bench].empty and len(series[bench]) > 120:
            br = series[bench].pct_change().dropna().iloc[-len(pnl):]
            bp = br * base
            var["bench"] = [{"conf": c, "hist_1d": -float(np.percentile(bp, (1 - c) * 100))} for c in (0.95, 0.99)]
        # risk stats vs benchmark
        port_ret = pnl / base
        stats = {"vol": float(port_ret.std(ddof=1) * math.sqrt(252)), "beta": None, "corr": None, "bench_vol": None}
        if bench and not series[bench].empty:
            br = series[bench].pct_change().dropna()
            j = pd.concat([port_ret, br], axis=1, join="inner").dropna()
            if len(j) > 60 and j.iloc[:, 1].var() > 0:
                stats["beta"] = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
                stats["corr"] = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
                stats["bench_vol"] = float(j.iloc[:, 1].std(ddof=1) * math.sqrt(252))
        var["stats"] = stats
    out["var"] = var

    # ---- benchmark comparison (total return incl. dividends, today's holdings replayed) ----
    comp = None
    if bench and not series[bench].empty:
        end = max(series[t].index[-1] for t in have)
        rows, charts = [], {}
        for p in PERIODS:
            start = _period_start(p, end)
            inc = [t for t in have if series[t].index[0] <= start + pd.Timedelta(days=7)]
            b0 = _asof(series[bench], start)
            if b0 is None or not inc:
                rows.append({"period": p, "portfolio": None, "benchmark": None, "excess": None, "excluded": len(have) - len(inc)})
                continue
            v0 = {t: shares[t] * _asof(series[t], start) for t in inc}
            v1 = {t: shares[t] * last[t] for t in inc}
            b = _base_value(v0)
            pr = (sum(v1.values()) - sum(v0.values())) / b if b else None
            br = float(series[bench].iloc[-1]) / b0 - 1
            rows.append({"period": p, "portfolio": pr, "benchmark": br,
                         "excess": None if pr is None else pr - br, "excluded": len(have) - len(inc)})
            if pr is not None:
                px = pd.concat({t: series[t] for t in inc}, axis=1).sort_index().ffill()
                px = px[px.index >= px.index[px.index >= start][0]].dropna()
                pv = sum(px[t] * shares[t] for t in inc)
                pidx = 100 * (1 + (pv - pv.iloc[0]) / b)
                bs = series[bench][series[bench].index >= px.index[0]].reindex(px.index).ffill()
                bidx = 100 * bs / bs.iloc[0]
                step = max(1, len(px) // 400)
                charts[p] = {"dates": [str(d.date()) for d in px.index[::step]],
                             "portfolio": [round(float(x), 3) for x in pidx.iloc[::step]],
                             "benchmark": [round(float(x), 3) for x in bidx.iloc[::step]]}
        comp = {"rows": rows, "charts": charts, "period": period}
    out["compare"] = comp
    return out


# ---- tracking error, active risk and factor exposures ------------------------------------------------
_INFO_CACHE: dict = {}


def _info_cached(t: str) -> dict:
    now = time.time()
    with _HIST_LOCK:
        hit = _INFO_CACHE.get(t)
        if hit and now - hit[0] < 1800:
            return hit[1]
    try:
        info = yf.Ticker(t).get_info() or {}
    except Exception:
        info = {}
    with _HIST_LOCK:
        _INFO_CACHE[t] = (now, info)
    return info


# code, label, (long ETF, short ETF or None), meaning when the portfolio is OVER, meaning when UNDER the benchmark
FACTOR_DEFS = [
    ("MKT", "Market beta (SPY)", ("SPY", None), "more aggressive than the benchmark (higher market sensitivity)",
     "more defensive than the benchmark (lower market sensitivity)"),
    ("SIZE", "Size (small minus large)", ("IWM", "IWB"), "tilted toward smaller companies", "tilted toward larger companies"),
    ("VAL", "Value (value minus growth)", ("IWD", "IWF"), "tilted toward value stocks", "tilted toward growth stocks"),
    ("MOM", "Momentum", ("MTUM", "SPY"), "tilted toward recent winners", "tilted away from recent winners"),
    ("QUAL", "Quality", ("QUAL", "SPY"), "tilted toward high-quality (profitable, stable) companies", "tilted away from high-quality companies"),
    ("LVOL", "Low volatility", ("USMV", "SPY"), "more defensive / lower-volatility stocks", "higher-volatility stocks"),
]


def _ols(y: np.ndarray, X: np.ndarray):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    s2 = float(resid @ resid) / max(n - k, 1)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta, se


def _factor_status(beta: float, t: float, over: str, under: str):
    if abs(t) >= 2 and abs(beta) >= 0.05:
        return ("Overexposed" if beta > 0 else "Underexposed"), (over if beta > 0 else under)
    if abs(t) >= 1 and abs(beta) >= 0.03:
        return ("Slightly over" if beta > 0 else "Slightly under"), (over if beta > 0 else under)
    return "In line", "no meaningful difference from the benchmark"


def _norm_name(s) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


# ---- State Street (SSGA) sector weights for SPY --------------------------------------------------------
# Yahoo's sector breakdown for SPY is unreliable, so for SPY the benchmark sector weights come straight from
# State Street's own daily holdings file (the same data behind the Sector Breakdown on the SPY fund page).
SSGA_SPY_XLSX = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
SSGA_SPY_PAGE = "https://www.ssga.com/sg/en/institutional/etfs/state-street-spdr-sp-500-etf-trust-s27"  # same SPY trust, server-rendered
_SSGA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                 "Accept": "*/*"}
# GICS sector names (State Street) -> the sector keys used in this app (Yahoo-style names, normalised)
_GICS_KEY = {"informationtechnology": "technology", "financials": "financialservices", "healthcare": "healthcare",
             "consumerdiscretionary": "consumercyclical", "consumerstaples": "consumerdefensive", "materials": "basicmaterials",
             "communicationservices": "communicationservices", "industrials": "industrials", "energy": "energy",
             "utilities": "utilities", "realestate": "realestate"}
_SSGA_CACHE: dict = {}


def _read_xlsx_rows(content: bytes) -> list:
    """Minimal .xlsx reader (standard library only): first worksheet as a list of rows."""
    import zipfile
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(io.BytesIO(content))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(ns + "si"):
            shared.append("".join(t.text or "" for t in si.iter(ns + "t")))
    sheets = sorted(n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
    rows = []
    for r in ET.fromstring(z.read(sheets[0])).iter(ns + "row"):
        cells = {}
        for c in r.iter(ns + "c"):
            col = 0
            for ch in re.match(r"[A-Z]+", c.get("r")).group():
                col = col * 26 + ord(ch) - 64
            v, t = c.find(ns + "v"), c.get("t")
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(ns + "t"))
            else:
                val = v.text if v is not None else None
            cells[col - 1] = val
        if cells:
            rows.append([cells.get(i) for i in range(max(cells) + 1)])
    return rows


def _ssga_from_xlsx() -> dict:
    r = requests.get(SSGA_SPY_XLSX, timeout=30, headers=_SSGA_HEADERS)
    r.raise_for_status()
    rows = _read_xlsx_rows(r.content)
    as_of = None
    for row in rows[:12]:
        for cell in row:
            if isinstance(cell, str) and "as of" in cell.lower():
                as_of = re.sub(r"(?i).*as of:?\s*", "", cell).strip() or None
    hdr = next((i for i, row in enumerate(rows) if {"weight", "sector"} <= {str(c).strip().lower() for c in row if c}), None)
    if hdr is None:
        raise ValueError("holdings file layout changed (no Weight/Sector header)")
    names = [str(c).strip().lower() if c else "" for c in rows[hdr]]
    iw, isec, itk = names.index("weight"), names.index("sector"), (names.index("ticker") if "ticker" in names else None)
    weights, tickers = {}, {}
    for row in rows[hdr + 1:]:
        if len(row) <= max(iw, isec):
            continue
        key = _GICS_KEY.get(_norm_name(row[isec]))
        try:
            w = float(row[iw])
        except (TypeError, ValueError):
            continue
        if not key:
            continue                                   # cash, currency and footer rows have no GICS sector
        weights[key] = weights.get(key, 0.0) + w
        if itk is not None and row[itk]:
            tickers[str(row[itk]).strip().upper().replace(".", "-")] = key
    total = sum(weights.values())
    if len(weights) < 9 or not 90 <= total <= 110:
        raise ValueError("holdings file didn't look like the full SPY portfolio")
    return {"weights": {k: v / total for k, v in weights.items()}, "tickers": tickers, "as_of": as_of,
            "source": "State Street daily holdings file for SPY"}


def _ssga_from_page() -> dict:
    r = requests.get(SSGA_SPY_PAGE, timeout=30, headers=_SSGA_HEADERS)
    r.raise_for_status()
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
    start = text.find("Fund Sector Breakdown")
    if start < 0:
        raise ValueError("sector table not found on the State Street page")
    m_date = re.search(r"Fund Sector Breakdown as of (\d{1,2} \w{3} \d{4})", text)
    weights = {}
    for name, pct in re.findall(r"(Information Technology|Financials|Health Care|Consumer Discretionary|Communication Services|"
                                r"Industrials|Consumer Staples|Energy|Utilities|Real Estate|Materials)\s*(\d{1,2}\.\d{1,2})\s*%", text[start:]):
        weights.setdefault(_GICS_KEY[_norm_name(name)], float(pct))      # first table on the page is the fund's
    total = sum(weights.values())
    if len(weights) < 9 or not 97 <= total <= 103:
        raise ValueError("couldn't read the sector table on the State Street page")
    return {"weights": {k: v / total for k, v in weights.items()}, "tickers": {},
            "as_of": m_date.group(1) if m_date else None, "source": "State Street SPY fund page"}


def _ssga_spy_sectors():
    """SPY sector weights from State Street ({key: fraction}), cached 6 hours; None if State Street can't be reached."""
    hit = _SSGA_CACHE.get("spy")
    if hit and time.time() - hit[0] < 6 * 3600:
        return hit[1]
    data = None
    for fn in (_ssga_from_xlsx, _ssga_from_page):
        try:
            data = fn()
            break
        except Exception:
            data = None
    _SSGA_CACHE["spy"] = (time.time(), data)
    return data


def port_active(holdings, benchmark="") -> dict:
    """Tracking error, active-risk breakdown and factor / sector exposures versus the benchmark ETF."""
    holdings = _port_clean_holdings(holdings)
    bench = _port_clean_bench(benchmark)
    if not bench:
        raise ValueError("Enter a benchmark ETF (SPY, QQQ, IWM, ...) in the Benchmark box first.")
    names = sorted({h["ticker"] for h in holdings})
    fac_tickers = sorted({t for _, _, (a, b), _, _ in FACTOR_DEFS for t in (a, b) if t})
    series = _px_many(names + [bench] + fac_tickers)
    if series[bench].empty:
        raise ValueError(f"No price history found for the benchmark '{bench}'.")
    shares, missing = _port_positions(holdings, series)
    have = [t for t in shares if t not in missing]
    values = {t: shares[t] * float(series[t].iloc[-1]) for t in have}
    base = _base_value(values) if values else 0
    long_enough = [t for t in have if len(series[t]) >= 250]
    if not long_enough or not base:
        raise ValueError("Need at least a year of price history for at least one holding.")
    pnl, rets, vals = _portfolio_pnl(shares, series, long_enough, days=756)
    cov_base = _base_value(dict(vals))
    scale = base / cov_base if cov_base else 1.0
    w = vals * scale / base                         # portfolio weights of the holdings that have enough history
    r_p = pnl * scale / base

    cols = {"p": r_p, "b": series[bench].pct_change()}
    used = []
    for code, label, (a, b), _, _ in FACTOR_DEFS:
        if series[a].empty or (b and series[b].empty):
            continue
        cols[code] = series[a].pct_change() - (series[b].pct_change() if b else 0)
        used.append(code)
    df = pd.concat(cols, axis=1, join="inner").dropna().tail(504)
    if len(df) < 120:
        raise ValueError("Not enough overlapping price history between the portfolio, benchmark and factor ETFs.")
    ra = df["p"] - df["b"]
    n = len(df)

    def te(days):
        x = ra.tail(days)
        return float(x.std(ddof=1) * math.sqrt(252)) if len(x) >= 20 else None

    te_full = te(n)
    act_ret = float(ra.mean() * 252)
    out = {"benchmark": bench, "n": int(n), "from": str(df.index[0].date()), "to": str(df.index[-1].date()),
           "excluded": [t for t in have if t not in long_enough], "missing": missing,
           "tracking": {"te": te_full, "te_3m": te(63), "te_1y": te(252), "active_return": act_ret,
                        "info_ratio": (act_ret / te_full) if te_full else None,
                        "r2": float(df["p"].corr(df["b"]) ** 2),
                        "beta": float(df["p"].cov(df["b"]) / df["b"].var())}}

    # ---- factor exposures: regress portfolio, benchmark and (portfolio - benchmark) on the factor spreads ----
    X = np.column_stack([np.ones(n)] + [df[c].to_numpy() for c in used])
    bp, _ = _ols(df["p"].to_numpy(), X)
    bb, _ = _ols(df["b"].to_numpy(), X)
    ba, sea = _ols(ra.to_numpy(), X)
    var_a = float(ra.var(ddof=1))
    fac_rows, explained = [], 0.0
    defs = {d[0]: d for d in FACTOR_DEFS}
    for k, code in enumerate(used, start=1):
        share = float(ba[k] * np.cov(df[code], ra, ddof=1)[0, 1] / var_a) if var_a else 0.0
        explained += share
        status, why = _factor_status(float(ba[k]), float(ba[k] / sea[k]) if sea[k] else 0.0, defs[code][3], defs[code][4])
        fac_rows.append({"code": code, "label": defs[code][1], "portfolio": float(bp[k]), "benchmark": float(bb[k]),
                         "active": float(ba[k]), "t": float(ba[k] / sea[k]) if sea[k] else None,
                         "risk_share": share, "status": status, "meaning": why})
    out["factors"] = fac_rows
    out["risk_split"] = {"factors": explained, "specific": max(0.0, 1 - explained)}
    out["alpha"] = {"annual": float(ba[0] * 252), "t": float(ba[0] / sea[0]) if sea[0] else None}
    out["summary"] = [f"{r['label']}: {r['status'].lower()}, {r['meaning']}" for r in
                      sorted(fac_rows, key=lambda r: -abs(r["active"])) if r["status"] != "In line"][:4]

    # ---- which holdings drive the tracking error (shares of active variance; they sum to 100% with the benchmark) ----
    R = rets.loc[df.index]
    hold = []
    for t in long_enough:
        c = float(w[t] * np.cov(R[t], ra, ddof=1)[0, 1] / var_a) if var_a else 0.0
        hold.append({"ticker": t, "weight": float(w[t]), "risk_share": c, "te_points": c * (te_full or 0)})
    hold.sort(key=lambda r: -abs(r["risk_share"]))
    cb = float(-np.cov(df["b"], ra, ddof=1)[0, 1] / var_a) if var_a else 0.0
    out["holdings"] = hold
    out["bench_row"] = {"risk_share": cb, "te_points": cb * (te_full or 0)}

    # ---- sector weights vs the benchmark ETF (when Yahoo publishes the ETF's sector breakdown) ----
    with ThreadPoolExecutor(max_workers=8) as ex:
        infos = dict(zip(have, ex.map(_info_cached, have)))
    ssga = _ssga_spy_sectors() if bench == "SPY" else None
    pw, unclassified = {}, 0.0
    for t in have:
        # for SPY, use State Street's own sector label when the stock is in the fund, so both sides use the same classification
        sec = (ssga or {}).get("tickers", {}).get(t.upper().replace(".", "-")) or _norm_name(infos[t].get("sector"))
        if sec:
            pw[sec] = pw.get(sec, 0.0) + values[t] / base
        else:
            unclassified += values[t] / base
    bw, source = None, None
    if ssga:
        bw = ssga["weights"]
        source = f"{ssga['source']}{', as of ' + ssga['as_of'] if ssga['as_of'] else ''}"
    else:
        try:
            sw = yf.Ticker(bench).funds_data.sector_weightings
            bw = {_norm_name(k): float(v) for k, v in dict(sw).items()} or None
            source = "Yahoo Finance"
        except Exception:
            bw = None
    sectors = []
    for s in SECTORS:
        key = _norm_name(s)
        p, b = pw.get(key, 0.0), (bw.get(key, 0.0) if bw is not None else None)
        act = None if b is None else p - b
        status = "—" if act is None else ("Overweight" if act >= 0.05 else "Underweight" if act <= -0.05 else "In line")
        sectors.append({"sector": s, "portfolio": p, "benchmark": b, "active": act, "status": status})
    out["sectors"] = {"rows": sectors, "unclassified": unclassified, "benchmark_available": bw is not None,
                      "source": source, "ssga_failed": bench == "SPY" and ssga is None}
    return out


STRESS_EVENTS = [
    ("Black Monday 1987", "1987-10-13", "1987-10-19"),
    ("Dot-com crash (peak to trough)", "2000-03-24", "2002-10-09"),
    ("September 11 attacks (reopening week)", "2001-09-10", "2001-09-21"),
    ("Global financial crisis (peak to trough)", "2007-10-09", "2009-03-09"),
    ("Lehman collapse (Sep 12 - Nov 20, 2008)", "2008-09-12", "2008-11-20"),
    ("Flash crash 2010", "2010-04-23", "2010-07-02"),
    ("US credit downgrade 2011", "2011-07-22", "2011-08-08"),
    ("China devaluation selloff 2015", "2015-08-17", "2015-08-25"),
    ("Volmageddon / Q4 2018 selloff", "2018-09-20", "2018-12-24"),
    ("COVID-19 crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
    ("Regional bank crisis 2023", "2023-03-08", "2023-03-15"),
    ("Yen carry-trade unwind 2024", "2024-07-16", "2024-08-05"),
    ("Tariff shock 2025", "2025-02-19", "2025-04-08"),
]


def _beta_of(t: str) -> float:
    try:
        b = _eqs_num((yf.Ticker(t).get_info() or {}).get("beta"))
    except Exception:
        b = None
    return 1.0 if b is None else b


def port_stress(holdings, benchmark="", custom=None) -> dict:
    """Apply each historical event's price moves to TODAY's market values. A holding that didn't trade through
    the whole window is estimated as its beta x the S&P 500's move (flagged 'est')."""
    holdings = _port_clean_holdings(holdings)
    bench = _port_clean_bench(benchmark)
    events = list(STRESS_EVENTS)
    if custom and custom.get("start") and custom.get("end"):
        try:
            cs, ce = pd.Timestamp(custom["start"]), pd.Timestamp(custom["end"])
        except Exception:
            raise ValueError("Custom dates must look like 2020-02-19.")
        if ce <= cs:
            raise ValueError("The custom end date must be after the start date.")
        events.insert(0, (f"Custom: {cs.date()} to {ce.date()}", str(cs.date()), str(ce.date())))
    names = sorted({h["ticker"] for h in holdings})
    series = _px_many(names + ["^GSPC"] + ([bench] if bench and bench not in names else []))
    sp = series["^GSPC"]
    if sp.empty:
        raise ValueError("Couldn't download S&P 500 history (^GSPC). Check your connection and try again.")
    shares, missing = _port_positions(holdings, series)
    live = {t: shares[t] * float(series[t].iloc[-1]) for t in shares if t not in missing}
    for t in missing:                                      # no history at all: use the live quote if there is one
        try:
            live[t] = shares[t] * get_last_price(t)
        except Exception:
            pass
    base = _base_value(live)
    if not live or not base:
        raise ValueError("No prices found for these holdings.")
    betas: dict = {}
    results = []
    for name, s0, e0 in events:
        s_ts, e_ts = pd.Timestamp(s0), pd.Timestamp(e0)
        sp0, sp1 = _asof(sp, s_ts), _asof(sp, e_ts)
        if sp0 is None or sp1 is None:
            results.append({"name": name, "start": s0, "end": e0, "unavailable": True})
            continue
        spr = sp1 / sp0 - 1
        rows, pnl_tot, est_n = [], 0.0, 0
        for t, v in live.items():
            s = series.get(t)
            est = True
            ret = None
            if s is not None and not s.empty and s.index[0] <= s_ts + pd.Timedelta(days=5) and s.index[-1] >= e_ts:
                a, b = _asof(s, s_ts), _asof(s, e_ts)
                if a and b:
                    ret, est = b / a - 1, False
            if ret is None:
                betas.setdefault(t, _beta_of(t))
                ret = betas[t] * spr
                est_n += 1
            rows.append({"ticker": t, "value": v, "ret": ret, "pnl": v * ret, "est": est})
            pnl_tot += v * ret
        brt = None
        if bench and bench in series and not series[bench].empty and series[bench].index[0] <= s_ts + pd.Timedelta(days=5):
            a, b = _asof(series[bench], s_ts), _asof(series[bench], e_ts)
            brt = (b / a - 1) if a and b and series[bench].index[-1] >= e_ts else None
        rows.sort(key=lambda r: r["pnl"])
        results.append({"name": name, "start": s0, "end": e0, "sp500": spr, "pnl": pnl_tot, "pct": pnl_tot / base,
                        "benchmark": brt, "est_count": est_n, "rows": rows})
    return {"base": base, "benchmark": bench, "scenarios": results}


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#000000; --panel:#050505; --panel-2:#0b0b0b; --border:#2a2a2a;
    --text:#ff9a1f; --muted:#b87420; --accent:#33ffb8; --accent-2:#ffb454;
    --danger:#ff5c7a; --warn:#ff9a1f;
  }
  * { box-sizing: border-box; }
  html,body { margin:0; padding:0; background:var(--bg); color:var(--text);
    font-family:'Space Grotesk',system-ui,sans-serif; min-height:100%; }
  body { background-image:none; }
  .mono { font-family:'JetBrains Mono',monospace; }
  a { color: var(--accent-2); }

  header { padding:24px 24px 0; max-width:1100px; margin:0 auto; }
  .logo { display:flex; align-items:center; gap:10px; font-family:'JetBrains Mono',monospace;
    font-size:13px; color:var(--muted); letter-spacing:2px; text-transform:uppercase; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--accent);
    box-shadow:0 0 8px var(--accent); animation:pulse 1.8s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.35;} }
  h1 { font-size:clamp(26px,4vw,38px); margin:8px 0 14px;
    background:linear-gradient(90deg,#ff8c1a,#ffb454);
    -webkit-background-clip:text; background-clip:text; color:transparent; font-weight:700; }

  nav.tabs { display:flex; gap:4px; border-bottom:1px solid var(--border); margin-bottom:20px; flex-wrap:wrap; }
  nav.tabs button {
    background:none; border:none; color:var(--muted); padding:12px 18px; font-size:14px;
    cursor:pointer; border-bottom:2px solid transparent; font-family:'Space Grotesk',sans-serif;
    font-weight:500; transition:color .15s, border-color .15s;
  }
  nav.tabs button.active { color:var(--accent); border-color:var(--accent); }
  nav.tabs button:hover { color:var(--text); }

  main { max-width:1100px; margin:0 auto; padding:0 24px 60px; }
  .tabpanel { display:none; }
  .tabpanel.active { display:block; }

  .panel { background:linear-gradient(180deg,var(--panel),var(--panel-2));
    border:1px solid var(--border); border-radius:16px; padding:22px 24px; margin-bottom:18px; }
  .panel h2 { margin:0 0 16px; font-size:13px; text-transform:uppercase; letter-spacing:2px;
    color:var(--accent-2); font-family:'JetBrains Mono',monospace; }

  input[type=text], input[type=password], input[type=number], select {
    background:var(--panel); border:1px solid var(--border); color:var(--text);
    padding:11px 14px; border-radius:9px; font-size:14px; font-family:'JetBrains Mono',monospace;
    outline:none; transition:border-color .15s;
  }
  input:focus, select:focus { border-color:var(--accent-2); }
  button.primary, button.action {
    background:linear-gradient(90deg,#ff8c1a,#ffb454); color:#000; border:none;
    border-radius:9px; padding:11px 20px; font-size:14px; font-weight:700; cursor:pointer;
    transition:transform .1s, box-shadow .15s; font-family:'Space Grotesk',sans-serif;
  }
  button.primary:hover, button.action:hover { transform:translateY(-1px); box-shadow:0 6px 20px rgba(255,154,31,0.3); }
  button.danger { background:linear-gradient(90deg,#ff8095,var(--danger)); color:#1a0006; }
  button:disabled { opacity:.5; cursor:default; transform:none; box-shadow:none; }
  button.ghost { background:none; border:1px solid var(--border); color:var(--muted); border-radius:9px; padding:8px 14px; cursor:pointer; }

  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:14px; }
  .stat .label { font-size:11px; text-transform:uppercase; letter-spacing:1px; color:var(--muted); }
  .stat .value { font-family:'JetBrains Mono',monospace; font-size:20px; margin-top:3px; }
  .up { color:var(--accent); } .down { color:var(--danger); }

  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:9px 8px; border-bottom:1px solid var(--border); font-family:'JetBrains Mono',monospace; }
  th { color:var(--muted); text-transform:uppercase; font-size:11px; letter-spacing:1px; }
  tr:last-child td { border-bottom:none; }

  .badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px;
    font-family:'JetBrains Mono',monospace; font-weight:700; letter-spacing:.5px; text-transform:uppercase; }
  .badge.buy, .badge.new { background:rgba(51,255,184,0.15); color:var(--accent); }
  .badge.sell, .badge.exit { background:rgba(255,92,122,0.15); color:var(--danger); }
  .badge.trim { background:rgba(255,154,31,0.15); color:var(--warn); }

  #status { margin:14px 0; color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:13px; }
  .error { color:var(--danger); }
  .spinner { width:14px; height:14px; border-radius:50%; border:2px solid var(--border);
    border-top-color:var(--accent); display:inline-block; animation:spin .7s linear infinite;
    margin-right:8px; vertical-align:middle; }
  @keyframes spin { to { transform:rotate(360deg); } }

  .tv-container { border-radius:14px; overflow:hidden; border:1px solid var(--border); height:max(780px, 85vh); }
  .tv-container .tradingview-widget-container, #tv_chart { height:100%; width:100%; }
  main.wide { max-width:1800px; }
  header.wide { max-width:1800px; }
  .muted { color:var(--muted); }
  .small { font-size:12px; }
  .stat .sub { font-size:11px; color:var(--muted); margin-top:2px; }
  tr.fold > td { background:#070707; padding:6px 14px 14px; }
  .linkish { cursor:pointer; color:var(--accent-2); user-select:none; }
  footer { text-align:center; color:var(--muted); font-size:12px; padding:20px; }
  .term-bar { display:flex; align-items:center; gap:10px; background:#000; border:1px solid var(--border); border-radius:10px; padding:8px 14px; }
  .term-bar span { color:var(--warn); font-family:'JetBrains Mono',monospace; font-weight:700; }
  .term-bar input[type=text] { flex:1; background:transparent; border:none; color:var(--warn); font-size:16px; text-transform:uppercase; outline:none; padding:6px; }
  .term-hint { color:var(--muted); font-size:12px; margin-top:8px; font-family:'JetBrains Mono',monospace; }
  input[type=date] { background:var(--panel); border:1px solid var(--border); color:var(--text); padding:11px 14px; border-radius:9px; font-size:14px; font-family:'JetBrains Mono',monospace; color-scheme:dark; }
  .news-row { display:flex; gap:10px; align-items:baseline; padding:10px 0; border-bottom:1px solid var(--border); }
  .news-row:last-child { border-bottom:none; }
  .news-row a { flex:1; color:var(--text); text-decoration:none; font-weight:500; }
  .news-row a:hover { color:var(--accent-2); }
  .kind { font-size:11px; letter-spacing:1px; text-transform:uppercase; color:var(--accent-2); font-family:'JetBrains Mono',monospace; min-width:72px; }
  select option { background:#000; color:var(--text); }
</style>
</head>
<body>

<header>
  <div class="logo"><span class="dot"></span> LOCAL // NOT INVESTMENT ADVICE</div>
  <h1>Research Terminal</h1>
  <nav class="tabs">
    <button data-tab="research" class="active">Research</button>
    <button data-tab="trading">Paper Trading</button>
    <button data-tab="charts">Charts &amp; Gaps</button>
    <button data-tab="terminal">Terminal</button>
    <button data-tab="barrons">Barron’s</button>
  </nav>
</header>

<main>

  <!-- RESEARCH TAB -->
  <section id="tab-research" class="tabpanel active">
    <div class="panel">
      <div class="row">
        <input type="text" id="r-ticker" placeholder="TICKER e.g. AAPL" style="flex:1; min-width:160px; text-transform:uppercase;">
        <input type="password" id="r-avkey" placeholder="Alpha Vantage key (optional)" style="width:240px;">
        <button class="primary" id="r-go">Research →</button>
      </div>
    </div>
    <div id="r-status"></div>
    <div id="r-results"></div>
  </section>

  <!-- PAPER TRADING TAB -->
  <section id="tab-trading" class="tabpanel">
    <div class="panel">
      <h2>Account</h2>
      <div id="pt-summary" class="grid"></div>
      <div style="margin-top:14px;"><button class="ghost" id="pt-reset">Reset to $10,000</button></div>
    </div>

    <div class="panel">
      <h2>Place a Trade</h2>
      <div class="row">
        <select id="pt-action">
          <option value="buy">Buy (long)</option>
          <option value="sell">Sell (close long)</option>
          <option value="short">Short</option>
          <option value="cover">Cover (close short)</option>
          <option value="buy_put">Buy Put</option>
        </select>
        <input type="text" id="pt-ticker" placeholder="TICKER" style="text-transform:uppercase; width:120px;">
        <input type="number" id="pt-qty" placeholder="Qty / Contracts" value="1" min="0" style="width:150px;">
        <span id="pt-put-fields" style="display:none;" class="row">
          <select id="pt-expiration"></select>
          <select id="pt-strike"></select>
        </span>
        <button class="action" id="pt-submit">Submit</button>
      </div>
      <div id="pt-trade-status" style="margin-top:10px; font-size:13px;"></div>
    </div>

    <div class="panel"><h2>Long Positions</h2><div id="pt-long"></div></div>
    <div class="panel"><h2>Short Positions</h2><div id="pt-short"></div></div>
    <div class="panel"><h2>Options Positions</h2><div id="pt-options"></div></div>
    <div class="panel"><h2>Trade History</h2><div id="pt-log"></div></div>
  </section>

  <!-- CHARTS & GAPS TAB -->
  <section id="tab-charts" class="tabpanel">
    <div class="panel">
      <div class="row">
        <input type="text" id="c-ticker" placeholder="TICKER e.g. AAPL" style="flex:1; min-width:160px; text-transform:uppercase;">
        <button class="primary" id="c-go">Load Chart &amp; Find Gaps →</button>
      </div>
    </div>
    <div class="panel">
      <h2>TradingView Chart (10-minute)</h2>
      <!-- the chart fills the full width of the page on this tab and is at least 780px tall -->
      <div class="tv-container">
        <div class="tradingview-widget-container">
          <div id="tv_chart"></div>
        </div>
      </div>
    </div>
    <div id="g-status"></div>
    <div class="panel">
      <h2>Detected Price Gaps (10-min candles, Yahoo Finance data)</h2>
      <p style="color:var(--muted); font-size:13px; margin-top:-8px;">
        A gap = this candle's open price jumped from the prior candle's close by more than 0.3%.
        Most gaps naturally cluster at the first candle of each trading day (overnight gaps) —
        those are flagged separately below.
        A gap is <b>filled</b> when price later trades back to the prior close (a gap up fills when the price
        drops back to it; a gap down fills when the price climbs back to it). <b>Open</b> means it hasn't
        happened yet in the data window, so the table shows how much of the gap has closed so far.
      </p>
      <div class="row" style="margin-bottom:10px;">
        <select id="g-filter"><option value="all">Show all gaps</option><option value="open">Open gaps only</option><option value="filled">Filled gaps only</option></select>
        <span id="g-summary" class="muted small"></span>
      </div>
      <div id="g-results"></div>
    </div>
  </section>

  <!-- TERMINAL TAB -->
  <section id="tab-terminal" class="tabpanel">
    <div class="panel">
      <div class="term-bar"><span>&gt;</span>
        <input type="text" id="term-input" placeholder="NEWS, EQS, ECO, AAPL US EQUITY DES, HELP…" autocomplete="off" spellcheck="false">
      </div>
      <div class="term-hint">Press Enter to run. Up/Down arrows recall earlier commands. Try: HELP</div>
    </div>
    <div class="panel" id="port-panel" style="display:none;">
      <div class="row" style="justify-content:space-between;">
        <h2 style="margin:0;">PORT — Portfolio builder</h2>
        <button class="ghost" id="port-close">Close</button>
      </div>
      <div class="row" style="margin-top:14px;">
        <input type="text" id="port-name" placeholder="Portfolio name" style="flex:1; min-width:180px;">
        <select id="port-saved"><option value="">Load a saved portfolio…</option></select>
        <button class="ghost" id="port-new">New</button>
        <button class="ghost" id="port-delete">Delete</button>
      </div>
      <div class="row" style="margin-top:10px;">
        <input type="text" id="port-bench" placeholder="Benchmark ETF (e.g. SPY)" style="width:240px; text-transform:uppercase;">
        <span class="muted small">Benchmark for the comparison, VaR and stress-test sections. Saved with the portfolio.</span>
      </div>
      <div id="port-table" style="margin-top:14px; overflow-x:auto;"></div>
      <div class="row" style="margin-top:12px;">
        <button class="ghost" id="port-add">+ Add holding</button>
        <button class="ghost" id="port-refresh">Update prices</button>
        <button class="action" id="port-save">Save</button>
        <button class="primary" id="port-analyze">Run analysis (metrics, VaR, benchmark)</button>
        <button class="primary" id="port-active">Tracking error &amp; factors</button>
      </div>
      <div id="port-totals" class="grid" style="margin-top:16px;"></div>
      <div id="port-analysis"></div>
      <div id="port-active-out"></div>
      <div style="margin-top:22px; border-top:1px solid var(--border); padding-top:16px;">
        <h2 style="margin-bottom:10px;">Stress test: historical market events</h2>
        <div class="row">
          <button class="primary" id="port-stress-run">Run stress tests</button>
          <span class="muted small">or add your own window:</span>
          <input type="text" id="port-st-start" placeholder="Start YYYY-MM-DD" style="width:160px;">
          <input type="text" id="port-st-end" placeholder="End YYYY-MM-DD" style="width:160px;">
        </div>
        <div id="port-stress"></div>
      </div>
      <div id="port-status" style="margin-top:10px; font-size:13px;"></div>
      <p style="color:var(--muted); font-size:12px; margin:10px 0 0;">
        Shares can be negative for a short position. Cost per share is optional and only used for P&amp;L.
        Saved portfolios are stored in portfolios.json next to app.py.
      </p>
    </div>
    <div id="term-status"></div>
    <div id="term-output"></div>
  </section>

  <!-- BARRONS TAB -->
  <section id="tab-barrons" class="tabpanel">
    <div class="panel">
      <h2>Barron’s via ASU ProQuest</h2>
      <div class="row">
        <input type="text" id="bp-url" placeholder="Paste your ASU ProQuest Barron’s page address" style="flex:1; min-width:260px;">
        <button class="primary" id="bp-open">Open Barron’s →</button>
      </div>
      <p style="color:var(--muted); font-size:13px; margin-bottom:0;">
        Find Barron’s in ASU’s library ProQuest once, copy the address from your browser’s address bar, and paste it above.
        It’s saved on this computer. The button opens it in a new tab, where you sign in with your normal ASU login.
      </p>
    </div>
    <div class="panel">
      <div class="row" style="justify-content:space-between;">
        <h2 style="margin:0;">Top finance topics this week</h2>
        <button class="ghost" id="bp-topics-refresh">Refresh</button>
      </div>
      <p style="color:var(--muted); font-size:13px;">
        Built from Yahoo Finance’s earnings, IPO and economic-release calendars: up to 5 per day, ranked by size and importance.
        Copy a topic, open Barron’s with the button above, and search for it there.
      </p>
      <div id="bp-topics"></div>
    </div>
    <div class="panel">
      <h2>Add an article to a day’s top 5</h2>
      <div class="row">
        <input type="date" id="bp-date">
        <input type="text" id="bp-title" placeholder="Article title" style="flex:1; min-width:220px;">
        <input type="text" id="bp-link" placeholder="ProQuest link to the article" style="flex:1; min-width:220px;">
        <button class="action" id="bp-add">Add</button>
      </div>
      <p style="color:var(--muted); font-size:13px; margin-bottom:0;">
        Tip: on a ProQuest article page, use its Permalink (or Document link) option and paste that link here.
        Your titles and links are saved in this browser only.
      </p>
    </div>
    <div id="bp-status"></div>
    <div id="bp-list"></div>
  </section>

</main>

<footer>Runs entirely on your machine. Not investment advice. Paper trading is simulated — no real money involved.</footer>

<script src="https://s3.tradingview.com/tv.js"></script>
<script>
const $ = (id) => document.getElementById(id);
function fmt(x, d=2) { if (x===null||x===undefined) return '—'; return Number(x).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:0}); }
function fmtPct(x) { if (x===null||x===undefined) return '—'; return (x*100).toFixed(2)+'%'; }
function fmtMoney(x) { if (x===null||x===undefined) return '—'; const sign = x<0?'-':''; return sign+'$'+fmt(Math.abs(x)); }
function plClass(x) { return x>=0 ? 'up' : 'down'; }

// ---------- TABS ----------
document.querySelectorAll('nav.tabs button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('nav.tabs button').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tabpanel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    $('tab-'+btn.dataset.tab).classList.add('active');
    document.querySelector('main').classList.toggle('wide', btn.dataset.tab === 'charts');
    document.querySelector('header').classList.toggle('wide', btn.dataset.tab === 'charts');
  });
});

// ---------- RESEARCH TAB ----------
$('r-avkey').value = localStorage.getItem('av_key') || '';
$('r-avkey').addEventListener('change', () => localStorage.setItem('av_key', $('r-avkey').value));

function trendBadge(t){ if(t==='uptrend') return '<span class="up">▲ Uptrend</span>'; if(t==='downtrend') return '<span class="down">▼ Downtrend</span>'; return t||'—'; }

function renderResearch(report) {
  const f=report.fundamentals, t=report.technicals, c=report.cross_check;
  const stat=(l,v)=>`<div class="stat"><div class="label">${l}</div><div class="value">${v}</div></div>`;
  let html = `<div class="panel"><div class="row" style="align-items:baseline;">
      <span class="mono" style="font-size:26px; font-weight:700;">${report.ticker}</span>
      <span class="mono up" style="font-size:22px; font-weight:700;">$${fmt(t.last_price)}</span>
      ${trendBadge(t.trend)}
    </div><div style="color:var(--muted); margin-top:6px;">${f.name||''} ${f.sector?'· '+f.sector:''}</div></div>`;

  html += `<div class="panel"><h2>Technicals</h2><div class="grid">
    ${stat('50-day SMA', fmt(t.sma_50))} ${stat('200-day SMA', fmt(t.sma_200))}
    ${stat('RSI (14d)', fmt(t.rsi_14,1))} ${stat('Volatility', fmtPct(t.volatility_annualized))}
    ${stat('52wk High', fmt(t.high_52wk))} ${stat('52wk Low', fmt(t.low_52wk))}
  </div></div>`;

  html += `<div class="panel"><h2>Fundamentals</h2><div class="grid">
    ${stat('Market Cap', f.market_cap? '$'+fmt(f.market_cap/1e9,2)+'B':'—')}
    ${stat('P/E', fmt(f.pe_ratio))} ${stat('Fwd P/E', fmt(f.forward_pe))}
    ${stat('PEG', fmt(f.peg_ratio))} ${stat('P/B', fmt(f.price_to_book))}
    ${stat('Margin', fmtPct(f.profit_margin))} ${stat('Rev Growth', fmtPct(f.revenue_growth))}
    ${stat('Div Yield', fmtPct(f.dividend_yield))}
  </div></div>`;

  if (c) html += `<div class="panel"><h2>Cross-Source Check</h2>
    <table><tr><th>Source</th><th>Price</th></tr>
    <tr><td>Yahoo Finance</td><td>$${fmt(c.yahoo_last_price)}</td></tr>
    <tr><td>Alpha Vantage</td><td>$${fmt(c.alpha_vantage_last_price)}</td></tr></table>
    <p style="color:var(--muted); font-size:13px; margin-top:8px;">${c.note||''}</p></div>`;

  html += `<div class="panel"><h2>Signals</h2>${(report.signals||[]).map(s=>`<p>🟢 ${s}</p>`).join('')||'<p>None flagged</p>'}</div>`;
  html += `<div class="panel"><h2>Risks</h2>${(report.risks||[]).map(s=>`<p>🔴 ${s}</p>`).join('')||'<p>None flagged</p>'}</div>`;
  html += `<div class="panel"><h2>Summary</h2><p>${report.summary||''}</p></div>`;
  $('r-results').innerHTML = html;
}

$('r-go').addEventListener('click', async () => {
  const ticker = $('r-ticker').value.trim().toUpperCase();
  if (!ticker) return;
  $('r-status').innerHTML = '<span class="spinner"></span> Researching…';
  $('r-results').innerHTML = '';
  try {
    const resp = await fetch(`/api/research?ticker=${encodeURIComponent(ticker)}&avkey=${encodeURIComponent($('r-avkey').value.trim())}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Something went wrong.');
    $('r-status').textContent = 'Updated ' + new Date(data.generated_at).toLocaleString();
    renderResearch(data);
  } catch(e) { $('r-status').innerHTML = `<span class="error">⚠ ${e.message}</span>`; }
});

// ---------- PAPER TRADING TAB ----------
async function loadPortfolio() {
  const resp = await fetch('/api/portfolio');
  const p = await resp.json();

  $('pt-summary').innerHTML = `
    <div class="stat"><div class="label">Cash</div><div class="value">${fmtMoney(p.cash)}</div></div>
    <div class="stat"><div class="label">Total Value</div><div class="value">${fmtMoney(p.total_value)}</div></div>
    <div class="stat"><div class="label">Total Return</div><div class="value ${plClass(p.total_return_pct)}">${fmtPct(p.total_return_pct)}</div></div>`;

  $('pt-long').innerHTML = p.long.length ? `<table><tr><th>Ticker</th><th>Shares</th><th>Avg Cost</th><th>Price</th><th>Value</th><th>Unrl. P&amp;L</th></tr>
    ${p.long.map(x=>`<tr><td>${x.ticker}</td><td>${fmt(x.shares)}</td><td>$${fmt(x.avg_cost)}</td><td>$${fmt(x.price)}</td><td>${fmtMoney(x.value)}</td><td class="${plClass(x.unrealized_pl)}">${fmtMoney(x.unrealized_pl)}</td></tr>`).join('')}
    </table>` : '<p style="color:var(--muted);">No long positions.</p>';

  $('pt-short').innerHTML = p.short.length ? `<table><tr><th>Ticker</th><th>Shares</th><th>Avg Price</th><th>Price</th><th>Liability</th><th>Unrl. P&amp;L</th></tr>
    ${p.short.map(x=>`<tr><td>${x.ticker}</td><td>${fmt(x.shares)}</td><td>$${fmt(x.avg_price)}</td><td>$${fmt(x.price)}</td><td>${fmtMoney(x.liability)}</td><td class="${plClass(x.unrealized_pl)}">${fmtMoney(x.unrealized_pl)}</td></tr>`).join('')}
    </table>` : '<p style="color:var(--muted);">No short positions.</p>';

  $('pt-options').innerHTML = p.options.length ? `<table><tr><th>Ticker</th><th>Type</th><th>Strike</th><th>Exp</th><th>Contracts</th><th>Entry Prem.</th><th>Now</th><th>Value</th><th>Unrl. P&amp;L</th><th></th></tr>
    ${p.options.map(x=>`<tr><td>${x.ticker}</td><td>${x.type}</td><td>$${fmt(x.strike)}</td><td>${x.expiration}</td><td>${fmt(x.contracts)}</td><td>$${fmt(x.entry_premium)}</td><td>$${fmt(x.current_premium)}</td><td>${fmtMoney(x.value)}</td><td class="${plClass(x.unrealized_pl)}">${fmtMoney(x.unrealized_pl)}</td><td><button class="ghost" onclick="closePut('${x.id}')">Close</button></td></tr>`).join('')}
    </table>` : '<p style="color:var(--muted);">No options positions.</p>';

  $('pt-log').innerHTML = p.log.length ? `<table><tr><th>Time</th><th>Action</th><th>Ticker</th><th>Qty</th><th>Price</th><th>Details</th></tr>
    ${p.log.map(x=>`<tr><td>${new Date(x.time).toLocaleString()}</td><td>${x.action}</td><td>${x.ticker}</td><td>${fmt(x.qty)}</td><td>$${fmt(x.price)}</td><td>${x.details||''}</td></tr>`).join('')}
    </table>` : '<p style="color:var(--muted);">No trades yet.</p>';
}

async function closePut(id) {
  $('pt-trade-status').innerHTML = '<span class="spinner"></span> Closing…';
  try {
    const resp = await fetch('/api/trade', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action:'sell_put', opt:{id}})});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Failed to close.');
    $('pt-trade-status').textContent = 'Closed.';
    loadPortfolio();
  } catch(e) { $('pt-trade-status').innerHTML = `<span class="error">⚠ ${e.message}</span>`; }
}

$('pt-action').addEventListener('change', async () => {
  const isPut = $('pt-action').value === 'buy_put';
  $('pt-put-fields').style.display = isPut ? 'flex' : 'none';
  if (isPut) await loadExpirationsForPut();
});
$('pt-ticker').addEventListener('change', async () => {
  if ($('pt-action').value === 'buy_put') await loadExpirationsForPut();
});

async function loadExpirationsForPut() {
  const ticker = $('pt-ticker').value.trim().toUpperCase();
  if (!ticker) return;
  const resp = await fetch(`/api/options?ticker=${encodeURIComponent(ticker)}`);
  const data = await resp.json();
  $('pt-expiration').innerHTML = (data.expirations||[]).map(e=>`<option value="${e}">${e}</option>`).join('');
  if (data.expirations && data.expirations.length) await loadStrikesForPut();
}
$('pt-expiration').addEventListener('change', loadStrikesForPut);
async function loadStrikesForPut() {
  const ticker = $('pt-ticker').value.trim().toUpperCase();
  const exp = $('pt-expiration').value;
  if (!ticker || !exp) return;
  const resp = await fetch(`/api/options?ticker=${encodeURIComponent(ticker)}&expiration=${encodeURIComponent(exp)}`);
  const data = await resp.json();
  $('pt-strike').innerHTML = (data.puts||[]).map(p=>`<option value="${p.strike}">$${p.strike} (last $${p.lastPrice ?? '—'})</option>`).join('');
}

$('pt-submit').addEventListener('click', async () => {
  const action = $('pt-action').value;
  const ticker = $('pt-ticker').value.trim().toUpperCase();
  const qty = parseFloat($('pt-qty').value);
  if (!ticker || !qty) return;
  $('pt-trade-status').innerHTML = '<span class="spinner"></span> Submitting…';
  const body = { action, ticker, qty };
  if (action === 'buy_put') body.opt = { expiration: $('pt-expiration').value, strike: $('pt-strike').value };
  try {
    const resp = await fetch('/api/trade', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Trade failed.');
    $('pt-trade-status').textContent = 'Done.';
    loadPortfolio();
  } catch(e) { $('pt-trade-status').innerHTML = `<span class="error">⚠ ${e.message}</span>`; }
});

$('pt-reset').addEventListener('click', async () => {
  if (!confirm('Reset paper trading account back to $10,000? This clears all positions and history.')) return;
  await fetch('/api/reset', {method:'POST'});
  loadPortfolio();
});

// ---------- CHARTS & GAPS TAB ----------
let tvWidget = null;
$('c-go').addEventListener('click', async () => {
  const ticker = $('c-ticker').value.trim().toUpperCase();
  if (!ticker) return;

  document.getElementById('tv_chart').innerHTML = '';
  new TradingView.widget({
    "autosize": true, "symbol": ticker, "interval": "10", "timezone": "Etc/UTC",
    "theme": "dark", "style": "1", "locale": "en", "toolbar_bg": "#000000", "backgroundColor": "#000000",
    "overrides": {"paneProperties.background": "#000000", "paneProperties.backgroundType": "solid",
                  "paneProperties.vertGridProperties.color": "#1a1a1a", "paneProperties.horzGridProperties.color": "#1a1a1a"},
    "enable_publishing": false, "hide_top_toolbar": false, "container_id": "tv_chart"
  });

  $('g-status').innerHTML = '<span class="spinner"></span> Scanning for gaps…';
  $('g-results').innerHTML = '';
  try {
    const resp = await fetch(`/api/gaps?ticker=${encodeURIComponent(ticker)}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Something went wrong.');
    $('g-status').textContent = `${data.gaps.length} gap(s) found in the last ~30 trading days.`;
    gapData = data.gaps; renderGaps();
  } catch(e) { $('g-status').innerHTML = `<span class="error">⚠ ${e.message}</span>`; }
});

let gapData = [];
function gapDuration(g) {
  if (g.fill_minutes === 0) return 'Same candle';
  if (g.fill_minutes < 60) return `${Math.round(g.fill_minutes)} min`;
  if (g.fill_minutes < 1440) return `${(g.fill_minutes / 60).toFixed(1)} hrs`;
  return `${g.fill_days} day${g.fill_days === 1 ? '' : 's'}`;
}
function renderGaps() {
  const f = $('g-filter').value;
  const list = gapData.filter(g => f === 'all' || (f === 'filled' ? g.filled : !g.filled));
  const nFilled = gapData.filter(g => g.filled).length;
  $('g-summary').textContent = gapData.length ? `${nFilled} of ${gapData.length} filled, ${gapData.length - nFilled} still open.` : '';
  $('g-results').innerHTML = list.length ? `<div style="overflow-x:auto;"><table><tr><th>Time</th><th>Prev Close</th><th>Open</th><th>Gap</th><th>Direction</th><th>Type</th><th>Status</th><th>Filled at</th><th>Time to fill</th></tr>
    ${list.map(g => `<tr><td>${new Date(g.time).toLocaleString()}</td><td>$${fmt(g.prev_close)}</td><td>$${fmt(g.open)}</td>
      <td class="${g.direction==='up'?'up':'down'}">${fmtPct(g.gap_pct)}</td>
      <td>${g.direction==='up'?'▲ Up':'▼ Down'}</td><td>${g.overnight?'Overnight':'Intraday'}</td>
      <td>${g.filled ? '<span class="up">✓ Filled</span>' : `<span class="down">Open</span> <span class="muted small">(${(g.closed_pct*100).toFixed(0)}% closed)</span>`}</td>
      <td>${g.filled ? new Date(g.filled_time).toLocaleString() : '—'}</td>
      <td>${g.filled ? gapDuration(g) : `open ${g.open_for_days} day${g.open_for_days === 1 ? '' : 's'}`}</td></tr>`).join('')}
    </table></div>` : (gapData.length ? '<p style="color:var(--muted);">No gaps match that filter.</p>' : '<p style="color:var(--muted);">No gaps of 0.3%+ found.</p>');
}
$('g-filter').addEventListener('change', renderGaps);

// ---------- TERMINAL TAB ----------
const termHistory = []; let termIdx = 0;
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function cell(v) { const s = String(v ?? ''); return /^https?:\/\/\S+$/.test(s) ? `<a href="${esc(s)}" target="_blank" rel="noopener noreferrer">open</a>` : esc(s); }
function renderChart(ch) {
  const W = 680, H = 270, padL = 58, padR = 18, padT = 30, padB = 42;
  const vals = ch.values, n = vals.length;
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const pad = (mx - mn) * 0.15 || Math.abs(mx) * 0.15 || 1;
  const lo = mn - pad, hi = mx + pad, span = hi - lo;
  const y = v => padT + (hi - v) / span * (H - padT - padB);
  const slot = (W - padL - padR) / n;
  const xs = vals.map((_, i) => padL + slot * (i + 0.5));
  const mono = "font:11px 'JetBrains Mono',monospace;";
  let s = `<svg viewBox="0 0 ${W} ${H}" style="width:100%; max-width:${W}px; height:auto; display:block; margin:6px 0 14px;" role="img" aria-label="Line chart">`;
  s += `<text x="${padL}" y="14" style="fill:var(--muted); ${mono}">${esc(ch.unit)}</text>`;
  for (let i = 0; i <= 3; i++) {
    const t = lo + span * i / 3, yy = y(t);
    s += `<line x1="${padL}" x2="${W - padR}" y1="${yy}" y2="${yy}" style="stroke:var(--border);"/>` +
         `<text x="${padL - 8}" y="${yy + 4}" text-anchor="end" style="fill:var(--muted); ${mono}">${esc(fmt(t, 2))}</text>`;
  }
  if (lo < 0 && hi > 0) {
    s += `<line x1="${padL}" x2="${W - padR}" y1="${y(0)}" y2="${y(0)}" stroke-dasharray="4 4" style="stroke:var(--muted);"/>`;
  }
  if (n > 1) {
    s += `<polyline points="${xs.map((x, i) => x + ',' + y(vals[i])).join(' ')}" fill="none" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" style="stroke:var(--accent-2);"/>`;
  }
  vals.forEach((v, i) => {
    const yv = y(v), above = yv - 12 >= padT - 4;
    s += `<circle cx="${xs[i]}" cy="${yv}" r="4.5" style="fill:${v >= 0 ? 'var(--accent-2)' : 'var(--danger)'}; stroke:var(--panel); stroke-width:2;"/>` +
         `<text x="${xs[i]}" y="${above ? yv - 10 : yv + 18}" text-anchor="middle" style="fill:var(--text); ${mono}">${esc(fmt(v, 2))}</text>` +
         `<text x="${xs[i]}" y="${H - padB + 18}" text-anchor="middle" style="fill:var(--muted); ${mono}">${esc(ch.labels[i])}</text>`;
  });
  return s + `</svg>`;
}
function termTable(cols, rows) {
  if (!cols || !cols.length) return '';
  return `<div style="overflow-x:auto;"><table><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr>` +
    rows.map(r => `<tr>${r.map(v => `<td>${cell(v)}</td>`).join('')}</tr>`).join('') + `</table></div>`;
}
function renderTerm(d, cmd) {
  let h = `<div class="panel"><h2>${esc(d.title)}</h2>`;
  if (d.chart) h += renderChart(d.chart);
  h += termTable(d.columns, d.rows);
  (d.sections || []).forEach(sec => {
    h += `<h3 style="margin:22px 0 6px; font-size:12px; letter-spacing:1px; text-transform:uppercase; color:var(--accent-2);">${esc(sec.title)}</h3>` + termTable(sec.columns, sec.rows);
  });
  if (d.note) h += `<p style="color:var(--muted); font-size:13px; margin-top:10px;">${esc(d.note)}</p>`;
  return h + `</div>`;
}
async function runTerm() {
  const cmd = $('term-input').value.trim();
  if (!cmd) return;
  termHistory.push(cmd); termIdx = termHistory.length;
  $('term-input').value = '';
  if (cmd.toUpperCase().split(/\s+/)[0] === 'PORT') { portOpen(cmd.split(/\s+/).slice(1).join(' ')); return; }
  $('term-status').innerHTML = '<span class="spinner"></span> Running ' + esc(cmd.toUpperCase()) + '…';
  try {
    const resp = await fetch('/api/terminal?cmd=' + encodeURIComponent(cmd));
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Command failed.');
    $('term-status').textContent = '';
    $('term-output').innerHTML = renderTerm(data, cmd) + $('term-output').innerHTML;
  } catch (e) { $('term-status').innerHTML = `<span class="error">⚠ ${esc(e.message)}</span>`; }
}
$('term-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') runTerm();
  else if (e.key === 'ArrowUp' && termHistory.length) { termIdx = Math.max(0, termIdx - 1); $('term-input').value = termHistory[termIdx]; e.preventDefault(); }
  else if (e.key === 'ArrowDown' && termHistory.length) { termIdx = Math.min(termHistory.length, termIdx + 1); $('term-input').value = termHistory[termIdx] || ''; e.preventDefault(); }
});

// ---------- BARRONS TAB ----------
const BP_KEY = 'barrons_picks', BP_URL_KEY = 'barrons_proquest_url';
function bpLoad() { try { return JSON.parse(localStorage.getItem(BP_KEY)) || []; } catch (e) { return []; } }
function bpSave(a) { localStorage.setItem(BP_KEY, JSON.stringify(a)); }
function bpToday() { const d = new Date(); return d.getFullYear() + '-' + String(d.getMonth()+1).padStart(2,'0') + '-' + String(d.getDate()).padStart(2,'0'); }
function bpErr(msg) { $('bp-status').innerHTML = `<span class="error">⚠ ${esc(msg)}</span>`; }
const bpIsUrl = (u) => /^https?:\/\//i.test(u);
$('bp-date').value = bpToday();
$('bp-url').value = localStorage.getItem(BP_URL_KEY) || '';
$('bp-url').addEventListener('change', () => localStorage.setItem(BP_URL_KEY, $('bp-url').value.trim()));
$('bp-open').addEventListener('click', () => {
  const u = $('bp-url').value.trim();
  if (!bpIsUrl(u)) return bpErr('Paste your ASU ProQuest Barron’s page address first (it must start with https://).');
  localStorage.setItem(BP_URL_KEY, u);
  $('bp-status').textContent = '';
  window.open(u, '_blank', 'noopener');
});
function bpRender() {
  const all = bpLoad().map((p, i) => ({...p, i}));
  if (!all.length) { $('bp-list').innerHTML = '<div class="panel"><p style="color:var(--muted); margin:0;">No articles saved yet.</p></div>'; return; }
  const dates = [...new Set(all.map(p => p.date))].sort().reverse();
  $('bp-list').innerHTML = dates.map(d => {
    const items = all.filter(p => p.date === d);
    const label = new Date(d + 'T12:00:00').toLocaleDateString(undefined, {weekday:'long', year:'numeric', month:'long', day:'numeric'});
    return `<div class="panel"><h2>${esc(label)} — ${items.length}/5</h2>` +
      items.map((p, n) => `<div class="news-row"><span class="mono" style="color:var(--muted);">${n+1}.</span>
        <a href="${esc(p.url)}" target="_blank" rel="noopener noreferrer">${esc(p.title)}</a>
        <button class="ghost" onclick="bpDelete(${p.i})">✕</button></div>`).join('') + `</div>`;
  }).join('');
}
function bpDelete(i) { const a = bpLoad(); a.splice(i, 1); bpSave(a); bpRender(); }
$('bp-add').addEventListener('click', () => {
  const title = $('bp-title').value.trim(), url = $('bp-link').value.trim(), date = $('bp-date').value || bpToday();
  if (!title) return bpErr('Enter the article title.');
  if (!bpIsUrl(url)) return bpErr('Paste the article’s ProQuest link (it must start with https://).');
  const all = bpLoad();
  if (all.filter(p => p.date === date).length >= 5) return bpErr('That day already has 5 articles. Remove one to add another.');
  all.push({date, title, url}); bpSave(all);
  $('bp-title').value = ''; $('bp-link').value = ''; $('bp-status').textContent = '';
  bpRender();
});
bpRender();

let bpTopicsLoaded = false;
async function bpLoadTopics(force) {
  if (bpTopicsLoaded && !force) return;
  $('bp-topics').innerHTML = '<span class="spinner"></span> Loading this week’s calendar…';
  try {
    const resp = await fetch('/api/topics' + (force ? '?refresh=1' : ''));
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Couldn’t load topics.');
    bpTopicsLoaded = true;
    $('bp-topics').innerHTML = data.days.map(day => {
      const lab = new Date(day.date + 'T12:00:00').toLocaleDateString(undefined, {weekday:'long', month:'short', day:'numeric'})
        + (day.date === bpToday() ? ' — Today' : '');
      const items = day.topics.length ? day.topics.map((t, n) => `<div class="news-row">
          <span class="mono" style="color:var(--muted);">${n+1}.</span><span class="kind">${esc(t.kind)}</span>
          <span style="flex:1;"><b>${esc(t.title)}</b><br><span style="color:var(--muted); font-size:12px;">${esc(t.detail)}</span></span>
          <button class="ghost" data-t="${esc(t.title)}" onclick="bpCopy(this)">Copy</button></div>`).join('')
        : '<p style="color:var(--muted); margin:0;">Nothing major scheduled.</p>';
      return `<h3 style="margin:18px 0 4px; font-size:13px; color:var(--accent);">${esc(lab)}</h3>${items}`;
    }).join('');
  } catch (e) { $('bp-topics').innerHTML = `<span class="error">⚠ ${esc(e.message)}</span>`; }
}
function bpCopy(btn) {
  const t = btn.dataset.t;
  const done = () => { btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = 'Copy'; }, 1200); };
  if (navigator.clipboard) navigator.clipboard.writeText(t).then(done).catch(() => window.prompt('Copy this:', t));
  else window.prompt('Copy this:', t);
}
$('bp-topics-refresh').addEventListener('click', () => bpLoadTopics(true));
document.querySelector('nav.tabs button[data-tab="barrons"]').addEventListener('click', () => bpLoadTopics(false));

// ---------- PORT (portfolio builder) ----------
const portBlank = () => ({ticker: '', shares: '', cost: ''});
let portRows = [portBlank()], portVal = null;
function portMsg(h) { $('port-status').innerHTML = h; }
function portErr(m) { portMsg(`<span class="error">⚠ ${esc(m)}</span>`); }
async function portJson(url, opts) {
  const r = await fetch(url, opts), d = await r.json();
  if (!r.ok) throw new Error(d.error || 'Request failed.');
  return d;
}
const portPost = (url, body) => portJson(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
function portSet(i, k, v) { portRows[i][k] = v; }
function portCollect() {
  const sent = [], map = [];
  portRows.forEach((r, i) => {
    const t = String(r.ticker).trim();
    if (!t && String(r.shares).trim() === '') return;
    sent.push({ticker: t, shares: String(r.shares).trim() === '' ? null : Number(r.shares), cost: String(r.cost).trim() === '' ? null : Number(r.cost)});
    map.push(i);
  });
  return {sent, map};
}
function portRender() {
  const dash = '—', pc = (x) => x === null || x === undefined ? dash : `<span class="${plClass(x)}">${fmtPct(x)}</span>`;
  const mo = (x) => x === null || x === undefined ? dash : fmtMoney(x);
  const pl = (x) => x === null || x === undefined ? dash : `<span class="${plClass(x)}">${fmtMoney(x)}</span>`;
  const body = portRows.map((r, i) => {
    const v = portVal && portVal.byRow[i];
    return `<tr>
      <td><input type="text" value="${esc(r.ticker)}" placeholder="AAPL" style="width:90px; text-transform:uppercase;" oninput="portSet(${i},'ticker',this.value)"></td>
      <td><input type="number" value="${esc(r.shares)}" step="any" placeholder="0" style="width:100px;" oninput="portSet(${i},'shares',this.value)"></td>
      <td><input type="number" value="${esc(r.cost)}" step="any" min="0" placeholder="optional" style="width:110px;" oninput="portSet(${i},'cost',this.value)"></td>
      <td>${v && v.price !== null ? '$' + fmt(v.price) : dash}</td><td>${v ? pc(v.chg_pct) : dash}</td>
      <td>${v ? mo(v.value) : dash}</td><td>${v && v.weight !== null ? (v.weight * 100).toFixed(1) + '%' : dash}</td>
      <td>${v ? pl(v.pl) : dash}</td><td>${v ? pc(v.tot_rtn) : dash}</td><td>${v ? pc(v.ctr) : dash}</td>
      <td><button class="ghost" onclick="portRemove(${i})">✕</button></td></tr>`;
  }).join('');
  const tt = portVal && portVal.totals;
  const foot = tt ? `<tr style="font-weight:700;"><td>Total</td><td></td><td></td><td></td><td></td><td>${mo(tt.value)}</td><td></td><td>${pl(tt.pl)}</td><td>${pc(tt.tot_rtn)}</td><td>${pc(tt.tot_rtn)}</td><td></td></tr>` : '';
  $('port-table').innerHTML = `<table><tr><th>Ticker</th><th>Shares</th><th>Cost / share</th><th>Price</th><th>Day %</th><th>Mkt Val</th><th>Weight</th><th>P&amp;L</th><th>Tot Rtn</th><th>CTR</th><th></th></tr>${body}${foot}</table>
    <p class="muted small" style="margin:8px 0 0;">Tot Rtn = price return vs your cost (a short gains when the price falls). CTR = contribution to return: each holding's P&amp;L divided by the portfolio's total cost, so the CTR column adds up to the portfolio's Tot Rtn.</p>`;
  const t = portVal && portVal.totals;
  const stat = (l, v) => `<div class="stat"><div class="label">${l}</div><div class="value">${v}</div></div>`;
  $('port-totals').innerHTML = t ? stat('Market value', fmtMoney(t.value)) +
    (t.day_pl !== null ? stat('Day change', `<span class="${plClass(t.day_pl)}">${fmtMoney(t.day_pl)}${t.day_pct !== null ? ' (' + fmtPct(t.day_pct) + ')' : ''}</span>`) : '') +
    (t.pl !== null ? stat('Total P&amp;L', `<span class="${plClass(t.pl)}">${fmtMoney(t.pl)}</span>`) + stat('Tot Rtn', t.tot_rtn !== null ? `<span class="${plClass(t.tot_rtn)}">${fmtPct(t.tot_rtn)}</span>` : '—') + stat('Cost basis', fmtMoney(t.cost)) : '') : '';
}
function portAdd() { portRows.push(portBlank()); portRender(); }
function portRemove(i) {
  portRows.splice(i, 1);
  if (!portRows.length) portRows.push(portBlank());
  const had = !!portVal; portVal = null; portRender();
  if (had) portValue();
}
async function portValue() {
  const {sent, map} = portCollect();
  if (!sent.length) return portErr('Add at least one holding.');
  portMsg('<span class="spinner"></span> Getting prices…');
  try {
    const d = await portPost('/api/port/value', {holdings: sent});
    const byRow = {}; d.rows.forEach((r, j) => { byRow[map[j]] = r; });
    portVal = {byRow, totals: d.totals}; portRender();
    portMsg(d.missing && d.missing.length ? `<span class="error">⚠ No price found for: ${esc(d.missing.join(', '))}</span>` : '');
  } catch (e) { portErr(e.message); }
}
function portFillSaved(names, selected) {
  $('port-saved').innerHTML = '<option value="">Load a saved portfolio…</option>' +
    (names || []).map(n => `<option value="${esc(n)}"${n === selected ? ' selected' : ''}>${esc(n)}</option>`).join('');
}
function portNew() { portRows = [portBlank()]; portVal = null; $('port-name').value = ''; $('port-saved').value = ''; $('port-bench').value = ''; $('port-analysis').innerHTML = ''; $('port-active-out').innerHTML = ''; $('port-stress').innerHTML = ''; portMsg(''); portRender(); }
async function portLoad(name) {
  try {
    const d = await portJson('/api/port/get?name=' + encodeURIComponent(name));
    portRows = d.holdings.map(h => ({ticker: h.ticker, shares: String(h.shares), cost: h.cost === null ? '' : String(h.cost)}));
    portVal = null; $('port-name').value = d.name; $('port-saved').value = d.name; $('port-bench').value = d.benchmark || '';
    $('port-analysis').innerHTML = ''; $('port-active-out').innerHTML = ''; $('port-stress').innerHTML = ''; portRender();
    await portValue();
  } catch (e) { portErr(e.message); }
}
async function portSave() {
  const name = $('port-name').value.trim();
  if (!name) return portErr('Give the portfolio a name first.');
  const {sent} = portCollect();
  if (!sent.length) return portErr('Add at least one holding before saving.');
  try {
    const d = await portPost('/api/port/save', {name, holdings: sent, benchmark: $('port-bench').value.trim()});
    portFillSaved(d.names, d.name); $('port-name').value = d.name;
    portMsg(`<span class="up">✓ ${d.replaced ? 'Updated' : 'Saved'} “${esc(d.name)}” (${sent.length} holding${sent.length === 1 ? '' : 's'}).</span>`);
  } catch (e) { portErr(e.message); }
}
async function portDelete() {
  const name = $('port-saved').value || $('port-name').value.trim();
  if (!name) return portErr('Pick a saved portfolio to delete.');
  if (!confirm(`Delete the saved portfolio “${name}”? This can’t be undone.`)) return;
  try {
    const d = await portPost('/api/port/delete', {name});
    portFillSaved(d.names, ''); portNew(); portMsg(`<span class="up">✓ Deleted “${esc(name)}”.</span>`);
  } catch (e) { portErr(e.message); }
}
async function portOpen(name) {
  const fresh = $('port-panel').style.display === 'none';
  $('port-panel').style.display = 'block';
  if (fresh) { portRender(); }
  try { portFillSaved((await portJson('/api/port/list')).names, ''); } catch (e) { portErr(e.message); }
  if (name) await portLoad(name); else if (fresh) portNew();
  $('port-panel').scrollIntoView({behavior: 'smooth', block: 'nearest'});
}
const pct1 = (x, d = 2) => x === null || x === undefined ? '—' : (x * 100).toFixed(d) + '%';
const num2 = (x) => x === null || x === undefined ? '—' : Number(x).toFixed(2);
const covNote = (c) => c < 0.995 ? `<div class="sub">${Math.round(c * 100)}% of weight has data</div>` : '';
function lineChart(ch, bench) {
  const W = 900, H = 300, L = 52, R = 14, T = 14, B = 28;
  const all = ch.portfolio.concat(ch.benchmark), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.08 || 1;
  const y0 = lo - pad, y1 = hi + pad, n = ch.dates.length;
  const X = i => L + (W - L - R) * (n <= 1 ? 0 : i / (n - 1)), Y = v => T + (H - T - B) * (1 - (v - y0) / (y1 - y0));
  const path = a => a.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join(' ');
  let s = `<svg viewBox="0 0 ${W} ${H}" style="width:100%; height:auto; display:block;" role="img" aria-label="Portfolio vs benchmark, indexed to 100">`;
  for (let k = 0; k <= 4; k++) {
    const v = y0 + (y1 - y0) * k / 4, y = Y(v);
    s += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#2a2a2a"/><text x="${L - 6}" y="${y + 4}" fill="#b87420" font-size="11" text-anchor="end">${v.toFixed(0)}</text>`;
  }
  s += `<text x="${L}" y="${H - 8}" fill="#b87420" font-size="11">${ch.dates[0]}</text><text x="${W - R}" y="${H - 8}" fill="#b87420" font-size="11" text-anchor="end">${ch.dates[n - 1]}</text>`;
  s += `<path d="${path(ch.benchmark)}" fill="none" stroke="#b87420" stroke-width="1.6" stroke-dasharray="5 3"/><path d="${path(ch.portfolio)}" fill="none" stroke="#33ffb8" stroke-width="2"/></svg>`;
  return s + `<div class="small" style="margin-top:4px;"><span style="color:#33ffb8;">━ Portfolio</span> &nbsp; <span style="color:#b87420;">╌ ${esc(bench)}</span> &nbsp; <span class="muted">(both start at 100; total return with dividends)</span></div>`;
}
let portCmp = null;
function portPeriodChart() {
  const p = $('port-period') ? $('port-period').value : '1Y';
  const ch = portCmp && portCmp.charts[p];
  $('port-chart').innerHTML = ch ? lineChart(ch, portCmp.bench) : '<p class="muted small">Not enough history for this period.</p>';
}
function portAnalysisRender(d) {
  const m = d.metrics, stat = (l, v, c) => `<div class="stat"><div class="label">${l}</div><div class="value">${v}</div>${covNote(c)}</div>`;
  let h = `<h2 style="margin:22px 0 10px;">Aggregate metrics (weighted by market value)</h2><div class="grid">` +
    stat('Dividend yield', pct1(m.yield.value), m.yield.coverage) + stat('P/E (trailing)', m.pe.value === null ? '—' : num2(m.pe.value) + 'x', m.pe.coverage) +
    stat('Debt / equity', m.de.value === null ? '—' : num2(m.de.value) + 'x', m.de.coverage) + stat('Current ratio', num2(m.cr.value), m.cr.coverage) +
    stat('ROE', pct1(m.roe.value), m.roe.coverage) + `</div>`;
  h += `<details style="margin-top:10px;"><summary class="linkish">Show each holding's numbers</summary><div style="overflow-x:auto;"><table><tr><th>Ticker</th><th>Weight</th><th>Div yield</th><th>P/E</th><th>D/E</th><th>Current ratio</th><th>ROE</th></tr>` +
    d.holdings_metrics.map(r => `<tr><td>${esc(r.ticker)}</td><td>${pct1(r.weight, 1)}</td><td>${pct1(r.yield)}</td><td>${num2(r.pe)}</td><td>${num2(r.de)}</td><td>${num2(r.cr)}</td><td>${pct1(r.roe)}</td></tr>`).join('') +
    `</table></div></details><p class="muted small">P/E is the harmonic weighted average (earnings yields averaged, then inverted); holdings with negative or missing earnings are left out, and the other metrics are averaged over the holdings that report them. ETFs usually don't report D/E, current ratio or ROE.</p>`;
  const v = d.var;
  h += `<h2 style="margin:26px 0 10px;">Value at risk (VaR)</h2>`;
  if (!v) h += `<p class="muted">Not enough price history (about 6 months) to compute VaR.</p>`;
  else {
    const cell = x => `${fmtMoney(x)} <span class="muted small">(${pct1(x / d.base, 2)})</span>`;
    h += `<div style="overflow-x:auto;"><table><tr><th>Confidence</th><th>1-day (historical)</th><th>1-day CVaR (avg loss beyond VaR)</th><th>10-day (historical)</th><th>1-day (parametric)</th><th>10-day (parametric)</th></tr>` +
      v.rows.map(r => `<tr><td>${(r.conf * 100).toFixed(0)}%</td><td class="down">${cell(r.hist_1d)}</td><td class="down">${cell(r.cvar_1d)}</td><td class="down">${cell(r.hist_10d)}</td><td class="down">${cell(r.param_1d)}</td><td class="down">${cell(r.param_10d)}</td></tr>`).join('') + `</table></div>`;
    if (v.bench) h += `<p class="small muted">Same-size position in ${esc(d.benchmark)}, 1-day historical VaR: ` + v.bench.map(b => `${(b.conf * 100).toFixed(0)}%: ${fmtMoney(b.hist_1d)}`).join(' &nbsp;|&nbsp; ') + `</p>`;
    const st = v.stats;
    h += `<div class="grid" style="margin-top:10px;">` + stat('Annualized volatility', pct1(st.vol, 1), 1) + (st.beta !== null ? stat('Beta vs ' + esc(d.benchmark), num2(st.beta), 1) + stat('Correlation', num2(st.corr), 1) + stat(esc(d.benchmark) + ' volatility', pct1(st.bench_vol, 1), 1) : '') + `</div>`;
    h += `<p class="muted small">Replays today's positions through ${v.days} trading days of price moves (${v.from} to ${v.to}). "95% 1-day VaR of $X" means that on 1 day in 20 the historical record says you'd lose at least $X. 10-day figures scale the 1-day number by √10. Parametric assumes normally distributed returns, which understates crash risk; CVaR shows how bad the bad days averaged.` +
      (v.excluded.length ? ` Left out for short history: ${esc(v.excluded.join(', '))}.` : '') + `</p>`;
  }
  h += `<h2 style="margin:26px 0 10px;">Portfolio vs benchmark${d.benchmark ? ' (' + esc(d.benchmark) + ')' : ''}</h2>`;
  if (!d.compare) h += `<p class="muted">Enter a benchmark ETF above (SPY, QQQ, IWM, ...) and run the analysis again to compare.</p>`;
  else {
    h += `<div style="overflow-x:auto;"><table><tr><th>Period</th><th>Portfolio</th><th>${esc(d.benchmark)}</th><th>Difference</th></tr>` +
      d.compare.rows.map(r => `<tr><td>${r.period}</td><td>${r.portfolio === null ? '—' : `<span class="${plClass(r.portfolio)}">${pct1(r.portfolio)}</span>`}</td><td>${r.benchmark === null ? '—' : `<span class="${plClass(r.benchmark)}">${pct1(r.benchmark)}</span>`}</td><td>${r.excess === null ? '—' : `<span class="${plClass(r.excess)}">${r.excess >= 0 ? '+' : ''}${pct1(r.excess)}</span>`}${r.excluded ? ` <span class="muted small">(${r.excluded} holding${r.excluded === 1 ? '' : 's'} too new, left out)</span>` : ''}</td></tr>`).join('') + `</table></div>`;
    h += `<div class="row" style="margin:14px 0 6px;"><span class="muted small">Chart period:</span><select id="port-period">${PERIOD_LIST.map(p => `<option${p === '1Y' ? ' selected' : ''}>${p}</option>`).join('')}</select></div><div id="port-chart"></div>`;
    h += `<p class="muted small">Total returns include dividends. These replay your current share counts over each period (what today's portfolio would have done), not your actual trade history.</p>`;
  }
  if (d.missing.length) h += `<p class="error small">⚠ No price history for: ${esc(d.missing.join(', '))}</p>`;
  $('port-analysis').innerHTML = h;
  portCmp = d.compare ? {charts: d.compare.charts, bench: d.benchmark} : null;
  if ($('port-period')) { $('port-period').addEventListener('change', portPeriodChart); portPeriodChart(); }
}
const PERIOD_LIST = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y'];
async function portAnalyze() {
  const {sent} = portCollect();
  if (!sent.length) return portErr('Add at least one holding.');
  $('port-analysis').innerHTML = '<p><span class="spinner"></span> Running analysis (downloads price history, can take a few seconds)…</p>';
  try {
    const d = await portPost('/api/port/analyze', {holdings: sent, benchmark: $('port-bench').value.trim()});
    portAnalysisRender(d);
  } catch (e) { $('port-analysis').innerHTML = `<p class="error">⚠ ${esc(e.message)}</p>`; }
}
function portStressRender(d) {
  const ok = d.scenarios.filter(s => !s.unavailable);
  let h = `<div style="overflow-x:auto; margin-top:12px;"><table><tr><th></th><th>Event</th><th>Window</th><th>S&amp;P 500</th><th>Portfolio P&amp;L</th><th>Portfolio %</th><th>${d.benchmark ? esc(d.benchmark) : 'Benchmark'}</th></tr>`;
  d.scenarios.forEach((s, i) => {
    if (s.unavailable) { h += `<tr><td></td><td>${esc(s.name)}</td><td>${s.start} → ${s.end}</td><td colspan="4" class="muted">No S&amp;P 500 data for these dates.</td></tr>`; return; }
    h += `<tr><td class="linkish" onclick="const r=document.getElementById('st-${i}'); r.style.display = r.style.display==='none' ? 'table-row' : 'none';">▸</td><td>${esc(s.name)}</td><td>${s.start} → ${s.end}</td>
      <td class="${plClass(s.sp500)}">${pct1(s.sp500)}</td><td class="${plClass(s.pnl)}">${fmtMoney(s.pnl)}</td><td class="${plClass(s.pct)}">${pct1(s.pct)}${s.est_count ? ` <span class="muted small">(${s.est_count} est.)</span>` : ''}</td>
      <td>${s.benchmark === null || s.benchmark === undefined ? '<span class="muted">n/a</span>' : `<span class="${plClass(s.benchmark)}">${pct1(s.benchmark)}</span>`}</td></tr>
      <tr class="fold" id="st-${i}" style="display:none;"><td></td><td colspan="6"><table><tr><th>Holding</th><th>Mkt Val today</th><th>Move in event</th><th>P&amp;L</th></tr>` +
      s.rows.map(r => `<tr><td>${esc(r.ticker)}${r.est ? ' <span class="muted small">(est.)</span>' : ''}</td><td>${fmtMoney(r.value)}</td><td class="${plClass(r.ret)}">${pct1(r.ret)}</td><td class="${plClass(r.pnl)}">${fmtMoney(r.pnl)}</td></tr>`).join('') + `</table></td></tr>`;
  });
  h += `</table></div><p class="muted small">Applies each event's actual price moves to today's market values (portfolio % is of ${fmtMoney(d.base)}). Click ▸ for the holding-by-holding breakdown. "est." marks holdings that didn't trade through the whole window (e.g. a company that IPO'd later): they're estimated as beta × the S&amp;P 500's move, so treat those numbers as rough. Moves are total returns including dividends.</p>`;
  $('port-stress').innerHTML = h;
}
async function portStress() {
  const {sent} = portCollect();
  if (!sent.length) return portErr('Add at least one holding.');
  const cs = $('port-st-start').value.trim(), ce = $('port-st-end').value.trim();
  if ((cs || ce) && !(/^\d{4}-\d{2}-\d{2}$/.test(cs) && /^\d{4}-\d{2}-\d{2}$/.test(ce))) { $('port-stress').innerHTML = '<p class="error">⚠ Enter both custom dates as YYYY-MM-DD, or leave both blank.</p>'; return; }
  $('port-stress').innerHTML = '<p><span class="spinner"></span> Running stress tests (downloads decades of price history, can take a few seconds)…</p>';
  try {
    const d = await portPost('/api/port/stress', {holdings: sent, benchmark: $('port-bench').value.trim(), custom: cs ? {start: cs, end: ce} : null});
    portStressRender(d);
  } catch (e) { $('port-stress').innerHTML = `<p class="error">⚠ ${esc(e.message)}</p>`; }
}
$('port-analyze').addEventListener('click', portAnalyze);
$('port-stress-run').addEventListener('click', portStress);
function dBar(v, mx) {
  const w = Math.min(Math.abs(v) / (mx || 1), 1) * 50;
  return `<div style="position:relative; height:10px; width:150px; background:#111; border-radius:3px;"><div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:#555;"></div><div style="position:absolute; top:1px; bottom:1px; ${v >= 0 ? 'left:50%' : 'right:50%'}; width:${w}%; background:${v >= 0 ? 'var(--accent-2)' : '#7a7a7a'};"></div></div>`;
}
function statusTag(s) {
  const over = /over/i.test(s), under = /under/i.test(s);
  return `<span style="font-weight:700; color:${over || under ? 'var(--warn)' : 'var(--muted)'};">${over ? '▲ ' : under ? '▼ ' : ''}${esc(s)}</span>`;
}
function portActiveRender(d) {
  const t = d.tracking, stat = (l, v, sub) => `<div class="stat"><div class="label">${l}</div><div class="value">${v}</div>${sub ? `<div class="sub">${sub}</div>` : ''}</div>`;
  const b = esc(d.benchmark);
  let h = `<h2 style="margin:26px 0 10px;">Tracking error &amp; active risk vs ${b}</h2><div class="grid">` +
    stat('Tracking error (ann.)', pct1(t.te), `${d.n} days, ${d.from} to ${d.to}`) + stat('Last 3 months', pct1(t.te_3m)) + stat('Last 12 months', pct1(t.te_1y)) +
    stat('Active return (ann.)', t.active_return === null ? '—' : `<span class="${plClass(t.active_return)}">${t.active_return >= 0 ? '+' : ''}${pct1(t.active_return)}</span>`) +
    stat('Information ratio', num2(t.info_ratio)) + stat('R² vs benchmark', num2(t.r2), `beta ${num2(t.beta)}`) + `</div>`;
  h += `<p class="muted small">Tracking error is the annualized standard deviation of your daily return minus the benchmark's, which is your active risk. Roughly, two-thirds of the time your annual return should land within ± one tracking error of the benchmark's. Information ratio = active return ÷ tracking error. All figures replay today's holdings over the window.</p>`;
  if (d.summary.length) h += `<div class="panel" style="margin:12px 0; background:#0b0b0b;"><b>Biggest tilts vs ${b}</b><ul style="margin:8px 0 0; padding-left:20px;">${d.summary.map(s => `<li>${esc(s)}</li>`).join('')}</ul></div>`;
  else h += `<p class="muted">No factor tilt stands out against ${b}.</p>`;
  const mx = Math.max(0.1, ...d.factors.map(f => Math.abs(f.active)));
  h += `<h2 style="margin:22px 0 10px;">Factor exposures (portfolio vs benchmark)</h2><div style="overflow-x:auto;"><table><tr><th>Factor</th><th>Portfolio</th><th>${b}</th><th>Active</th><th></th><th>t-stat</th><th>Status</th><th>Share of active risk</th></tr>` +
    d.factors.map(f => `<tr><td>${esc(f.label)}</td><td>${num2(f.portfolio)}</td><td>${num2(f.benchmark)}</td><td><b>${f.active >= 0 ? '+' : ''}${num2(f.active)}</b></td><td>${dBar(f.active, mx)}</td><td>${f.t === null ? '—' : num2(f.t)}</td><td>${statusTag(f.status)}</td><td>${pct1(f.risk_share, 1)}</td></tr>`).join('') +
    `<tr><td>Stock-specific (everything the factors don't explain)</td><td colspan="6"></td><td>${pct1(d.risk_split.specific, 1)}</td></tr></table></div>`;
  h += `<p class="muted small">Exposures are regression betas of daily returns on factor-mimicking ETF spreads (market = SPY; size = IWM−IWB; value = IWD−IWF; momentum = MTUM−SPY; quality = QUAL−SPY; low vol = USMV−SPY). "Active" is portfolio minus benchmark: positive = overexposed, negative = underexposed. A tilt is flagged when it's statistically significant (|t| ≥ 2) and at least 0.05 in size. These are ETF proxies, not the academic Fama-French factors, so use them as a guide. Unexplained alpha: ${d.alpha.annual >= 0 ? '+' : ''}${pct1(d.alpha.annual, 1)}/yr (t ${num2(d.alpha.t)}).</p>`;
  h += `<h2 style="margin:22px 0 10px;">What drives your tracking error</h2><div style="overflow-x:auto;"><table><tr><th>Holding</th><th>Weight</th><th>Share of active risk</th><th>Tracking-error points</th></tr>` +
    d.holdings.map(r => `<tr><td>${esc(r.ticker)}</td><td>${pct1(r.weight, 1)}</td><td>${pct1(r.risk_share, 1)}</td><td>${pct1(r.te_points)}</td></tr>`).join('') +
    `<tr><td>${b} (the benchmark you're measured against)</td><td></td><td>${pct1(d.bench_row.risk_share, 1)}</td><td>${pct1(d.bench_row.te_points)}</td></tr></table></div><p class="muted small">A holding's share is how much of your active variance it accounts for. Negative = it offsets other bets or moves with the benchmark. The rows add up to 100%.</p>`;
  const sc = d.sectors;
  h += `<h2 style="margin:22px 0 10px;">Sector exposure vs ${b}</h2>`;
  h += `<div style="overflow-x:auto;"><table><tr><th>Sector</th><th>Portfolio</th>${sc.benchmark_available ? `<th>${b}</th><th>Active</th><th></th><th>Status</th>` : ''}</tr>` +
    sc.rows.map(r => `<tr><td>${esc(r.sector)}</td><td>${pct1(r.portfolio, 1)}</td>${sc.benchmark_available ? `<td>${pct1(r.benchmark, 1)}</td><td>${r.active >= 0 ? '+' : ''}${pct1(r.active, 1)}</td><td>${dBar(r.active, 0.2)}</td><td>${statusTag(r.status)}</td>` : ''}</tr>`).join('') +
    (sc.unclassified > 0.001 ? `<tr><td>Unclassified (ETFs, funds, missing data)</td><td>${pct1(sc.unclassified, 1)}</td>${sc.benchmark_available ? '<td colspan="4"></td>' : ''}</tr>` : '') + `</table></div>`;
  if (sc.ssga_failed) h += `<p class="error small">⚠ Couldn't reach State Street's SPY data, so the benchmark weights below come from Yahoo, which can be inaccurate for SPY. Try again in a moment.</p>`;
  h += `<p class="muted small">${sc.benchmark_available ? 'Benchmark weights: ' + esc(sc.source) + '. Over/underweight is flagged at 5 percentage points or more.' : `Yahoo didn't return a sector breakdown for ${b}, so only your portfolio's sector weights are shown.`}${d.excluded.length ? ' Left out of the risk numbers for under a year of history: ' + esc(d.excluded.join(', ')) + '.' : ''}</p>`;
  $('port-active-out').innerHTML = h;
}
async function portActive() {
  const {sent} = portCollect();
  if (!sent.length) return portErr('Add at least one holding.');
  const bench = $('port-bench').value.trim();
  if (!bench) { $('port-active-out').innerHTML = '<p class="error">⚠ Enter a benchmark ETF in the Benchmark box first (SPY, QQQ, IWM, ...).</p>'; return; }
  $('port-active-out').innerHTML = '<p><span class="spinner"></span> Running tracking error and factor analysis (downloads price history, can take a few seconds)…</p>';
  try { portActiveRender(await portPost('/api/port/active', {holdings: sent, benchmark: bench})); }
  catch (e) { $('port-active-out').innerHTML = `<p class="error">⚠ ${esc(e.message)}</p>`; }
}
$('port-active').addEventListener('click', portActive);

$('port-add').addEventListener('click', portAdd);
$('port-refresh').addEventListener('click', portValue);
$('port-save').addEventListener('click', portSave);
$('port-new').addEventListener('click', portNew);
$('port-delete').addEventListener('click', portDelete);
$('port-close').addEventListener('click', () => { $('port-panel').style.display = 'none'; });
$('port-saved').addEventListener('change', () => { if ($('port-saved').value) portLoad($('port-saved').value); });
portRender();

loadPortfolio();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        try:
            if parsed.path == "/":
                self._html(HTML_PAGE)

            elif parsed.path == "/api/research":
                ticker = qs.get("ticker", [""])[0].strip()
                avkey = qs.get("avkey", [""])[0].strip()
                if not ticker:
                    return self._json({"error": "No ticker provided."}, 400)
                agent = StockResearchAgent(av_api_key=avkey)
                self._json(agent.research(ticker).to_dict())

            elif parsed.path == "/api/portfolio":
                self._json(portfolio_snapshot())

            elif parsed.path == "/api/options":
                ticker = qs.get("ticker", [""])[0].strip().upper()
                expiration = qs.get("expiration", [""])[0].strip()
                if not ticker:
                    return self._json({"error": "No ticker provided."}, 400)
                if expiration:
                    self._json({"puts": get_put_chain(ticker, expiration)})
                else:
                    self._json({"expirations": get_option_expirations(ticker)})

            elif parsed.path == "/api/gaps":
                ticker = qs.get("ticker", [""])[0].strip().upper()
                if not ticker:
                    return self._json({"error": "No ticker provided."}, 400)
                self._json({"gaps": find_gaps(ticker)})

            elif parsed.path == "/api/port/list":
                self._json(port_list())

            elif parsed.path == "/api/port/get":
                self._json(port_get(qs.get("name", [""])[0]))

            elif parsed.path == "/api/topics":
                self._json(get_finance_topics(force=bool(qs.get("refresh"))))

            elif parsed.path == "/api/terminal":
                self._json(run_terminal_command(qs.get("cmd", [""])[0]))

            else:
                self.send_response(404)
                self.end_headers()

        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"

        try:
            if parsed.path == "/api/trade":
                body = json.loads(raw)
                result = execute_trade(
                    body.get("action"), body.get("ticker", ""), body.get("qty", 0), body.get("opt"),
                )
                self._json(result)

            elif parsed.path == "/api/port/value":
                self._json(port_value(json.loads(raw).get("holdings")))

            elif parsed.path == "/api/port/save":
                body = json.loads(raw)
                self._json(port_save(body.get("name"), body.get("holdings"), body.get("benchmark")))

            elif parsed.path == "/api/port/analyze":
                body = json.loads(raw)
                self._json(port_analyze(body.get("holdings"), body.get("benchmark"), body.get("period") or "1Y"))

            elif parsed.path == "/api/port/active":
                body = json.loads(raw)
                self._json(port_active(body.get("holdings"), body.get("benchmark")))

            elif parsed.path == "/api/port/stress":
                body = json.loads(raw)
                self._json(port_stress(body.get("holdings"), body.get("benchmark"), body.get("custom")))

            elif parsed.path == "/api/port/delete":
                self._json(port_delete(json.loads(raw).get("name")))

            elif parsed.path == "/api/reset":
                save_portfolio(_blank_portfolio())
                self._json({"ok": True})

            else:
                self.send_response(404)
                self.end_headers()

        except Exception as e:
            self._json({"error": str(e)}, 500)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stock research terminal — local web app")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\nResearch Terminal running at {url}")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
