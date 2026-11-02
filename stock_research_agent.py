"""
Autonomous Stock Research Agent
================================

A self-contained pipeline that researches a stock the way a junior analyst
would: pull fundamentals, pull price history, compute technical signals,
pull recent news, then synthesize everything into a single report.

Design pattern (the "agentic" part):
    fetch -> analyze -> synthesize -> (optional) act
Each stage is a separate method so you can swap implementations (e.g. plug
in a different data provider, or hand the final synthesis step to an LLM)
without touching the rest of the pipeline.

Requirements:
    pip install yfinance pandas numpy

Optional (for LLM-written narrative summaries instead of the built-in
rule-based one — see `LLMSummarizer` at the bottom):
    pip install anthropic

Usage:
    python stock_research_agent.py AAPL MSFT NVDA
    python stock_research_agent.py AAPL --json report_aapl.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class Fundamentals:
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dividend_yield: Optional[float] = None
    analyst_target_mean: Optional[float] = None
    analyst_recommendation: Optional[str] = None


@dataclass
class Technicals:
    last_price: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    volatility_annualized: Optional[float] = None
    high_52wk: Optional[float] = None
    low_52wk: Optional[float] = None
    pct_off_52wk_high: Optional[float] = None
    trend: Optional[str] = None  # "uptrend" / "downtrend" / "sideways"


@dataclass
class NewsItem:
    title: str
    publisher: str
    link: str
    published: Optional[str] = None


@dataclass
class CrossCheck:
    yahoo_last_price: Optional[float] = None
    alpha_vantage_last_price: Optional[float] = None
    diff_pct: Optional[float] = None
    note: str = ""


@dataclass
class StockReport:
    ticker: str
    generated_at: str
    fundamentals: Fundamentals
    technicals: Technicals
    cross_check: Optional[CrossCheck] = None
    news: list[NewsItem] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_markdown(self) -> str:
        f, t = self.fundamentals, self.technicals
        lines = [
            f"# Research Report: {f.name or self.ticker} ({self.ticker})",
            f"_Generated {self.generated_at}_",
            "",
            f"**Sector / Industry:** {f.sector or '—'} / {f.industry or '—'}",
            "",
            "## Price & Technicals",
            f"- Last price: {_fmt(t.last_price)}",
            f"- 52-week range: {_fmt(t.low_52wk)} – {_fmt(t.high_52wk)} "
            f"({_fmt_pct(t.pct_off_52wk_high)} off high)",
            f"- 50/200-day SMA: {_fmt(t.sma_50)} / {_fmt(t.sma_200)}",
            f"- RSI (14d): {_fmt(t.rsi_14)}",
            f"- Annualized volatility: {_fmt_pct(t.volatility_annualized)}",
            f"- Trend: **{t.trend or '—'}**",
            "",
            "## Fundamentals",
            f"- Market cap: {_fmt_big(f.market_cap)}",
            f"- P/E (trailing / forward): {_fmt(f.pe_ratio)} / {_fmt(f.forward_pe)}",
            f"- PEG ratio: {_fmt(f.peg_ratio)}",
            f"- Price/Book: {_fmt(f.price_to_book)}",
            f"- Profit margin: {_fmt_pct(f.profit_margin)}",
            f"- Revenue growth (YoY): {_fmt_pct(f.revenue_growth)}",
            f"- Earnings growth (YoY): {_fmt_pct(f.earnings_growth)}",
            f"- Debt/Equity: {_fmt(f.debt_to_equity)}",
            f"- Dividend yield: {_fmt_pct(f.dividend_yield)}",
            f"- Analyst mean target: {_fmt(f.analyst_target_mean)} "
            f"(rec: {f.analyst_recommendation or '—'})",
            "",
            "## Cross-Source Price Check (Yahoo Finance vs. Alpha Vantage)",
        ]
        if self.cross_check:
            c = self.cross_check
            lines += [
                f"- Yahoo Finance last price: {_fmt(c.yahoo_last_price)}",
                f"- Alpha Vantage last price: {_fmt(c.alpha_vantage_last_price)}",
                f"- Difference: {_fmt_pct(c.diff_pct)}" if c.diff_pct is not None else "- Difference: —",
                f"- {c.note}" if c.note else "",
            ]
        else:
            lines += ["- Not available"]
        lines += ["", "## Signals"]
        lines += [f"- {s}" for s in self.signals] or ["- None flagged"]
        lines += ["", "## Risks"]
        lines += [f"- {r}" for r in self.risks] or ["- None flagged"]
        lines += ["", "## Recent News"]
        lines += [f"- [{n.title}]({n.link}) — {n.publisher}" for n in self.news] or ["- None found"]
        lines += ["", "## Summary", self.summary]
        return "\n".join(lines)


def _fmt(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.2f}"


def _fmt_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:,.1f}%"


def _fmt_big(x: Optional[float]) -> str:
    if x is None:
        return "—"
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if abs(x) >= div:
            return f"${x / div:,.2f}{unit}"
    return f"${x:,.0f}"


# --------------------------------------------------------------------------- #
# Second data source, for cross-checking Yahoo Finance
# --------------------------------------------------------------------------- #

class AlphaVantageDataSource:
    """
    Real-time/delayed quote data from Alpha Vantage, used as an independent
    cross-check against Yahoo Finance. Requires a free API key from
    alphavantage.co/support/#api-key.

    Note: the free tier is rate-limited (roughly 25 requests/day, 5/minute
    as of this writing) — fine for checking a handful of tickers, but you
    will hit the limit if you run this against many tickers in one day.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    @staticmethod
    def fetch_last_close(ticker: str, api_key: str) -> Optional[float]:
        if not api_key:
            return None
        try:
            resp = requests.get(
                AlphaVantageDataSource.BASE_URL,
                params={"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            quote = data.get("Global Quote", {})
            price_str = quote.get("05. price")
            return float(price_str) if price_str else None
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class StockResearchAgent:
    """
    Orchestrates the research pipeline for a single ticker.
    Each stage returns plain data so the pipeline is easy to test, log,
    cache, or swap out piece by piece.
    """

    def __init__(self, lookback_period: str = "1y", av_api_key: str = ""):
        self.lookback_period = lookback_period
        self.av_api_key = av_api_key

    # ---- Stage 1: fetch ---------------------------------------------------

    def fetch_ticker_object(self, ticker: str) -> yf.Ticker:
        return yf.Ticker(ticker)

    def fetch_history(self, tk: yf.Ticker) -> pd.DataFrame:
        hist = tk.history(period=self.lookback_period)
        if hist.empty:
            raise ValueError("No price history returned — check the ticker symbol.")
        return hist

    def fetch_info(self, tk: yf.Ticker) -> dict:
        try:
            return tk.get_info()
        except Exception:
            return {}

    def fetch_news(self, tk: yf.Ticker, limit: int = 5) -> list[NewsItem]:
        items: list[NewsItem] = []
        try:
            raw = tk.news or []
        except Exception:
            raw = []
        for entry in raw[:limit]:
            content = entry.get("content", entry)  # yfinance schema varies by version
            title = content.get("title") or entry.get("title", "Untitled")
            publisher = (
                content.get("provider", {}).get("displayName")
                if isinstance(content.get("provider"), dict)
                else entry.get("publisher", "Unknown")
            )
            link = (
                content.get("canonicalUrl", {}).get("url")
                if isinstance(content.get("canonicalUrl"), dict)
                else entry.get("link", "")
            )
            published = content.get("pubDate") or None
            items.append(NewsItem(title=title, publisher=publisher or "Unknown",
                                   link=link or "", published=published))
        return items

    # ---- Stage 2: analyze --------------------------------------------------

    def compute_technicals(self, hist: pd.DataFrame) -> Technicals:
        close = hist["Close"].dropna()  # drop any incomplete/placeholder rows Yahoo sometimes appends
        if close.empty:
            raise ValueError("No valid closing prices found in the price history.")
        last_price = float(close.iloc[-1])

        sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        rsi_14 = self._rsi(close, period=14)

        daily_returns = close.pct_change().dropna()
        volatility_annualized = (
            float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 1 else None
        )

        high_52wk = float(close.max())
        low_52wk = float(close.min())
        pct_off_high = (last_price - high_52wk) / high_52wk if high_52wk else None

        trend = self._classify_trend(last_price, sma_50, sma_200)

        return Technicals(
            last_price=last_price,
            sma_50=sma_50,
            sma_200=sma_200,
            rsi_14=rsi_14,
            volatility_annualized=volatility_annualized,
            high_52wk=high_52wk,
            low_52wk=low_52wk,
            pct_off_52wk_high=pct_off_high,
            trend=trend,
        )

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> Optional[float]:
        if len(close) < period + 1:
            return None
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return None if pd.isna(val) else float(val)

    @staticmethod
    def _classify_trend(price: float, sma_50: Optional[float], sma_200: Optional[float]) -> str:
        if sma_50 is None or sma_200 is None:
            return "insufficient data"
        if price > sma_50 > sma_200:
            return "uptrend"
        if price < sma_50 < sma_200:
            return "downtrend"
        return "sideways / mixed"

    def compute_fundamentals(self, ticker: str, info: dict) -> Fundamentals:
        g = info.get  # shorthand
        return Fundamentals(
            ticker=ticker,
            name=g("longName") or g("shortName"),
            sector=g("sector"),
            industry=g("industry"),
            market_cap=g("marketCap"),
            pe_ratio=g("trailingPE"),
            forward_pe=g("forwardPE"),
            peg_ratio=g("pegRatio") or g("trailingPegRatio"),
            price_to_book=g("priceToBook"),
            profit_margin=g("profitMargins"),
            revenue_growth=g("revenueGrowth"),
            earnings_growth=g("earningsGrowth"),
            debt_to_equity=g("debtToEquity"),
            free_cash_flow=g("freeCashflow"),
            dividend_yield=(g("dividendYield") / 100 if g("dividendYield") is not None else None),
            analyst_target_mean=g("targetMeanPrice"),
            analyst_recommendation=g("recommendationKey"),
        )

    # ---- Stage 3: synthesize -----------------------------------------------

    def derive_signals_and_risks(
        self, f: Fundamentals, t: Technicals
    ) -> tuple[list[str], list[str]]:
        signals, risks = [], []

        if t.trend == "uptrend":
            signals.append("Price trading above both 50- and 200-day moving averages (bullish trend structure).")
        elif t.trend == "downtrend":
            risks.append("Price trading below both 50- and 200-day moving averages (bearish trend structure).")

        if t.rsi_14 is not None:
            if t.rsi_14 >= 70:
                risks.append(f"RSI at {t.rsi_14:.0f} — potentially overbought.")
            elif t.rsi_14 <= 30:
                signals.append(f"RSI at {t.rsi_14:.0f} — potentially oversold, could indicate a bounce setup.")

        if t.pct_off_52wk_high is not None and t.pct_off_52wk_high <= -0.20:
            risks.append(f"Trading {abs(t.pct_off_52wk_high) * 100:.0f}% below its 52-week high.")

        if f.pe_ratio is not None and f.pe_ratio > 40:
            risks.append(f"Trailing P/E of {f.pe_ratio:.1f} is elevated — priced for high growth.")

        if f.peg_ratio is not None and f.peg_ratio < 1:
            signals.append(f"PEG ratio of {f.peg_ratio:.2f} suggests growth may be underpriced relative to earnings growth.")

        if f.revenue_growth is not None and f.revenue_growth < 0:
            risks.append(f"Revenue growth is negative ({f.revenue_growth * 100:.1f}%).")

        if f.debt_to_equity is not None and f.debt_to_equity > 150:
            risks.append(f"Debt/Equity of {f.debt_to_equity:.0f} is high — elevated leverage.")

        if t.volatility_annualized is not None and t.volatility_annualized > 0.5:
            risks.append(f"Annualized volatility of {t.volatility_annualized * 100:.0f}% is well above a typical large-cap.")

        return signals, risks

    def cross_check_price(self, ticker: str, yahoo_last_price: Optional[float]) -> CrossCheck:
        if not self.av_api_key:
            return CrossCheck(
                yahoo_last_price=yahoo_last_price,
                alpha_vantage_last_price=None,
                diff_pct=None,
                note="No Alpha Vantage API key provided — skipped cross-check.",
            )

        av_price = AlphaVantageDataSource.fetch_last_close(ticker, self.av_api_key)

        if yahoo_last_price is None or av_price is None:
            return CrossCheck(
                yahoo_last_price=yahoo_last_price,
                alpha_vantage_last_price=av_price,
                diff_pct=None,
                note="Could not compare — one or both sources returned no data "
                     "(check your API key, or you may have hit Alpha Vantage's daily rate limit).",
            )

        diff_pct = (yahoo_last_price - av_price) / av_price
        if abs(diff_pct) > 0.02:
            note = (
                f"Sources disagree by more than 2% — likely a stale quote on one "
                f"side (markets closed, delayed feed, etc.), not necessarily an error."
            )
        else:
            note = "Sources agree closely."

        return CrossCheck(
            yahoo_last_price=yahoo_last_price,
            alpha_vantage_last_price=av_price,
            diff_pct=diff_pct,
            note=note,
        )

    def write_summary(self, f: Fundamentals, t: Technicals, signals: list[str], risks: list[str]) -> str:
        """Simple rule-based narrative. Swap in `LLMSummarizer` below for
        a model-written version instead."""
        name = f.name or f.ticker
        bias = "constructive" if len(signals) >= len(risks) else "cautious"
        parts = [
            f"{name} is in a {t.trend or 'unclear'} technical posture, trading at "
            f"{_fmt(t.last_price)} versus a 52-week range of {_fmt(t.low_52wk)}–{_fmt(t.high_52wk)}."
        ]
        if f.pe_ratio:
            parts.append(f"It trades at a trailing P/E of {f.pe_ratio:.1f}.")
        if signals:
            parts.append("Notable positives: " + "; ".join(signals) + ".")
        if risks:
            parts.append("Key risks to watch: " + "; ".join(risks) + ".")
        parts.append(f"Overall read: {bias}, pending your own diligence and current news.")
        return " ".join(parts)

    # ---- Orchestration ------------------------------------------------------

    def research(self, ticker: str) -> StockReport:
        ticker = ticker.upper().strip()
        tk = self.fetch_ticker_object(ticker)

        hist = self.fetch_history(tk)
        info = self.fetch_info(tk)
        news = self.fetch_news(tk)

        technicals = self.compute_technicals(hist)
        fundamentals = self.compute_fundamentals(ticker, info)
        cross_check = self.cross_check_price(ticker, technicals.last_price)
        signals, risks = self.derive_signals_and_risks(fundamentals, technicals)
        summary = self.write_summary(fundamentals, technicals, signals, risks)

        return StockReport(
            ticker=ticker,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            fundamentals=fundamentals,
            technicals=technicals,
            cross_check=cross_check,
            news=news,
            signals=signals,
            risks=risks,
            summary=summary,
        )


# --------------------------------------------------------------------------- #
# Optional: hand synthesis off to an LLM instead of the rule-based writer.
# Uncomment / wire in if you have an Anthropic API key available.
# --------------------------------------------------------------------------- #

class LLMSummarizer:
    """
    Drop-in replacement for `StockResearchAgent.write_summary`.
    Requires `pip install anthropic` and an ANTHROPIC_API_KEY env var.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # imported lazily so it's not a hard dependency
        self.client = anthropic.Anthropic()
        self.model = model

    def summarize(self, report: StockReport) -> str:
        prompt = (
            "You are a financial research analyst. Write a concise, balanced "
            "3-5 sentence summary of this stock based on the structured data "
            "below. Do not give investment advice or price targets of your own; "
            "just synthesize what the data shows.\n\n"
            f"{json.dumps(report.to_dict(), indent=2, default=str)}"
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Autonomous stock research agent")
    parser.add_argument("tickers", nargs="+", help="One or more ticker symbols, e.g. AAPL MSFT")
    parser.add_argument("--period", default="1y", help="History lookback (default: 1y)")
    parser.add_argument("--av-key", default="", metavar="KEY",
                         help="Your free Alpha Vantage API key (get one at alphavantage.co/support/#api-key). "
                              "Used to cross-check prices against Yahoo Finance.")
    parser.add_argument("--json", metavar="PATH", help="Also write JSON output to this path")
    parser.add_argument("--llm-summary", action="store_true",
                         help="Use LLMSummarizer instead of the rule-based summary")
    args = parser.parse_args()

    agent = StockResearchAgent(lookback_period=args.period, av_api_key=args.av_key)
    all_reports = []

    for ticker in args.tickers:
        try:
            report = agent.research(ticker)
        except Exception as e:
            print(f"⚠️  Failed to research {ticker}: {e}", file=sys.stderr)
            continue

        if args.llm_summary:
            try:
                report.summary = LLMSummarizer().summarize(report)
            except Exception as e:
                print(f"⚠️  LLM summary failed for {ticker}, keeping rule-based summary ({e})",
                      file=sys.stderr)

        print(report.to_markdown())
        print("\n" + "=" * 80 + "\n")
        all_reports.append(report.to_dict())

    if args.json and all_reports:
        with open(args.json, "w") as f:
            json.dump(all_reports, f, indent=2, default=str)
        print(f"Saved JSON to {args.json}")


if __name__ == "__main__":
    main()
