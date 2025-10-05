# mcp_server.py
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP

# yfinance-backed datasource functions (your updated datasource.py)
from datasource import (
    search_symbols as ds_search,
    latest_quote as ds_quote,
    price_series as ds_series,
)

mcp = FastMCP("stocks-analyzer")

# ---------- indicators & helpers ----------
def calc_sma(s: pd.Series, w: int = 20) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 2)).mean()

def calc_ema(s: pd.Series, w: int = 20) -> pd.Series:
    return s.ewm(span=w, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_cagr(price_curve: pd.Series, periods_per_year: int = 252):
    if len(price_curve) < 2:
        return float("nan")
    ret = float(price_curve.iloc[-1]) / float(price_curve.iloc[0])
    yrs = len(price_curve) / periods_per_year
    return float(ret ** (1 / yrs) - 1) if yrs > 0 else float("nan")

def simple_ma_crossover(close: pd.Series, fast: int = 10, slow: int = 20) -> Dict[str, Any]:
    f = calc_sma(close, fast)
    s = calc_sma(close, slow)
    sig = (f > s).astype(int)  # 1=long, 0=flat
    daily_ret = close.pct_change().fillna(0)
    strat_ret = daily_ret * sig.shift(1).fillna(0)
    equity = (1 + strat_ret).cumprod()
    return {
        "fast": fast,
        "slow": slow,
        "cagr": calc_cagr(equity),
        "win_rate": float((strat_ret > 0).mean()),
    }

def flag_gaps(df: pd.DataFrame, threshold: float = 0.03) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    gap = (df["open"] - prev_close) / prev_close
    df = df.copy()
    df["gap_up"] = gap >= threshold
    df["gap_down"] = gap <= -threshold
    return df

def flag_volatility(df: pd.DataFrame, window: int = 20, mult: float = 2.0) -> pd.DataFrame:
    ret = df["close"].pct_change()
    vol = ret.rolling(window, min_periods=5).std()
    df = df.copy()
    df["vol_spike"] = ret.abs() > (mult * vol)
    return df

def flag_52w_extremes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    roll_max = df["close"].rolling(252, min_periods=30).max()
    roll_min = df["close"].rolling(252, min_periods=30).min()
    df["is_52w_high"] = df["close"] >= roll_max
    df["is_52w_low"] = df["close"] <= roll_min
    return df

def _coerce_close(df: pd.DataFrame) -> pd.Series:
    """Return a numeric close series or empty series."""
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df["close"], errors="coerce").dropna()

# ---------- MCP tools ----------
@mcp.tool()
def search_symbols(query: str) -> str:
    """Symbol lookup by company name/ticker. Returns a JSON array."""
    try:
        return json.dumps(ds_search(query), ensure_ascii=False)
    except Exception as e:
        return json.dumps([{"error": "search_failed", "message": str(e)}])

@mcp.tool()
def latest_quote(symbol: str) -> str:
    """Latest price, change %, volume. Returns a JSON object."""
    try:
        return json.dumps(ds_quote(symbol), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"symbol": symbol, "error": "quote_failed", "message": str(e)})
# mcp_server.py -> price_series tool
@mcp.tool()
def price_series(symbol: str, interval: str = "daily", lookback: int = 180) -> str:
    """OHLCV series as a JSON array (date ISO)."""
    try:
        df = ds_series(symbol, interval, lookback)
        # Guarantee expected columns even if empty
        for col in ["date", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = pd.Series(dtype="float64" if col != "date" else "datetime64[ns]")
        return df.to_json(orient="records", date_format="iso")
    except Exception as e:
        # <— include message so the UI shows it
        return json.dumps({"symbol": symbol, "error": "series_failed", "message": str(e)})

@mcp.tool()
def indicators(
    symbol: str,
    window_sma: int = 20,
    window_ema: int = 50,
    window_rsi: int = 14,
) -> str:
    """SMA/EMA/RSI and last snapshot. Returns a JSON object; never raises."""
    try:
        df = ds_series(symbol, "daily", 300)
        close = _coerce_close(df)
        if close.empty:
            return json.dumps({"symbol": symbol, "error": "no_data"})
        sma = calc_sma(close, window_sma).iloc[-1]
        ema = calc_ema(close, window_ema).iloc[-1]
        rsi = calc_rsi(close, window_rsi).iloc[-1]
        out = {
            "symbol": symbol,
            "last_close": float(close.iloc[-1]),
            "sma": float(sma) if pd.notna(sma) else None,
            "ema": float(ema) if pd.notna(ema) else None,
            "rsi": float(rsi) if pd.notna(rsi) else None,
        }
        return json.dumps(out)
    except Exception as e:
        return json.dumps({"symbol": symbol, "error": "indicators_failed", "message": str(e)})

@mcp.tool()
def detect_events(symbol: str) -> str:
    """Gap up/down, volatility spikes, 52w extremes on the last bar. Returns a JSON object; never raises."""
    try:
        df = ds_series(symbol, "daily", 400)
        if df is None or df.empty:
            return json.dumps({"symbol": symbol, "error": "no_data"})
        # ensure numeric
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
        if df.empty:
            return json.dumps({"symbol": symbol, "error": "no_data"})

        df = flag_gaps(df)
        df = flag_volatility(df)
        df = flag_52w_extremes(df)
        last_row = df.iloc[-1]
        last = {
            "symbol": symbol,
            "date": str(pd.to_datetime(last_row["date"]).date()) if "date" in df.columns else None,
            "gap_up": bool(last_row.get("gap_up", False)),
            "gap_down": bool(last_row.get("gap_down", False)),
            "vol_spike": bool(last_row.get("vol_spike", False)),
            "is_52w_high": bool(last_row.get("is_52w_high", False)),
            "is_52w_low": bool(last_row.get("is_52w_low", False)),
        }
        return json.dumps(last)
    except Exception as e:
        return json.dumps({"symbol": symbol, "error": "events_failed", "message": str(e)})

@mcp.tool()
def backtest_ma_cross(symbol: str, fast: int = 10, slow: int = 20) -> str:
    """Toy MA crossover (long-only). Returns CAGR and win rate; never raises."""
    try:
        if fast >= slow:
            return json.dumps({"symbol": symbol, "error": "bad_params", "message": "fast must be < slow"})
        df = ds_series(symbol, "daily", 600)
        close = _coerce_close(df)
        if close.empty:
            return json.dumps({"symbol": symbol, "error": "no_data"})
        stats = simple_ma_crossover(close, fast, slow)
        stats["symbol"] = symbol
        # Convert NaNs to None for JSON
        for k in ("cagr", "win_rate"):
            v = stats.get(k)
            stats[k] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        return json.dumps(stats)
    except Exception as e:
        return json.dumps({"symbol": symbol, "error": "backtest_failed", "message": str(e)})

@mcp.tool()
def explain(
    symbol: str,
    language: str = "en",
    tone: str = "neutral",
    risk_profile: str = "balanced",
    horizon_days: int = 30,
    bullets: bool = True,
) -> str:
    """
    LLM explanation of the current technical snapshot with guardrails.
    Returns a JSON object: {"text": "...", "rationale": [...], "disclaimers": "..."}.
    Never raises. Falls back to a templated narrative if no OPENAI_API_KEY.
    """
    import json as _json

    def _safe_json(s):
        try:
            return _json.loads(s) if isinstance(s, str) else (s or {})
        except Exception:
            return {}

    # 1) Gather fresh local context (no external calls here)
    ind = _safe_json(indicators(symbol))
    evt = _safe_json(detect_events(symbol))

    # 2) Build a good fallback in case LLM is unavailable or fails
    def _fallback():
        last = ind.get("last_close")
        sma = ind.get("sma"); ema = ind.get("ema"); rsi = ind.get("rsi")
        gap_up = evt.get("gap_up"); gap_down = evt.get("gap_down")
        vspike = evt.get("vol_spike")
        hi52 = evt.get("is_52w_high"); lo52 = evt.get("is_52w_low")

        # quick interpretive templates
        dir_sma = "above" if (last is not None and sma is not None and last >= sma) else "below"
        dir_ema = "above" if (last is not None and ema is not None and last >= ema) else "below"
        rsi_note = (
            "neutral momentum" if rsi is None else
            ("overbought-ish" if rsi >= 70 else "oversold-ish" if rsi <= 30 else "balanced momentum")
        )
        events = []
        if gap_up: events.append("gap up")
        if gap_down: events.append("gap down")
        if vspike: events.append("volatility spike")
        if hi52: events.append("near 52-week high")
        if lo52: events.append("near 52-week low")
        ev_text = (", ".join(events) if events else "no unusual session events detected")

        lang_he = (language or "en").lower().startswith("he")
        if lang_he:
            txt = (
                f"{symbol}: המחיר האחרון {last}. יחסית לממוצעים נעים: מעל/מתחת—"
                f"SMA: {dir_sma}, EMA: {dir_ema}. RSI מצביע על {rsi_note}. "
                f"אירועים: {ev_text}. אין זו המלצה להשקעה."
            )
        else:
            txt = (
                f"{symbol}: last close {last}. Versus moving averages: {dir_sma} the SMA, {dir_ema} the EMA. "
                f"RSI suggests {rsi_note}. Session events: {ev_text}. This is not investment advice."
            )

        out = {
            "text": txt,
            "rationale": [
                f"Snapshot uses last_close={last}, SMA={sma}, EMA={ema}, RSI={rsi}.",
                f"Events flags: gap_up={gap_up}, gap_down={gap_down}, vol_spike={vspike}, 52wHigh={hi52}, 52wLow={lo52}.",
                f"Horizon considered: ~{horizon_days} days; risk profile: {risk_profile}."
            ],
            "disclaimers": "Educational summary of technical signals. Not financial advice.",
        }
        return _json.dumps(out, ensure_ascii=False)

    if not os.getenv("OPENAI_API_KEY"):
        return _fallback()

    # 3) LLM path with strict guardrails and JSON schema
    try:
        from openai import OpenAI
        client = OpenAI()

        # System message
        system_msg = (
            "You are an impartial market analyst. Summarize technical signals clearly, "
            "avoid predictions and avoid financial advice. Use short, concrete language. "
            "If inputs are missing, acknowledge uncertainty. Output in the requested language."
        )

        # User prompt with all context (indicators + events + knobs)
        prompt = {
            "symbol": symbol,
            "language": language,
            "tone": tone,
            "risk_profile": risk_profile,
            "horizon_days": horizon_days,
            "bullets": bool(bullets),
            "indicators": ind,
            "events": evt,
            "instructions": [
                "Keep it under ~120 words if bullets=False, or 3-5 bullets if bullets=True.",
                "No investment advice. No price targets.",
                "Explain what each signal implies in plain language.",
                "If RSI or MAs are missing, say so briefly.",
                "Mention 52-week context if flagged."
            ],
        }

        # JSON schema for structured output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "tech_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "rationale": {"type": "array", "items": {"type": "string"}},
                        "disclaimers": {"type": "string"}
                    },
                    "required": ["text", "disclaimers"]
                }
            },
        }

        # Compose messages + call
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Generate a technical summary:\n{json.dumps(prompt, ensure_ascii=False)}"},
        ]

        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0.2,
            response_format=response_format,
        )
        content = (r.choices[0].message.content or "").strip()
        # If model didn't honor JSON schema, wrap as text
        try:
            _ = json.loads(content)
            return content
        except Exception:
            return json.dumps({"text": content or "", "disclaimers": "Not investment advice."}, ensure_ascii=False)
    except Exception:
        return _fallback()