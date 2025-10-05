MCP + Streamlit: Stocks Analyzer
==================================

Overview
--------
This repo is a small end‑to‑end demo that combines:
- **Data source**: Yahoo Finance via `yfinance` (no API key required).
- **MCP tools server** (`mcp_server.py`): indicators, event flags, price series, toy backtest, and an optional LLM explain tool.
- **FastAPI bridge** (`api.py`): exposes the MCP tools as HTTP routes that Streamlit can call.
- **Streamlit UI** (`streamlit_app.py`): search a ticker, fetch quote/series/indicators/events, run a toy MA‑crossover backtest, and (optionally) get an LLM explanation.

Quick Start
-----------
1) Create a virtual environment and install deps:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

2) (Optional) Create a .env file
   - `OPENAI_API_KEY` — only needed if you want the **Explain (LLM)** feature.
   - `BACKEND_URL` — default is http://127.0.0.1:8001 for local dev.

3) Start the FastAPI bridge on :8001:
   uvicorn api:app --host 0.0.0.0 --port 8001 --reload

4) Start Streamlit on :8501:
   streamlit run streamlit_app.py

(There is also a `run.txt` with similar steps.)

Project Structure
-----------------
- datasource.py — Yahoo Finance search, latest quote, and daily OHLCV series normalization.
- mcp_server.py — MCP tools that wrap the datasource and compute indicators (SMA/EMA/RSI), detect events (gaps, volatility spikes, 52‑week extremes), a toy MA‑crossover backtest, and an optional LLM explanation with guardrails.
- api.py — FastAPI app with routes that call the MCP tools and normalize JSON outputs.
- streamlit_app.py — Streamlit UI that searches tickers, calls the API routes, normalizes series payloads, plots the close price with Matplotlib, and shows JSON panels.
- debug_mcp_stocks.py — Local debugging script to test the datasource, MCP tools, API, and Streamlit series normalization path end‑to‑end.
- requirements.txt — Python dependencies.
- run.txt — one‑page run notes (venv, install, start backend, start UI).

Environment Variables
---------------------
- `BACKEND_URL` — Streamlit reads this to locate the FastAPI server (defaults to http://127.0.0.1:8001).
- `OPENAI_API_KEY` — if set, the Explain endpoint/tool will produce an LLM‑generated summary; otherwise a safe, deterministic fallback text is used.

Running the Services
--------------------
Backend (FastAPI):
   uvicorn api:app --host 0.0.0.0 --port 8001 --reload

Frontend (Streamlit):
   streamlit run streamlit_app.py

Then open http://localhost:8501 in your browser. The top bar shows “Backend: Up/Down”. If it's down, (re)start the FastAPI service.

API Endpoints
-------------
GET  /health                → {"status":"ok"}
POST /search                → {q}
POST /quote                 → {symbol}
POST /series                → {symbol, interval="daily", lookback=int}
POST /indicators            → {symbol, window_sma, window_ema, window_rsi}
POST /events                → {symbol}
POST /backtest              → {symbol, fast, slow}   # requires fast < slow
POST /explain               → {symbol, language, tone, risk_profile, horizon_days, bullets}
POST /bundle                → {symbol, lookback, window_sma, window_ema, window_rsi}  # returns series+indicators+events

Notes on Data & Normalization
-----------------------------
- **Search** uses Yahoo’s public search endpoint and returns lightweight objects with `symbol` and `name` fields.
- **Quote** tries `yfinance.Ticker(...).fast_info` first and falls back to recent daily history if needed.
- **Series** uses `yfinance.download` and normalizes to columns: `date, open, high, low, close, volume` (daily, recent lookback only). The Streamlit app is defensive: it accepts a variety of shapes and column names, coerces types, sorts by date, then plots Close vs Date.
- **Indicators/Events/Backtest** operate on the normalized daily close series. 52‑week flags use a 252‑trading‑day rolling window.
- **Explain** collects local indicator/event context and, if `OPENAI_API_KEY` is set, asks an LLM to produce a short structured summary (otherwise it returns a template‑based fallback).

MCP Tools (from mcp_server)
---------------------------
- `search_symbols(query)` → JSON array
- `latest_quote(symbol)` → JSON object
- `price_series(symbol, interval="daily", lookback=180)` → JSON array of OHLCV rows (ISO dates)
- `indicators(symbol, window_sma=20, window_ema=50, window_rsi=14)` → JSON object
- `detect_events(symbol)` → JSON object (gap up/down, volatility spike, 52‑week high/low on the last bar)
- `backtest_ma_cross(symbol, fast=10, slow=20)` → JSON object with CAGR, win_rate (toy model)
- `explain(...)` → JSON object with {"text", "rationale"[], "disclaimers"}; robust fallback if LLM is unavailable

Streamlit Usage
---------------
1) Type a company or ticker in **Search** (e.g., “NVIDIA” or “NVDA”) and click **Search**.
2) Pick a symbol from the dropdown. The app stores your selection and enables buttons for:
   - **Get Quote**: latest price/change/volume snapshot
   - **Series**: daily OHLCV and a Matplotlib plot
   - **Indicators**: SMA/EMA/RSI snapshot
   - **Events**: gap/volatility/52‑week flags (last bar)
   - **Bundle**: one call that returns series + indicators + events
   - **Backtest MA Crossover**: toy long‑only strategy (fast/slow windows)
   - **Explain (LLM)**: optional short summary if `OPENAI_API_KEY` is set

Debugging
---------
- Use `debug_mcp_stocks.py` to test each layer independently:
  - `test_datasource()` → exercises `datasource.py`
  - `test_mcp_tools()` → calls MCP tools directly
  - `test_api_endpoint()` → hits FastAPI at `http://127.0.0.1:8001`
  - `test_streamlit_normalization()` → sanity‑checks the Streamlit series normalization against several payload shapes

Troubleshooting
---------------
- **Backend: Down** in Streamlit → start FastAPI: `uvicorn api:app --port 8001 --reload`
- **Empty/short series** → try a larger lookback or a different symbol; ensure internet access.
- **Backtest bad params** → ensure `fast < slow`.
- **Explain returns fallback** → set `OPENAI_API_KEY` if you want LLM output; otherwise fallback is expected.
- **CORS issues** in non‑local setups → tighten CORS in `api.py` for production.

License
-------
For demo/educational purposes only. No investment advice.
