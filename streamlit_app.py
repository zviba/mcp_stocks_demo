# streamlit_app.py
import json, os, requests, pandas as pd, streamlit as st

# Configuration via environment variables

# --- Config helpers: use environment variables ---
def get_cfg(key: str, default: str | None = None) -> str | None:
    # Use environment variables instead of st.secrets
    return os.getenv(key, default)

# --- Backend URL (default: local uvicorn) ---
BACKEND = get_cfg("BACKEND_URL", "http://127.0.0.1:8001")

# === Auto-start FastAPI backend for Streamlit Cloud ===
import threading, time

def _is_backend_up() -> bool:
    try:
        r = requests.get(f"{BACKEND}/health", timeout=2)
        return r.ok
    except Exception:
        return False

def _start_backend_in_thread():
    # Run uvicorn in a daemon thread so Streamlit remains responsive
    def runner():
        import uvicorn
        from api import app  # ensure api.py is in the same repo
        uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
    t = threading.Thread(target=runner, daemon=True)
    t.start()

# Only auto-start if pointing to local loopback (Cloud: same container)
if BACKEND.startswith("http://127.0.0.1"):
    if not _is_backend_up():
        _start_backend_in_thread()
        # brief warmup loop
        for _ in range(40):  # ~12s total
            if _is_backend_up():
                break
            time.sleep(0.3)

st.set_page_config(page_title="MCP Stocks Analyzer", page_icon="📈")
st.title("📈 MCP + Streamlit: Stocks Analyzer")
st.caption("Live data source: Yahoo Finance (yfinance) • Indicators & events: custom functions • Tools via MCP bridge")

# OpenAI API Key input
st.subheader("Configuration")
openai_api_key = st.text_input(
    "OpenAI API Key", 
    value=st.session_state.get("openai_api_key", ""),
    type="password",
    help="Enter your OpenAI API key to enable LLM explanations",
    key="openai_api_key_input"
)
if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key

# --- small helpers ---
def backend_alive() -> bool:
    try:
        r = requests.get(f"{BACKEND}/health", timeout=3)
        return r.ok
    except Exception:
        return False

def post_json(path: str, payload: dict, timeout: int = 30):
    url = f"{BACKEND}{path}"
    try:
        r = requests.post(url, json=payload, timeout=timeout, headers={"accept": "application/json"})
    except requests.RequestException as e:
        return {"_error": "request_failed", "_message": str(e)}
    ctype = (r.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            return r.json()
        except Exception as e:
            return {"_error": "invalid_json", "_message": str(e), "_body": r.text[:500], "_status": r.status_code}
    return {"_error": "non_json_response", "_body": r.text[:500], "_status": r.status_code}

# --- status bar ---
st.write("Backend:", "✅ Up" if backend_alive() else "❌ Down")

# --- Search helpers ---
def _normalize_search_payload(payload):
    """
    Return (rows:list[{'symbol','name'}], err_msg:str|None, raw:any)
    Accepts: list | dict | anything
    Understands shapes: {"ok":True,"data":[...]}, {"error":...}, plain list, etc.
    """
    # top-level error?
    if isinstance(payload, dict):
        if payload.get("ok") is False:
            return [], payload.get("message") or payload.get("error") or "Search failed.", payload
        if payload.get("_error") or payload.get("error"):
            msg = payload.get("message") or payload.get("_message") or payload.get("_body") or payload.get("error")
            return [], (msg or "Search error."), payload
        # unwrap common containers
        for key in ("data", "result", "items", "records", "quotes"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break

    rows_in = payload if isinstance(payload, list) else []
    out, seen = [], set()
    err = None

    for r in rows_in:
        if not isinstance(r, dict):
            continue
        if r.get("error"):
            err = r.get("message") or r["error"]
            continue
        sym = (r.get("symbol") or r.get("ticker") or "").strip()
        if not sym or sym in seen:
            continue
        name = (r.get("name") or r.get("description") or r.get("shortname") or r.get("longname") or sym).strip()
        out.append({"symbol": sym, "name": name})
        seen.add(sym)

    return out, err, payload

# --- Search UI ---
st.divider()
st.subheader("Search")

q = st.text_input("Search company or ticker", value=st.session_state.get("last_query", "NVIDIA"))
colS, colC = st.columns([1, 1])
with colS:
    if st.button("Search", type="primary"):
        st.session_state["last_query"] = q
        if not backend_alive():
            st.error("Backend is not reachable. If running on Streamlit Cloud, the app should auto-start a local backend. If this persists, check logs and that FastAPI imports.")
            st.session_state["results"] = []
            st.session_state["results_err"] = None
            st.session_state["raw_results"] = None
        else:
            res = post_json("/search", {"q": q})
            st.session_state["raw_results"] = res
            cleaned, err, raw = _normalize_search_payload(res)
            st.session_state["results"] = cleaned
            st.session_state["results_err"] = err
with colC:
    if st.button("Clear"):
        st.session_state["results"] = []
        st.session_state["results_err"] = None
        st.session_state["raw_results"] = None
        # also clear selection & panels
        st.session_state["symbol"] = ""
        st.session_state.pop("symbol_pick", None)
        for k in ("quote", "series", "ind", "evt", "exp"):
            st.session_state.pop(k, None)

results = st.session_state.get("results") or []
results_err = st.session_state.get("results_err")
raw_results = st.session_state.get("raw_results")
symbol = st.session_state.get("symbol", "")

# Surface any search error
if results_err:
    st.warning(f"Search: {results_err}")

# Show raw payload for debugging when no valid rows
if not results and raw_results is not None and not results_err:
    st.info("No valid matches returned from the backend.")
    with st.expander("Raw search payload"):
        st.write(raw_results)

# Build selector (auto-pick if single result)
if results:
    labels = [f"{r['symbol']} — {r['name']}" for r in results]
    # auto-pick if one result and none selected yet
    if len(results) == 1 and not symbol:
        symbol = results[0]["symbol"]
        st.session_state["symbol"] = symbol
        st.success(f"Selected: {symbol}")

    # keep previously chosen symbol selected if present
    default_idx = 0
    if symbol:
        for i, r in enumerate(results):
            if r["symbol"] == symbol:
                default_idx = i
                break

    selected = st.selectbox("Pick symbol", labels, index=default_idx, key="symbol_pick")
    picked_symbol = selected.split(" — ")[0].strip() if selected else ""
    if picked_symbol and picked_symbol != symbol:
        st.session_state["symbol"] = picked_symbol
        symbol = picked_symbol

# --- Actions (only show when a symbol is picked) ---
if symbol:
    st.divider()
    st.subheader(f"Symbol: {symbol}")

    # Adjustable lookback for series
    lb = st.slider("Lookback (days)", min_value=90, max_value=720, value=180, step=30)

    cols = st.columns([1, 1, 1, 1, 1.5, 1])
    with cols[0]:
        if st.button("Get Quote"):
            res = post_json("/quote", {"symbol": symbol})
            if isinstance(res, dict) and res.get("error"):
                st.warning(f"Quote: {res['error']}")
            else:
                st.session_state["quote"] = res

    with cols[1]:
        if st.button("Series"):
            res = post_json("/series", {"symbol": symbol, "lookback": int(lb)})
            if isinstance(res, dict) and res.get("error"):
                msg = res.get("message") or res.get("_message") or ""
                st.warning(f"Series: {res['error']}{(' — ' + msg) if msg else ''}")
            else:
                st.session_state["series"] = res

    with cols[2]:
        if st.button("Indicators"):
            res = post_json("/indicators", {"symbol": symbol, "window_sma": 20, "window_ema": 50, "window_rsi": 14})
            if isinstance(res, dict) and res.get("error"):
                st.warning(f"Indicators: {res['error']}")
            else:
                st.session_state["ind"] = res

    with cols[3]:
        if st.button("Events"):
            res = post_json("/events", {"symbol": symbol})
            if isinstance(res, dict) and res.get("error"):
                st.warning(f"Events: {res['error']}")
            else:
                st.session_state["evt"] = res

    with cols[4]:
        if st.button("LLM Explain"):
            # Get API key from user input
            api_key = st.session_state.get("openai_api_key", "")
            if api_key:
                res = post_json("/explain", {"symbol": symbol, "openai_api_key": api_key}, timeout=60)
                st.session_state["exp"] = res
            else:
                st.error("Please enter your OpenAI API key above.")

    with cols[5]:
        if st.button("Bundle"):
            api_key = st.session_state.get("openai_api_key", "")
            bundle_payload = {
                "symbol": symbol, 
                "lookback": int(lb), 
                "window_sma": 20, 
                "window_ema": 50, 
                "window_rsi": 14
            }
            if api_key:
                bundle_payload["openai_api_key"] = api_key
            res = post_json("/bundle", bundle_payload, timeout=45)
            if isinstance(res, dict) and res.get("error"):
                st.warning(f"Bundle: {res['error']}")
            else:
                st.session_state["series"] = res.get("series", [])
                st.session_state["ind"] = res.get("indicators", {})
                st.session_state["evt"] = res.get("events", {})
                st.session_state["exp"] = res.get("explain", {})

# --- Render panels ---
if "quote" in st.session_state:
    st.subheader("Quote", divider="gray")
    st.json(st.session_state["quote"])

if "series" in st.session_state:
    st.subheader("Price Series", divider="gray")

    def _normalize_series(payload):
        """
        Accepts: list[dict] | dict | str(JSON)
        Returns: (DataFrame, err_dict|None)
        """
        rows = None
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return pd.DataFrame(), None

        if isinstance(payload, dict):
            # surface backend error dicts
            if payload.get("error") or payload.get("_error"):
                return pd.DataFrame(), payload
            # common wrappers
            for key in ("series", "data", "items", "result", "records"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
            if rows is None and any(isinstance(k, str) and k.isdigit() for k in payload.keys()):
                # odd numeric-key dicts
                try:
                    rows = list(payload.values())
                except Exception:
                    rows = None
        elif isinstance(payload, list):
            rows = payload

        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(), None

        df = pd.DataFrame(rows)

        # Normalize column names (lowercase)
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Map alt names -> expected names
        rename_map = {
            "date": "date", "datetime": "date", "time": "date", "timestamp": "date",
            "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
            "adj close": "close", "adjclose": "close", "adj_close": "close",
        }
        df = df.rename(columns=rename_map)

        # Handle capitalized yfinance names if they slipped through
        for up, lo in [("Date","date"),("Open","open"),("High","high"),("Low","low"),("Close","close"),("Volume","volume")]:
            if up.lower() not in df.columns and up in df.columns:
                df.rename(columns={up: lo}, inplace=True)

        # Ensure required cols exist
        for col in ["date","open","high","low","close","volume"]:
            if col not in df.columns:
                df[col] = pd.NA

        # Coerce types
        if pd.api.types.is_numeric_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Clean and order
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        return df[["date","open","high","low","close","volume"]], None

    series_payload = st.session_state["series"]
    df, err = _normalize_series(series_payload)

    if isinstance(err, dict):  # backend error surfaced
        st.warning(err.get("message") or err.get("_message") or "Series error")
        with st.expander("Raw series payload"):
            st.write(series_payload)
    elif df.empty or df["close"].dropna().empty or len(df) < 2:
        st.info("No series data to plot. Try a different symbol or increase Lookback.")
        with st.expander("Raw series payload"):
            st.write(series_payload)
    else:
        # Plot with Matplotlib (single plot, no specific colors)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(df["date"], df["close"])
            ax.set_xlabel("Date")
            ax.set_ylabel("Close")
            ax.set_title("Close vs. Date")
            fig.autofmt_xdate()
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"Plotting failed: {e}")

        st.dataframe(df.tail(10))

if "ind" in st.session_state:
    st.subheader("Indicators", divider="gray", help="SMA, EMA, RSI snapshot")
    st.json(st.session_state["ind"])

if "evt" in st.session_state:
    st.subheader("Events", divider="gray", help="Gap up/down, volatility spikes, 52w extremes")
    st.json(st.session_state["evt"])


if "exp" in st.session_state:
    st.subheader("Explanation", divider="gray")
    exp = st.session_state["exp"]
    st.write(exp if isinstance(exp, str) else exp.get("text") or json.dumps(exp, indent=2))
