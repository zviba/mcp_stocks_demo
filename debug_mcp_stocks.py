# debug_mcp_stocks.py
# Run this script to debug the data flow issues

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

# Test the datasource directly
def test_datasource():
    print("=== TESTING DATASOURCE DIRECTLY ===")
    try:
        from datasource import search_symbols, latest_quote, price_series
        
        # Test search
        print("1. Testing search...")
        search_result = search_symbols("NVDA")
        print(f"Search result: {search_result}")
        
        # Test quote
        print("\n2. Testing quote...")
        quote_result = latest_quote("NVDA")
        print(f"Quote result: {quote_result}")
        
        # Test price series
        print("\n3. Testing price series...")
        series_result = price_series("NVDA", "daily", 30)
        print(f"Series shape: {series_result.shape}")
        print(f"Series columns: {list(series_result.columns)}")
        print(f"First few rows:\n{series_result.head()}")
        print(f"Data types:\n{series_result.dtypes}")
        
        return series_result
        
    except Exception as e:
        print(f"Datasource test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test the MCP server tools
def test_mcp_tools():
    print("\n=== TESTING MCP TOOLS ===")
    try:
        from mcp_server import price_series as mcp_price_series, latest_quote as mcp_quote
        
        # Test MCP price series
        print("1. Testing MCP price_series...")
        mcp_result = mcp_price_series("NVDA", "daily", 30)
        print(f"MCP result type: {type(mcp_result)}")
        print(f"MCP result: {mcp_result[:500]}...")  # First 500 chars
        
        # Try to parse the JSON
        try:
            parsed = json.loads(mcp_result)
            print(f"Parsed successfully. Type: {type(parsed)}")
            if isinstance(parsed, list):
                print(f"List length: {len(parsed)}")
                if parsed:
                    print(f"First item: {parsed[0]}")
            elif isinstance(parsed, dict):
                print(f"Dict keys: {list(parsed.keys())}")
        except Exception as parse_e:
            print(f"JSON parse failed: {parse_e}")
            
        return mcp_result
        
    except Exception as e:
        print(f"MCP tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test the API endpoint
def test_api_endpoint():
    print("\n=== TESTING API ENDPOINT ===")
    backend_url = "http://127.0.0.1:8001"
    
    try:
        # Test health
        health_response = requests.get(f"{backend_url}/health", timeout=5)
        print(f"Health check: {health_response.status_code} - {health_response.json()}")
        
        # Test series endpoint
        payload = {"symbol": "NVDA", "interval": "daily", "lookback": 30}
        response = requests.post(f"{backend_url}/series", json=payload, timeout=30)
        print(f"Series response status: {response.status_code}")
        print(f"Series response headers: {dict(response.headers)}")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            result = response.json()
            print(f"API result type: {type(result)}")
            print(f"API result: {str(result)[:500]}...")
            return result
        else:
            print(f"Non-JSON response: {response.text[:500]}")
            
    except Exception as e:
        print(f"API test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test the Streamlit normalization function
def test_streamlit_normalization():
    print("\n=== TESTING STREAMLIT NORMALIZATION ===")
    
    # Simulate different payload formats
    test_payloads = [
        # Direct list format
        [{"date": "2023-01-01", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000}],
        
        # String JSON format
        '[{"date": "2023-01-01", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000}]',
        
        # Error format
        {"error": "test_error", "message": "This is a test error"},
        
        # Empty list
        [],
        
        # Wrapped format
        {"series": [{"date": "2023-01-01", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000}]}
    ]
    
    def _normalize_series(payload):
        """Copy of the function from streamlit_app.py"""
        rows = None
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return pd.DataFrame(), None

        if isinstance(payload, dict):
            if payload.get("error") or payload.get("_error"):
                return pd.DataFrame(), payload
            for key in ("series", "data", "items", "result", "records"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
            if rows is None and isinstance(payload.get("0"), dict):
                rows = list(payload.values())
        elif isinstance(payload, list):
            rows = payload

        if not isinstance(rows, list):
            return pd.DataFrame(), None

        if not rows:
            return pd.DataFrame(), None

        df = pd.DataFrame(rows)
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        rename_map = {
            "date": "date", "datetime": "date", "time": "date", "timestamp": "date",
            "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
            "adj close": "close", "adjclose": "close", "adj_close": "close",
        }
        df = df.rename(columns=rename_map)

        for up, lo in [("Date","date"),("Open","open"),("High","high"),("Low","low"),("Close","close"),("Volume","volume")]:
            if up.lower() not in df.columns and up in df.columns:
                df.rename(columns={up: lo}, inplace=True)

        for col in ["date","open","high","low","close","volume"]:
            if col not in df.columns:
                df[col] = pd.NA

        if pd.api.types.is_numeric_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df[["date","open","high","low","close","volume"]], None
    
    for i, payload in enumerate(test_payloads):
        print(f"\nTest payload {i+1}: {str(payload)[:100]}...")
        df, err = _normalize_series(payload)
        print(f"Result - Shape: {df.shape}, Error: {err}")
        if not df.empty:
            print(f"Columns: {list(df.columns)}")
            print(f"Sample:\n{df.head(2)}")

if __name__ == "__main__":
    # Set up environment
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(), override=False)
    except:
        pass
    
    print("MCP Stocks Analyzer Debug Script")
    print("=" * 50)
    
    # Run all tests
    datasource_result = test_datasource()
    mcp_result = test_mcp_tools()
    api_result = test_api_endpoint()
    test_streamlit_normalization()
    
    print("\n=== SUMMARY ===")
    print(f"Datasource test: {'✅ PASS' if datasource_result is not None else '❌ FAIL'}")
    print(f"MCP tools test: {'✅ PASS' if mcp_result is not None else '❌ FAIL'}")
    print(f"API endpoint test: {'✅ PASS' if api_result is not None else '❌ FAIL'}")
    
    print("\n=== RECOMMENDATIONS ===")
    if datasource_result is None:
        print("1. Check your FINNHUB_API_KEY and yfinance installation")
    if mcp_result is None:
        print("2. Check MCP server tool implementation")
    if api_result is None:
        print("3. Make sure the FastAPI server is running on port 8001")
    
    print("\n4. Check the console output above for specific error details")