# Pump strategy scanner — MEXC Spot USDT — 4h only
# Checks for a green full-body 4h candle whose CLOSE > max(high of last N) with N in {15, 20}
# Reports if such a breakout happened within last SEARCH_WINDOW candles
import requests
import pandas as pd
import numpy as np
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"
USDT_ONLY = True
INTERVAL = "4h"  # 4h only
N_LIST = [15, 20]  # set to [20] if you want strictly 20
SEARCH_WINDOW = 120  # increased to catch older examples like early Nov
BODY_RATIO = 0.7  # full-body threshold for 4h
EPS_PCT = 0.001  # 0.1% tolerance to avoid rounding edge cases
LIMIT = 300
MAX_WORKERS = 8
DEBUG_SYMBOLS = ["HIPPOUSDT", "AIAUSDT"]  # for extra debug

def get_symbols():
    r = requests.get(f"{BASE}/api/v3/exchangeInfo", timeout=25)
    r.raise_for_status()
    out = []
    for s in r.json().get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("spotTradingAllowed") is not True:
            continue
        sym = s.get("symbol", "")
        if USDT_ONLY and not sym.endswith("USDT"):
            continue
        if any(x in sym for x in ["3L", "3S", "5L", "5S"]):
            continue
        out.append(sym)
    return sorted(set(out))

def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        params["interval"] = "Hour4"  # MEXC alt name
        r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        print(f"Error fetching {symbol}: {r.status_code}")
        return None
    rows = r.json()
    if not rows or len(rows[0]) < 6:
        return None
    cols = ["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tqv", "i"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def body_ratio(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return 0.0
    return abs(c - o) / rng

def check_breakout_4h(df):
    """Return list of (index, n_used) for breakout candles matching the condition."""
    if df is None or len(df) < max(N_LIST) + 1:
        return None
    o = df["o"].to_numpy(float)
    h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)
    br = np.divide(np.abs(c - o), np.maximum(h - l, 1e-12))
    green_full = (c > o) & (br >= BODY_RATIO)
    
    candidates = []
    for n in N_LIST:
        # prior high for current bar = rolling max of previous n highs
        ph = pd.Series(h).shift(1).rolling(n).max().to_numpy()
        eps = np.maximum(np.abs(ph) * EPS_PCT, 1e-12)
        cond = green_full & (c > (ph + eps))
        idxs = np.where(cond)[0]
        if len(idxs):
            candidates.extend([(i, n) for i in idxs])  # all matching
    
    if not candidates:
        return None
    
    # sort by index (earliest first)
    candidates.sort(key=lambda x: x[0])
    return candidates  # list of (index, n_used)

def human_time(ms):
    return datetime.fromtimestamp(int(ms) // 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_symbol(sym):
    alerts = []
    try:
        df = fetch_klines(sym, INTERVAL)
        if df is None:
            print(f"DEBUG {sym}: No data fetched")
            return alerts
        res = check_breakout_4h(df)
        if res is None:
            i = len(df) - 1
            row = df.iloc[i]
            ph20 = pd.Series(df["h"]).shift(1).rolling(20).max().iloc[i]
            print(f"DEBUG {sym} last body={body_ratio(row['o'], row['h'], row['l'], row['c']):.3f} "
                  f"close={row['c']:.6g} prior20={ph20:.6g} time={human_time(row['t'])}")
            return alerts
        
        last_idx = len(df) - 1
        for idx, n_used in res:
            if idx >= last_idx - SEARCH_WINDOW + 1:  # only recent
                row = df.iloc[idx]
                candles_ago = last_idx - idx
                alerts.append(
                    f"{sym} 4h PUMP n={n_used} candles_ago={candles_ago} "
                    f"time {human_time(row['t'])} close {row['c']:.6g} "
                    f"body {body_ratio(row['o'], row['h'], row['l'], row['c']):.2f}"
                )
    except Exception as e:
        print(f"Error scanning {sym}: {e}")
    return alerts

def run_round(symbols):
    found = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(scan_symbol, symbols):
            if res:
                found.extend(res)
    return found

if __name__ == "__main__":
    # symbols = get_symbols()  # uncomment to scan all
    symbols = ["HIPPOUSDT", "AIAUSDT"]  # limited for testing
    print(f"Scanning {len(symbols)} symbols...")
    hits = run_round(symbols)
    if hits:
        for h in sorted(hits):
            print(h)
    else:
        print("no signals")
