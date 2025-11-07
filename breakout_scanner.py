# MEXC Spot USDT — 4H Pump Scanner
# Returns ALL symbols that have at least one candle in the recent window satisfying BOTH:
#   1) green full-body with body_ratio >= 0.7
#   2) close > max(high of previous N) for N in {15, 20}

import requests
import pandas as pd
import numpy as np
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"

INTERVAL = "4h"          # 4-hour only
N_LIST = [15, 20]        # both lookbacks
SEARCH_WINDOW = 400      # number of most-recent candles to scan per symbol

BODY_RATIO = 0.7         # full-body threshold
LIMIT = 300              # klines per request
MAX_WORKERS = 10

# ---------- symbol universe ----------
def get_symbols():
    # reliable broad list of spot tickers
    r = requests.get(f"{BASE}/api/v3/ticker/price", timeout=25)
    r.raise_for_status()
    syms = []
    for item in r.json():
        sym = item.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        # skip leveraged/synthetic names
        if any(x in sym for x in ["3L","3S","5L","5S","UP","DOWN","BULL","BEAR"]):
            continue
        syms.append(sym)
    return sorted(set(syms))

# ---------- data fetch ----------
def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        params["interval"] = "Hour4"  # fallback alias if needed
        r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows or len(rows[0]) < 6:
        return None
    cols = ["t","o","h","l","c","v","ct","qv","n","tb","tqv","i"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ---------- logic ----------
def find_hits_4h(df):
    """
    Return list of indices i that satisfy BOTH rules for ANY N in N_LIST
    within the last SEARCH_WINDOW candles
    """
    if df is None or len(df) < max(N_LIST) + 2:
        return []

    last_idx = len(df) - 1
    window_start = max(0, last_idx - SEARCH_WINDOW + 1)

    o = df["o"].to_numpy(float)
    h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)

    rng = np.maximum(h - l, 1e-12)
    br = np.abs(c - o) / rng
    green_full = (c > o) & (br >= BODY_RATIO)

    # collect all indices that satisfy for either N
    idx_set = set()
    for n in N_LIST:
        prior_high = pd.Series(h).rolling(n).max().shift(1).to_numpy()
        cond = green_full & (c > prior_high)  # strict breakout
        idxs = np.where(cond)[0]
        for i in idxs:
            if i >= window_start:
                idx_set.add(i)

    return sorted(idx_set)

def ts_utc(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def body_ratio_row(row):
    rng = row["h"] - row["l"]
    if rng <= 0:
        return 0.0
    return abs(row["c"] - row["o"]) / rng

def scan_symbol(sym):
    try:
        df = fetch_klines(sym, INTERVAL)
        idxs = find_hits_4h(df)
        if not idxs:
            return None

        last_idx = len(df) - 1
        # most recent occurrence for summary
        i_latest = idxs[-1]
        row = df.iloc[i_latest]
        summary = {
            "symbol": sym,
            "latest_time_utc": ts_utc(row["t"]),
            "latest_close": float(row["c"]),
            "latest_body": round(body_ratio_row(row), 3),
            "candles_ago": int(last_idx - i_latest),
            "occurrences": len(idxs),
        }
        return summary
    except Exception:
        return None

def run_round(symbols):
    results = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(scan_symbol, symbols):
            if res:
                results.append(res)
    # sort by most recent first
    return sorted(results, key=lambda x: x["candles_ago"])

# ---------- main ----------
if __name__ == "__main__":
    symbols = get_symbols()
    hits = run_round(symbols)

    if not hits:
        print("no signals")
    else:
        print("symbol, latest_time_utc, latest_close, latest_body, candles_ago, occurrences")
        for s in hits:
            print(f"{s['symbol']}, {s['latest_time_utc']}, {s['latest_close']:.8g}, "
                  f"{s['latest_body']:.2f}, {s['candles_ago']}, {s['occurrences']}")
