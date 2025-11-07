# Simple Pump Scanner for MEXC USDT Spot - 4h Candles
# Finds green full-body candles where close > max high of prior 15/20 candles
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

BASE = "https://api.mexc.com"
INTERVAL = "4h"
N_LIST = [15, 20]
SEARCH_WINDOW = 200  # Look back ~33 days
BODY_RATIO = 0.65  # Full-body threshold (adjust if needed)
LIMIT = 300  # Candles to fetch

def get_symbols():
    r = requests.get(f"{BASE}/api/v3/exchangeInfo")
    r.raise_for_status()
    symbols = []
    for s in r.json()["symbols"]:
        if s["status"] != "1":  # 1 means TRADING
            continue
        if not s.get("isSpotTradingAllowed", False):
            continue
        sym = s["symbol"]
        if not sym.endswith("USDT"):
            continue
        if "SPOT" not in s.get("permissions", []):
            continue
        if any(x in sym for x in ["3L", "3S", "5L", "5S"]):
            continue
        symbols.append(sym)
    return symbols

def fetch_klines(sym):
    params = {"symbol": sym, "interval": INTERVAL, "limit": LIMIT}
    r = requests.get(f"{BASE}/api/v3/klines", params=params)
    if r.status_code != 200:
        params["interval"] = "Hour4"
        r = requests.get(f"{BASE}/api/v3/klines", params=params)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data, columns=["t", "o", "h", "l", "c", "v"])
    df[["o", "h", "l", "c"]] = df[["o", "h", "l", "c"]].astype(float)
    return df

def body_ratio(o, h, l, c):
    return abs(c - o) / (h - l) if (h - l) > 0 else 0

def find_pumps(df):
    if df is None or len(df) < max(N_LIST) + 1:
        return []
    o, h, l, c = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    br = np.array([body_ratio(o[i], h[i], l[i], c[i]) for i in range(len(df))])
    green_full = (c > o) & (br >= BODY_RATIO)
    
    pumps = []
    for n in N_LIST:
        prior_high = pd.Series(h).shift(1).rolling(n).max().values
        eps = np.maximum(np.abs(prior_high) * 0.001, 1e-12)  # tolerance
        cond = green_full & (c > (prior_high + eps))
        for idx in np.where(cond)[0]:
            if idx >= len(df) - SEARCH_WINDOW:
                pumps.append((idx, n))
    return pumps

def human_time(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_all():
    symbols = get_symbols()
    hits = []
    for sym in symbols:
        try:
            df = fetch_klines(sym)
            pumps = find_pumps(df)
            for idx, n in pumps:
                row = df.iloc[idx]
                ago = len(df) - 1 - idx
                hit = f"{sym} 4h PUMP n={n} ago={ago} time={human_time(row['t'])} close={row['c']:.6g} body={body_ratio(row['o'], row['h'], row['l'], row['c']):.2f}"
                hits.append(hit)
        except Exception as e:
            print(f"Error for {sym}: {e}")
    if hits:
        for h in sorted(hits):
            print(h)
    else:
        print("No pumps found")

if __name__ == "__main__":
    scan_all()
