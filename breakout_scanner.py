# Breakout scanner for MEXC USDT spot pairs
# Rule: FIRST full-body breakout above the highs of the last 20 candles
# 4h body ratio = 0.7, 1d/1w = 0.6

import requests
import pandas as pd
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"
USDT_ONLY = True

INTERVALS = ["4h", "1d", "1w"]   # you can set ["4h"] if you want only 4H

N_LOOKBACK = 20                  # prior-high window
SEARCH_WINDOW = 60               # scan recent candles so we dont miss earlier signals

BODY_RATIO_BY_INTERVAL = {
    "4h": 0.7,
    "1d": 0.6,
    "1w": 0.6,
}

LIMIT = 300
MAX_WORKERS = 6

def get_symbols():
    r = requests.get(f"{BASE}/api/v3/exchangeInfo", timeout=20)
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
        if any(x in sym for x in ["3L","3S","5L","5S"]):
            continue
        out.append(sym)
    return sorted(set(out))

def fetch_klines(symbol, interval, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        alt = {"4h":"Hour4","1d":"Day1","1w":"Week1"}.get(interval)
        if alt:
            params["interval"] = alt
            r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    rows = r.json()
    if not rows or len(rows[0]) < 6:
        return None
    cols = ["t","o","h","l","c","v","ct","qv","n","tb","tqv","i"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def is_full_body(o, h, l, c, min_ratio):
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    return (body / rng) >= min_ratio

def qualifies(df, i, min_ratio, n_lookback):
    """does candle i qualify as full-body breakout with CLOSE above prior high"""
    o, h, l, c = df.iloc[i][["o","h","l","c"]]
    prior_high = df["h"].iloc[i - n_lookback : i].max()
    return is_full_body(o, h, l, c, min_ratio) and (c > prior_high)

def breakout_index(df, interval):
    if df is None or len(df) < N_LOOKBACK + 2:
        return None
    min_ratio = BODY_RATIO_BY_INTERVAL.get(interval, 0.6)
    start = max(N_LOOKBACK + 1, len(df) - SEARCH_WINDOW)
    for i in range(start, len(df)):
        # current candle must qualify
        if not qualifies(df, i, min_ratio, N_LOOKBACK):
            continue
        # previous candle must NOT qualify  so this is the FIRST full-body breakout
        if qualifies(df, i-1, min_ratio, N_LOOKBACK):
            continue
        return i
    return None

def human_time(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_symbol(sym):
    alerts = []
    for itv in INTERVALS:
        try:
            df = fetch_klines(sym, itv)
            idx = breakout_index(df, itv)
            if idx is not None:
                row = df.iloc[idx]
                alerts.append(
                    f"{sym}  {itv}  ALERT  FIRST FULL-BODY BREAK "
                    f"lookback {N_LOOKBACK}  time {human_time(row['t'])}  close {row['c']:.6g}"
                )
        except Exception:
            pass
    return alerts

def run_round(symbols):
    found = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(scan_symbol, symbols):
            if res:
                found.extend(res)
    return found

if __name__ == "__main__":
    symbols = get_symbols()
    hits = run_round(symbols)
    if hits:
        for h in hits:
            print(h)
    else:
        print("no signals")
