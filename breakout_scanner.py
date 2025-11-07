# Pump strategy scanner — MEXC Spot USDT — 4h only
# any candle in last SEARCH_WINDOW bars that is:
#   green full-body (body >= 0.7) AND close > max(high of previous N), N in {15, 20}

import requests
import pandas as pd
import numpy as np
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"
USDT_ONLY = True

INTERVAL = "4h"
N_LIST = [15, 20]
SEARCH_WINDOW = 120          # recent window to report

BODY_RATIO = 0.7             # full-body threshold
EPS_PCT = 0.001              # 0.1% tolerance

LIMIT = 300
MAX_WORKERS = 8

def get_symbols():
    """Fetch MEXC spot symbols reliably  no Binance-only flags."""
    r = requests.get(f"{BASE}/api/v3/exchangeInfo", timeout=25)
    r.raise_for_status()
    out = []
    for s in r.json().get("symbols", []):
        status = str(s.get("status", "")).upper()
        if status not in ("TRADING", "ENABLED", "ONLINE"):
            continue
        sym = s.get("symbol", "")
        if USDT_ONLY and not sym.endswith("USDT"):
            continue
        # skip leveraged or synthetic names
        if any(x in sym for x in ["3L","3S","5L","5S","UP","DOWN","BULL","BEAR"]):
            continue
        perms = s.get("permissions") or s.get("permission") or []
        if perms and "SPOT" not in perms:
            continue
        out.append(sym)
    return sorted(set(out))

def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        params["interval"] = "Hour4"  # MEXC alt
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

def body_ratio(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return 0.0
    return abs(c - o) / rng

def find_breakouts_4h(df):
    if df is None or len(df) < max(N_LIST) + 2:
        return []
    o = df["o"].to_numpy(float)
    h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)

    br = np.divide(np.abs(c - o), np.maximum(h - l, 1e-12))
    green_full = (c > o) & (br >= BODY_RATIO)

    last_idx = len(df) - 1
    window_start = max(0, last_idx - SEARCH_WINDOW + 1)

    results = []
    for n in N_LIST:
        # prior high over previous n bars
        prior_high = pd.Series(h).rolling(n).max().shift(1).to_numpy()
        # tolerate tiny rounding by allowing close >= prior_high * (1 - eps)
        thresh = np.maximum(prior_high * (1 - EPS_PCT), prior_high - 1e-12)
        cond = green_full & (c >= thresh)

        idxs = np.where(cond)[0]
        for i in idxs:
            if i >= window_start:
                results.append((i, n))

    # dedupe per index prefer larger N
    results.sort(key=lambda x: (x[0], -x[1]))
    seen = {}
    for i, n in results:
        if i not in seen:
            seen[i] = n
    return [(i, seen[i]) for i in sorted(seen.keys())]

def human_time(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_symbol(sym):
    alerts = []
    try:
        df = fetch_klines(sym, INTERVAL)
        hits = find_breakouts_4h(df)
        last_idx = len(df) - 1 if df is not None else -1
        for i, n_used in hits:
            row = df.iloc[i]
            candles_ago = last_idx - i
            alerts.append(
                f"{sym} 4h PUMP n={n_used} candles_ago={candles_ago} "
                f"time {human_time(row['t'])} close {row['c']:.6g} "
                f"body {body_ratio(row['o'],row['h'],row['l'],row['c']):.2f}"
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
