# Pump strategy scanner for MEXC USDT spot pairs
# Detect FIRST green full-body candle whose CLOSE breaks above max high of last N candles (N in {15,20})
# Scans entire series to find the first breakout, then reports if it occurred within SEARCH_WINDOW candles

import requests
import pandas as pd
import numpy as np
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"
USDT_ONLY = True

INTERVALS = ["4h", "1d", "1w"]
N_LIST = [15, 20]
SEARCH_WINDOW = 60

BODY_RATIO_BY_INTERVAL = {"4h": 0.7, "1d": 0.6, "1w": 0.6}
EPS_PCT = 0.001  # 0.1% tolerance for close > prior_high

LIMIT = 300
MAX_WORKERS = 8

DEBUG_SYMBOLS = []  # e.g. ["HIPPOUSDT","AIAUSDT"] to print debug while scanning ALL symbols

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
        if any(x in sym for x in ["3L","3S","5L","5S"]):
            continue
        out.append(sym)
    return sorted(set(out))

def fetch_klines(symbol, interval, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        alt = {"4h": "Hour4", "1d": "Day1", "1w": "Week1"}.get(interval)
        if alt:
            params["interval"] = alt
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

def first_breakouts(df, interval):
    """return list of (idx, n_used) for FIRST breakouts in whole series"""
    if df is None:
        return []
    min_ratio = BODY_RATIO_BY_INTERVAL.get(interval, 0.6)
    nmax = max(N_LIST)
    if len(df) < nmax + 2:
        return []
    o = df["o"].to_numpy(float)
    h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float)
    c = df["c"].to_numpy(float)
    br = np.divide(np.abs(c - o), np.maximum(h - l, 1e-12))
    green_full = (c > o) & (br >= min_ratio)

    out = []
    for n in N_LIST:
        # rolling max of highs over previous n bars
        ph = pd.Series(h).rolling(n, min_periods=n).max().shift(0).to_numpy()
        # prior highs aligned so that ph[i-1] is max over bars [i-n, i-1]
        # we want close[i] > max(high[i-n : i-1])
        ph_now  = np.concatenate(([np.nan], pd.Series(h).rolling(n).max().shift(1).to_numpy()[1:]))
        ph_prev = np.concatenate(([np.nan], pd.Series(h).rolling(n).max().shift(1).to_numpy()[1:]))  # same window for i-1

        # due to shift trick above, compute directly for clarity
        ph_now  = pd.Series(h).shift(1).rolling(n).max().to_numpy()
        ph_prev = pd.Series(h).shift(2).rolling(n).max().to_numpy()

        eps = np.maximum(np.abs(ph_now) * EPS_PCT, 1e-12)
        cond_now  = green_full & (c > (ph_now + eps))
        cond_prev = (np.roll(green_full, 1)) & (np.roll(c, 1) > (ph_prev + np.maximum(np.abs(ph_prev)*EPS_PCT, 1e-12)))

        # first breakout = qualifies now AND not qualified on previous bar
        first = cond_now & (~cond_prev)

        idxs = np.where(first)[0]
        for i in idxs:
            if i >= n:  # ensure window exists
                out.append((i, n))
    # keep unique indices preferring larger N (20 over 15) by sorting
    out.sort(key=lambda x: (x[0], -x[1]))
    uniq = {}
    for i, n in out:
        if i not in uniq:
            uniq[i] = n
    return [(i, uniq[i]) for i in sorted(uniq.keys())]

def human_time(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_symbol(sym):
    alerts = []
    for itv in INTERVALS:
        try:
            df = fetch_klines(sym, itv)
            hits = first_breakouts(df, itv)
            if not hits:
                if sym in DEBUG_SYMBOLS and df is not None and len(df) > 0:
                    last = df.iloc[-1]
                    print(f"DEBUG {sym} {itv} last close {last['c']:.6g}")
                continue
            last_idx = len(df) - 1
            for i, n_used in hits:
                if i >= last_idx - SEARCH_WINDOW + 1:
                    row = df.iloc[i]
                    candles_ago = last_idx - i
                    alerts.append(
                        f"{sym} {itv} PUMP n={n_used} candles_ago={candles_ago} "
                        f"time {human_time(row['t'])} close {row['c']:.6g} body {body_ratio(row['o'],row['h'],row['l'],row['c']):.2f}"
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
