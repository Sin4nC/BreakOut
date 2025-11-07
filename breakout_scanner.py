# MEXC Spot USDT — 4H Pump Scanner (signal-only, de-noised)
# Prints ONLY useful signals meeting ALL of:
# - green full-body body_ratio >= 0.7
# - close > prior_high(N) for N in {15, 20}
# - breakout strength >= MARGIN_PCT (default 1%)
# - 4H quote volume >= MIN_QV_USDT
# Returns at most MAX_RESULTS most-recent signals

import requests
import pandas as pd
import numpy as np
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"

# -------- settings --------
INTERVAL       = "4h"          # 4H only
N_LIST         = [15, 20]
SEARCH_WINDOW  = 180           # how many last candles to scan
BODY_RATIO     = 0.7
MARGIN_PCT     = 0.01          # close must be > prior_high * (1 + MARGIN_PCT)
MIN_QV_USDT    = 100_000       # min quote volume on the signal candle
MAX_RESULTS    = 60            # limit printed rows
LIMIT          = 300           # klines per symbol
MAX_WORKERS    = 10

# exclude leveraged style tickers
EXCLUDES = ("3L","3S","5L","5S","UP","DOWN","BULL","BEAR")

# -------- helpers --------
def get_symbols():
    r = requests.get(f"{BASE}/api/v3/ticker/price", timeout=25)
    r.raise_for_status()
    syms = []
    for item in r.json():
        sym = item.get("symbol","")
        if not sym.endswith("USDT"):
            continue
        if any(x in sym for x in EXCLUDES):
            continue
        syms.append(sym)
    return sorted(set(syms))

def fetch_klines(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"{BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        params["interval"] = "Hour4"
        r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows or len(rows[0]) < 6:
        return None
    cols = ["t","o","h","l","c","v","ct","qv","n","tb","tqv","i"]
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # quote volume if present
    if "qv" in df.columns:
        df["qv"] = pd.to_numeric(df["qv"], errors="coerce")
    else:
        df["qv"] = np.nan
    return df

def body_ratio(o,h,l,c):
    rng = h - l
    if rng <= 0:
        return 0.0
    return abs(c - o) / rng

def scan_symbol(sym):
    try:
        df = fetch_klines(sym)
        if df is None or len(df) < max(N_LIST) + 2:
            return []

        o = df["o"].to_numpy(float)
        h = df["h"].to_numpy(float)
        l = df["l"].to_numpy(float)
        c = df["c"].to_numpy(float)
        qv = df["qv"].to_numpy(float)

        last_idx = len(df) - 1
        win_start = max(0, last_idx - SEARCH_WINDOW + 1)

        rng = np.maximum(h - l, 1e-12)
        br  = np.abs(c - o) / rng
        green_full = (c > o) & (br >= BODY_RATIO)

        hits_idx = set()
        best_n_at = {}

        for n in N_LIST:
            prior_high = pd.Series(h).rolling(n).max().shift(1).to_numpy()
            # breakout strength
            strength = (c / np.maximum(prior_high, 1e-12)) - 1.0
            cond = green_full & (strength >= MARGIN_PCT)
            idxs = np.where(cond)[0]
            for i in idxs:
                if i < win_start:
                    continue
                # volume filter on the same candle
                if not np.isfinite(qv[i]) or qv[i] < MIN_QV_USDT:
                    continue
                hits_idx.add(i)
                # prefer bigger n if both satisfied
                if i not in best_n_at or n > best_n_at[i]:
                    best_n_at[i] = n

        if not hits_idx:
            return []

        out = []
        for i in sorted(hits_idx):
            row = df.iloc[i]
            n_used = best_n_at.get(i, max(N_LIST))
            ph = pd.Series(h).rolling(n_used).max().shift(1).iloc[i]
            bratio = body_ratio(row["o"], row["h"], row["l"], row["c"])
            strength = (row["c"] / max(ph, 1e-12)) - 1.0
            out.append({
                "symbol": sym,
                "time_utc": datetime.fromtimestamp(int(row["t"])//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "close": float(row["c"]),
                "body": round(bratio, 3),
                "n_used": int(n_used),
                "breakout_pct": round(strength * 100, 2),
                "qv_usdt": float(qv[i]) if np.isfinite(qv[i]) else float("nan"),
                "candles_ago": int(last_idx - i),
            })
        return out
    except Exception:
        return []

def run_round(symbols):
    rows = []
    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for lst in ex.map(scan_symbol, symbols):
            if lst:
                rows.extend(lst)
    # sort recent first then stronger
    rows.sort(key=lambda r: (r["candles_ago"], -r["breakout_pct"], -r["body"]))
    return rows[:MAX_RESULTS]

if __name__ == "__main__":
    symbols = get_symbols()
    hits = run_round(symbols)

    if not hits:
        print("no signals")
    else:
        print("symbol,time_utc,close,body,n_used,breakout_pct,qv_usdt,candles_ago")
        for r in hits:
            print(f"{r['symbol']},{r['time_utc']},{r['close']:.8g},{r['body']:.2f},"
                  f"{r['n_used']},{r['breakout_pct']:.2f},{int(r['qv_usdt'])},{r['candles_ago']}")
