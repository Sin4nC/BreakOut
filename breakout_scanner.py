# Breakout scanner for MEXC USDT spot pairs
# Finds the FIRST near full-body breakout over the highs of last N candles
# Searches a recent window so you dont miss moves that happened a few candles ago

import requests
import pandas as pd
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"
USDT_ONLY = True

# for faster test keep only 4h first  you can add "1d","1w" later
INTERVALS = ["4h"]

N_LOOKBACK = 12          # set 10..15 as you like
BODY_RATIO_MIN = 0.6     # full body threshold
SEARCH_WINDOW = 30       # how many recent candles to scan
EPS_PCT = 0.002          # 0.2 percent tolerance around prior high

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
        alt = {"4h": "Hour4", "1d": "Day1", "1w": "Week1"}.get(interval)
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

def is_full_body(o, h, l, c, min_ratio=BODY_RATIO_MIN):
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    return (body / rng) >= min_ratio

def breakout_index(df):
    """return index of the FIRST breakout in the last SEARCH_WINDOW candles or None"""
    if df is None or len(df) < N_LOOKBACK + 2:
        return None
    start = max(N_LOOKBACK + 1, len(df) - SEARCH_WINDOW)
    for i in range(start, len(df)):
        o, h, l, c = df.iloc[i][["o","h","l","c"]]
        op, hp, lp, cp = df.iloc[i-1][["o","h","l","c"]]
        prior_high      = df["h"].iloc[i - N_LOOKBACK : i].max()
        prior_high_prev = df["h"].iloc[i - N_LOOKBACK - 1 : i - 1].max()

        # body quality
        body_ok = is_full_body(o, h, l, c)

        # strict body entirely above prior high with tiny tolerance
        strict_break = min(o, c) > prior_high * (1 - EPS_PCT)

        # relaxed  close above prior high and at least 90 percent of body sits above
        body_len = abs(c - o)
        above_len = max(0.0, max(o, c) - prior_high)
        relaxed_break = (c > prior_high) and (body_len > 0) and (above_len / body_len >= 0.9)

        # ensure previous candle was not already a qualifying breakout
        prev_body_ok = is_full_body(op, hp, lp, cp)
        prev_strict = min(op, cp) > prior_high_prev * (1 - EPS_PCT)
        prev_above_len = max(0.0, max(op, cp) - prior_high_prev)
        prev_body_len = abs(cp - op)
        prev_relaxed = (cp > prior_high_prev) and (prev_body_len > 0) and (prev_above_len / prev_body_len >= 0.9)
        prev_break = prev_body_ok and (prev_strict or prev_relaxed)

        if body_ok and (strict_break or relaxed_break) and not prev_break:
            return i
    return None

def human_time(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_symbol(sym):
    alerts = []
    for itv in INTERVALS:
        try:
            df = fetch_klines(sym, itv)
            idx = breakout_index(df)
            if idx is not None:
                row = df.iloc[idx]
                alerts.append(
                    f"{sym}  {itv}  ALERT  FIRST NEAR FULL BODY BREAK  "
                    f"lookback {N_LOOKBACK}  candle_time {human_time(row['t'])}  close {row['c']:.6g}"
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
