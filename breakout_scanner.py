# Breakout scanner for MEXC USDT spot pairs
# Rule: FIRST full-body breakout above the highs of the last 20 candles
# 4h body ratio = 0.7  daily/weekly = 0.6

import requests
import pandas as pd
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"
USDT_ONLY = True

# intervals to scan  add/remove as needed
INTERVALS = ["4h", "1d", "1w"]

# lookback for prior highs
N_LOOKBACK = 20

# window of recent candles to search so you do not miss earlier signals
SEARCH_WINDOW = 60

# body ratio per interval  per your rule 4h stricter
BODY_RATIO_BY_INTERVAL = {
    "4h": 0.7,
    "1d": 0.6,
    "1w": 0.6,
}

# request + scan settings
LIMIT = 300
MAX_WORKERS = 6

def get_symbols():
    """Fetch all spot USDT symbols from MEXC and filter out leveraged tokens."""
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
        if any(x in sym for x in ["3L", "3S", "5L", "5S"]):
            continue
        out.append(sym)
    return sorted(set(out))

def fetch_klines(symbol, interval, limit=LIMIT):
    """Read klines  fallback to Hour4 Day1 Week1 if needed."""
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

def is_full_body(o, h, l, c, min_ratio):
    """True if body length is at least min_ratio of total range."""
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    return (body / rng) >= min_ratio

def breakout_index(df, interval):
    """
    Return index of the FIRST candle in the last SEARCH_WINDOW that:
    - is full-body per BODY_RATIO_BY_INTERVAL[interval]
    - OPEN and CLOSE are BOTH above the max high of the previous N_LOOKBACK candles
    - previous candle was NOT already such a breakout
    """
    if df is None or len(df) < N_LOOKBACK + 2:
        return None

    min_ratio = BODY_RATIO_BY_INTERVAL.get(interval, 0.6)
    # search older -> newer to ensure we alert the first one
    start = max(N_LOOKBACK + 1, len(df) - SEARCH_WINDOW)
    for i in range(start, len(df)):
        o, h, l, c = df.iloc[i][["o","h","l","c"]]
        op, hp, lp, cp = df.iloc[i-1][["o","h","l","c"]]

        prior_high      = df["h"].iloc[i - N_LOOKBACK : i].max()
        prior_high_prev = df["h"].iloc[i - N_LOOKBACK - 1 : i - 1].max()

        # current candle conditions
        body_ok   = is_full_body(o, h, l, c, min_ratio)
        break_now = (o > prior_high) and (c > prior_high)

        # ensure previous candle was not already a valid breakout
        prev_body_ok   = is_full_body(op, hp, lp, cp, min_ratio)
        prev_break_now = (op > prior_high_prev) and (cp > prior_high_prev)
        was_prev_break = prev_body_ok and prev_break_now

        if body_ok and break_now and not was_prev_break:
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
                    f"lookback {N_LOOKBACK}  candle_time {human_time(row['t'])}  close {row['c']:.6g}"
                )
        except Exception:
            # ignore per-symbol errors to keep scanner running
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
