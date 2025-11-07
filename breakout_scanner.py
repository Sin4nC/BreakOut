# Breakout scanner for MEXC USDT spot pairs
# Rule: FIRST full-body candle whose CLOSE is above the max high of the last 20 candles
# Body ratio per interval: 4h=0.7  1d=0.6  1w=0.6

import requests
import pandas as pd
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"
USDT_ONLY = True

INTERVALS = ["4h", "1d", "1w"]   # for quick testing you can use ["4h"]

N_LOOKBACK = 20
SEARCH_WINDOW = 60

BODY_RATIO_BY_INTERVAL = {
    "4h": 0.7,
    "1d": 0.6,
    "1w": 0.6,
}

LIMIT = 300
MAX_WORKERS = 6

# optional focus and debug
WATCHLIST = []                  # e.g. ["AIAUSDT", "HIPPOUSDT"] to test only these
DEBUG_SYMBOLS = []              # e.g. ["AIAUSDT"] to print near-miss reasons

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

def body_ratio(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return 0.0
    return abs(c - o) / rng

def qualifies_close_break(df, i, min_ratio, n_lookback):
    o, h, l, c = df.iloc[i][["o","h","l","c"]]
    prior_high = df["h"].iloc[i - n_lookback : i].max()
    return (body_ratio(o, h, l, c) >= min_ratio) and (c > prior_high)

def breakout_index(df, interval, symbol=None):
    if df is None or len(df) < N_LOOKBACK + 2:
        return None
    min_ratio = BODY_RATIO_BY_INTERVAL.get(interval, 0.6)
    start = max(N_LOOKBACK + 1, len(df) - SEARCH_WINDOW)

    dbg = symbol in DEBUG_SYMBOLS

    # scan older -> newer to return the FIRST breakout in the window
    for i in range(start, len(df)):
        ok_now = qualifies_close_break(df, i,   min_ratio, N_LOOKBACK)
        ok_prev= qualifies_close_break(df, i-1, min_ratio, N_LOOKBACK)
        if dbg:
            if not ok_now:
                o,h,l,c = df.iloc[i][["o","h","l","c"]]
                ph = df["h"].iloc[i - N_LOOKBACK : i].max()
                print(f"DEBUG {symbol} {interval} i={i} body={body_ratio(o,h,l,c):.3f} "
                      f"need>={min_ratio} close={c:.6g} prior_high={ph:.6g}")
        if ok_now and not ok_prev:
            return i
    return None

def human_time(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_symbol(sym):
    alerts = []
    for itv in INTERVALS:
        try:
            df = fetch_klines(sym, itv)
            idx = breakout_index(df, itv, sym)
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
    symbols = WATCHLIST if WATCHLIST else get_symbols()
    hits = run_round(symbols)
    if hits:
        for h in hits:
            print(h)
    else:
        print("no signals")
