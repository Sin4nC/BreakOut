# Breakout scanner for MEXC USDT pairs
# Looks for the first full body candle that breaks above the highs of the last N candles
# pip install -r requirements.txt

import requests
import pandas as pd
import concurrent.futures as fut

BASE = "https://api.mexc.com"
USDT_ONLY = True
INTERVALS = ["4h", "1d", "1w"]        # if these fail the code will try Hour4 Day1 Week1
N_LOOKBACK = 12                       # set 10..15 as you like
BODY_RATIO_MIN = 0.6                  # full body threshold 0.6 means body >= 60% of candle range
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
        # skip leveraged tokens
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

def is_full_body(row):
    rng = row["h"] - row["l"]
    if rng <= 0:
        return False
    body = abs(row["c"] - row["o"])
    return (body / rng) >= BODY_RATIO_MIN

def breakout_signal(df):
    """
    Returns True if the last closed candle is the first full body breakout
    above the highs of the previous N_LOOKBACK candles
    """
    if df is None or len(df) < N_LOOKBACK + 2:
        return False
    i = len(df) - 1                # last closed candle
    last = df.iloc[i]
    prev = df.iloc[i - 1]

    # highs of prior N candles excluding current
    prior_high = df["h"].iloc[i - N_LOOKBACK : i].max()
    # same window for the previous candle to ensure this is the first breakout
    prior_high_prev = df["h"].iloc[i - N_LOOKBACK - 1 : i - 1].max()

    cond_break = (last["o"] > prior_high) and (last["c"] > prior_high)
    cond_body  = is_full_body(last)

    prev_was_break = (prev["o"] > prior_high_prev) and (prev["c"] > prior_high_prev) and is_full_body(prev)

    return cond_break and cond_body and (not prev_was_break)

def scan_symbol(sym):
    alerts = []
    for itv in INTERVALS:
        try:
            df = fetch_klines(sym, itv)
            if breakout_signal(df):
                last_close = df.iloc[-1]["c"]
                alerts.append(f"{sym}  {itv}  ALERT  FIRST FULL BODY BREAK  above last {N_LOOKBACK} highs  close {last_close:.6g}")
        except Exception:
            # ignore per symbol errors to keep the scan running
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
