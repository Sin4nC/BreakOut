# Pump strategy scanner for MEXC USDT spot pairs
# Detects green full-body candles whose CLOSE breaks above the highs of the last N candles (N in {15,20})
# Reports any FIRST breakout candles that occurred within the last SEARCH_WINDOW candles

import requests
import pandas as pd
import concurrent.futures as fut
from datetime import datetime, timezone

BASE = "https://api.mexc.com"
USDT_ONLY = True

# Timeframes to scan
INTERVALS = ["4h", "1d", "1w"]

# Lookback choices for your rule
N_LIST = [15, 20]          # can change to [20] if میخواهی فقط 20 باشد

# How far back to search for recent signals
SEARCH_WINDOW = 60

# Full-body thresholds per interval
BODY_RATIO_BY_INTERVAL = {"4h": 0.7, "1d": 0.6, "1w": 0.6}

# Request settings
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
        # skip leveraged tokens like 3L/3S/5L/5S
        if any(x in sym for x in ["3L", "3S", "5L", "5S"]):
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

def body_ratio(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return 0.0
    return abs(c - o) / rng

def is_green_full_body(row, min_ratio):
    o, h, l, c = row["o"], row["h"], row["l"], row["c"]
    return (c > o) and (body_ratio(o, h, l, c) >= min_ratio)

def prior_high(df, i, n):
    return df["h"].iloc[i - n : i].max()

def find_pump_breakouts(df, interval):
    """
    برمی‌گرداند لیستی از سیگنال‌ها به صورت [(i, n_used)] که
    i ایندکس کندلی است که:
      - سبز و فول‌بادی است
      - CLOSE بالاتر از سقف n کندل قبلی بسته شده
      - کندل قبلی همین شرط را نداشته  یعنی 'اولین' بریک‌اوت
    """
    if df is None:
        return []
    min_ratio = BODY_RATIO_BY_INTERVAL.get(interval, 0.6)
    nmax = max(N_LIST)
    if len(df) < nmax + 2:
        return []
    out = []
    for i in range(nmax + 1, len(df)):
        row = df.iloc[i]
        if not is_green_full_body(row, min_ratio):
            continue
        # بررسی برای هر N در لیست
        fired_with_any_n = None
        for n in N_LIST:
            ph_now = prior_high(df, i, n)
            ph_prev = prior_high(df, i - 1, n)
            c_now, o_now = row["c"], row["o"]
            # شرط اصلی: کلوز بالای سقف N کندل قبلی
            cond_now = c_now > ph_now
            # شرط 'اولین' بودن: کندل قبلی همین شرط را نداشته
            prev_row = df.iloc[i - 1]
            cond_prev = (prev_row["c"] > ph_prev) and is_green_full_body(prev_row, min_ratio)
            if cond_now and (not cond_prev):
                fired_with_any_n = n
                break
        if fired_with_any_n is not None:
            out.append((i, fired_with_any_n))
    return out

def human_time(ms):
    return datetime.fromtimestamp(int(ms)//1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def scan_symbol(sym):
    alerts = []
    for itv in INTERVALS:
        try:
            df = fetch_klines(sym, itv)
            hits = find_pump_breakouts(df, itv)
            if not hits:
                continue
            last_idx = len(df) - 1
            for i, n_used in hits:
                # فقط سیگنال‌هایی که در بازه جستجو هستند را گزارش بده
                if i >= last_idx - SEARCH_WINDOW + 1:
                    row = df.iloc[i]
                    candles_ago = last_idx - i
                    br = body_ratio(row["o"], row["h"], row["l"], row["c"])
                    alerts.append(
                        f"{sym} {itv} PUMP n={n_used} candles_ago={candles_ago} "
                        f"time {human_time(row['t'])} close {row['c']:.6g} body {br:.2f}"
                    )
        except Exception:
            # swallow per-symbol errors to keep the scan running
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
