#!/usr/bin/env python3
# breakout_scanner.py
import argparse, time, math, sys, csv, concurrent.futures as cf
from typing import List, Dict, Any, Tuple, Optional
import requests

# -------- HTTP helpers with retry & backoff --------
def http_get(session: requests.Session, url: str, **params):
    backoff = 0.6
    for i in range(6):
        r = session.get(url, params=params, timeout=20)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff *= 1.7
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r

# -------- Exchange metadata --------
def get_exchange_info_symbols(session, api: str) -> Tuple[List[str], Dict[str, float]]:
    syms, ticks = [], {}
    try:
        data = http_get(session, f"{api}/api/v3/exchangeInfo").json()
        for s in data.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            if not s.get("symbol", "").endswith("USDT"):
                continue
            syms.append(s["symbol"])
            tick = 1e-6
            for f in s.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    tick = float(f.get("tickSize", tick))
            ticks[s["symbol"]] = tick
    except Exception:
        pass
    return syms, ticks

def get_symbols_from_ticker(session, api: str) -> List[str]:
    data = http_get(session, f"{api}/api/v3/ticker/price").json()
    return [d["symbol"] for d in data if d["symbol"].endswith("USDT")]

def build_universe(session, api: str, symbols_file: Optional[str]) -> Tuple[List[str], Dict[str, float]]:
    ticks: Dict[str, float] = {}
    if symbols_file:
        with open(symbols_file, "r", encoding="utf-8") as f:
            syms = [ln.strip().upper() for ln in f if ln.strip()]
    else:
        syms, ticks = get_exchange_info_symbols(session, api)
        if not syms:
            syms = get_symbols_from_ticker(session, api)
    # تضمین عدم حذف اختیاری
    for must in ["HIPPOPUSDT", "AIAUSDT"]:
        if must not in syms:
            syms.append(must)
    return sorted(set(syms)), ticks

# -------- Market data --------
def get_klines(session, api: str, symbol: str, interval: str, limit: int) -> List[List[Any]]:
    return http_get(session, f"{api}/api/v3/klines", symbol=symbol, interval=interval, limit=limit).json()

# -------- Math --------
def body_ratio(o: float, h: float, l: float, c: float) -> float:
    rng = max(h - l, 0.0)
    if rng <= 0:
        return 0.0
    return abs(c - o) / rng

def is_green(o: float, c: float) -> bool:
    return c > o

# strict breakout above BOTH 10 and 15 highs
def strict_breakout_above(highs: List[float], idx: int, lookbacks: List[int], close_: float, tick: float) -> Tuple[bool, int, float]:
    # idx = index of signal candle
    maxes = []
    for n in lookbacks:
        if idx - n < 0:
            return False, 0, 0.0
        prev_max = max(highs[idx - n: idx])  # exclude current candle
        maxes.append(prev_max)
    bench = max(maxes)  # must clear the bigger of 10 or 15
    return close_ >= bench + tick, max(lookbacks), bench

# next candle must not touch signal low
def untouched_next_low(lows: List[float], idx: int, tick: float) -> bool:
    if idx + 1 >= len(lows):
        return False  # برای قانون next باید کندل بعدی موجود باشد
    return lows[idx + 1] > lows[idx] + tick

# did target get hit after signal
def target_hit_after(close_series: List[float], highs: List[float], idx: int, target_pct: float) -> bool:
    need = close_series[idx] * (1.0 + target_pct)
    for j in range(idx + 1, len(highs)):
        if highs[j] >= need:
            return True
    return False

# -------- Scanner per symbol --------
def scan_symbol(session, api: str, symbol: str, interval: str, window: int, lookbacks: List[int],
                min_body: float, max_candles_ago: int, target_pct: Optional[float],
                untouched_rule: str, tick_map: Dict[str, float]) -> Optional[Tuple]:
    limit = window + max(lookbacks) + 5
    try:
        raw = get_klines(session, api, symbol, interval, limit)
    except Exception:
        return None

    if len(raw) < max(lookbacks) + 2:
        return None

    O, H, L, C, T = [], [], [], [], []
    for r in raw:
        o, h, l, c = map(float, r[:4])
        O.append(o); H.append(h); L.append(l); C.append(c); T.append(int(r[0]))

    tick = tick_map.get(symbol, 1e-6)

    # در این نسخه فقط تازه ترین سیگنال معتبر هر نماد را میگیریم
    best = None
    # فقط در بازه آخر window کندل را ارزیابی کن
    start = max(len(C) - window, max(lookbacks))
    end = len(C) - 2  # حداقل یک کندل بعدی برای قانون next_low
    for i in range(end, start - 1, -1):  # از آخر به اول برای تازه تر
        candles_ago = len(C) - 1 - i
        if candles_ago > max_candles_ago:
            continue
        if not is_green(O[i], C[i]):
            continue
        br = body_ratio(O[i], H[i], L[i], C[i])
        if br < min_body:
            continue

        ok, n_used, bench = strict_breakout_above(H, i, lookbacks, C[i], tick)
        if not ok:
            continue

        if untouched_rule == "next_low" and not untouched_next_low(L, i, tick):
            continue

        if target_pct is not None and target_hit_after(C, H, i, target_pct):
            continue

        best = (symbol, i, candles_ago, C[i], br, n_used, H[i], L[i], tick)
        break

    if not best:
        return None
    sym, i, candles_ago, close_, br, n_used, hi, lo, tick = best
    ts = time.gmtime(T[i] // 1000)
    tstr = time.strftime("%Y-%m-%d %H:%M UTC", ts)
    bottom_wick_ticks = max(int(round((min(O[i], C[i]) - lo) / tick)), 0)
    return (sym, tstr, close_, round(br, 2), n_used, hi, lo, bottom_wick_ticks, tick, candles_ago)

# -------- Main --------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="https://api.mexc.com")
    p.add_argument("--interval", default="4h")
    p.add_argument("--window", type=int, default=45)
    p.add_argument("--lookbacks", default="10,15")
    p.add_argument("--min-body", type=float, default=0.70)
    p.add_argument("--untouched", choices=["none", "next_low"], default="next_low")
    p.add_argument("--max-candles-ago", type=int, default=1)
    p.add_argument("--target-pct", type=float, default=None)
    p.add_argument("--symbols-file", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    lookbacks = sorted({int(x) for x in args.lookbacks.split(",") if x.strip()})
    sess = requests.Session()

    syms, ticks = build_universe(sess, args.api, args.symbols_file)
    if not syms:
        print("# universe symbols=0")
        return

    print("# universe symbols=%d" % len(syms))
    print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")

    out: List[Tuple] = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(scan_symbol, sess, args.api, s, args.interval, args.window,
                             lookbacks, args.min_body, args.max_candles_ago,
                             args.target_pct, args.untouched, ticks) for s in syms]
        for fut in cf.as_completed(futures):
            row = fut.result()
            if row:
                out.append(row)

    # یکتا و فقط تازه ترین
    latest: Dict[str, Tuple] = {}
    for r in out:
        sym = r[0]
        if sym not in latest or r[-1] < latest[sym][-1]:
            latest[sym] = r

    for r in sorted(latest.values(), key=lambda x: (x[-1], x[0])):
        print(f"{r[0]},{r[1]},{r[2]},{r[3]:.2f},{r[4]},{r[5]},{r[6]},{r[7]},{r[8]},{r[9]}")

if __name__ == "__main__":
    main()
