# breakout_scanner.py
# MEXC Spot USDT 4H Breakout Scanner — fresh signals, strict close > max(high[10 or 15])
# Notes:
# - بدون فیلتر سایه پایینی  bottom wick کاملا حذف شد
# - گزینه شکست: either یا both  (پیش‌فرض either = عبور از سقف 10 یا 15)
# - قانون untouched: next_low (پیش‌فرض) یا all_low یا none
# - تازه بودن سیگنال با --max-candles-ago  مثلا 1 یعنی آخرین کندل بسته‌شده
# - جلوگیری از 429 با تاخیر سبک و کانکارنسی محدود

import argparse, sys, time, math, json, os
from concurrent.futures import ThreadPoolExecutor
import requests

# ---------- HTTP helpers with light rate-limit ----------
class ThrottledSession(requests.Session):
    def __init__(self, delay=0.12):
        super().__init__()
        self.delay = delay
        self._last = 0.0

    def get(self, url, **kwargs):
        # very light throttle to avoid 429
        now = time.time()
        wait = self._last + self.delay - now
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        for attempt in range(3):
            r = super().get(url, timeout=20, **kwargs)
            if r.status_code == 429:
                time.sleep(0.7 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r

# ---------- math helpers ----------
def tick_floor(x, tick):
    if tick <= 0:
        return x
    return math.floor(x / tick) * tick

def body_ratio(o, h, l, c):
    rng = max(h - l, 1e-15)
    return abs(c - o) / rng

# ---------- exchange helpers ----------
def get_exchange_info(sess, api):
    try:
        r = sess.get(f"{api}/api/v3/exchangeInfo")
        data = r.json()
        return data
    except Exception:
        return None

def build_universe(sess, api, prefer_usdt=True):
    # Try exchangeInfo first
    info = get_exchange_info(sess, api)
    syms = []
    ticks = {}
    if info and "symbols" in info and info["symbols"]:
        for s in info["symbols"]:
            if s.get("status") != "TRADING":
                continue
            sym = s.get("symbol")
            if prefer_usdt and not sym.endswith("USDT"):
                continue
            # spot only if field exists
            if s.get("permissions") and "SPOT" not in s["permissions"]:
                continue
            # price tick
            tick = 1e-6
            for f in s.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    try:
                        tick = float(f.get("tickSize", "0.000001"))
                    except Exception:
                        pass
            syms.append(sym)
            ticks[sym] = tick
    if syms:
        print(f"# universe symbols={len(syms)}", file=sys.stderr)
        return syms, ticks

    # Fallback to ticker/price
    r = sess.get(f"{api}/api/v3/ticker/price")
    arr = r.json()
    for it in arr:
        sym = it.get("symbol")
        if prefer_usdt and not sym.endswith("USDT"):
            continue
        syms.append(sym)
        ticks[sym] = 1e-6
    print("# exchangeInfo returned zero symbols for USDT, falling back", file=sys.stderr)
    print(f"# universe source=ticker/price symbols={len(syms)}", file=sys.stderr)
    return syms, ticks

def get_klines(sess, api, symbol, interval, limit):
    r = sess.get(f"{api}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    return r.json()

# ---------- logic ----------
def strict_breakout_above(highs, idx, lookbacks, close_, tick, mode="either"):
    # previous highs do NOT include current candle
    prev_max = {n: max(highs[idx - n: idx]) for n in lookbacks}
    if mode == "both":
        bench = max(prev_max.values())
        ok = close_ >= tick_floor(bench + tick, tick)
        n_used = max(lookbacks)
        return ok, n_used, bench
    # either: pass on first satisfied n (prefer the shorter 10)
    for n in sorted(lookbacks):
        bench = prev_max[n]
        if close_ >= tick_floor(bench + tick, tick):
            return True, n, bench
    return False, 0, 0.0

def check_untouched_lows(lows, sig_idx, mode):
    if mode == "none":
        return True
    if sig_idx + 1 >= len(lows):
        return True
    sig_low = lows[sig_idx]
    if mode == "next_low":
        return lows[sig_idx + 1] > sig_low
    # all_low
    return min(lows[sig_idx + 1 : ]) > sig_low

def target_not_hit(highs, sig_idx, close_price, target_pct):
    if target_pct <= 0:
        return True
    tgt = close_price * (1.0 + target_pct)
    future_high = max(highs[sig_idx + 1 : ]) if sig_idx + 1 < len(highs) else -1
    return future_high < tgt

def scan_symbol(sess, api, rec, interval, window, lookbacks, min_body, breakout_mode, untouched, max_candles_ago, target_pct):
    sym, tick = rec
    limit = window + max(lookbacks) + 5
    arr = get_klines(sess, api, sym, interval, limit)
    if not arr or len(arr) < max(lookbacks) + 2:
        return []

    O, H, L, C, T = [], [], [], [], []
    for k in arr:
        O.append(float(k[1])); H.append(float(k[2])); L.append(float(k[3])); C.append(float(k[4])); T.append(int(k[0]))

    # scan last `window` candles (closed ones, ignore the forming last)
    last_idx = len(C) - 1  # last closed
    start = max(max(lookbacks), last_idx - window + 1)

    rows = []
    for i in range(start, last_idx + 1):
        # body filter
        if body_ratio(O[i], H[i], L[i], C[i]) < min_body:
            continue

        ok, n_used, bench = strict_breakout_above(H, i, lookbacks, C[i], tick, mode=breakout_mode)
        if not ok:
            continue

        # untouched rule on lows relative to signal candle
        if not check_untouched_lows(L, i, untouched):
            continue

        # fresh filter
        candles_ago = last_idx - i
        if candles_ago > max_candles_ago:
            continue

        # target not hit after signal
        if not target_not_hit(H, i, C[i], target_pct):
            continue

        # compose output
        ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(T[i] // 1000))
        row = [
            sym,
            ts,
            f"{C[i]}",
            f"{round(body_ratio(O[i], H[i], L[i], C[i]), 2)}",
            f"{n_used}",
            f"{H[i]}",
            f"{L[i]}",
            # bottom_wick_ticks kept only as info, no filtering
            str(int(round((min(C[i], O[i]) - L[i]) / max(tick, 1e-12)))),
            f"{tick:g}",
            str(candles_ago),
        ]
        rows.append(",".join(row))
    return rows

def read_manual_symbols(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s:
                out.append(s)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="https://api.mexc.com", help="MEXC API base. alt: https://www.mexc.com")
    p.add_argument("--interval", default="4h")
    p.add_argument("--window", type=int, default=45)
    p.add_argument("--min-body", type=float, default=0.70, help="full-body ratio threshold")
    p.add_argument("--max-candles-ago", type=int, default=180, help="freshness filter. 1 = only last closed candle")
    p.add_argument("--lookbacks", default="10,15", help="comma list")
    p.add_argument("--breakout-mode", choices=["either", "both"], default="either")
    p.add_argument("--untouched", choices=["next_low", "all_low", "none"], default="next_low")
    p.add_argument("--target-pct", type=float, default=0.0, help="e.g. 0.05 = 5 percent target must NOT be hit after signal")
    p.add_argument("--symbols-file", default="", help="optional manual list to scan")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    lookbacks = sorted({int(x) for x in args.lookbacks.split(",") if x.strip()})

    sess = ThrottledSession(delay=0.12)
    sess.headers.update({"User-Agent": "BreakOutScanner/1.0"})

    # universe build
    manual = read_manual_symbols(args.symbols_file) if args.symbols_file else []
    if manual:
        syms = manual
        ticks = {s: 1e-6 for s in syms}
        print(f"# universe symbols(manual)={len(syms)}", file=sys.stderr)
    else:
        syms, ticks = build_universe(sess, args.api, prefer_usdt=True)

    # print header
    print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")

    # scan
    pairs = [(s, ticks.get(s, 1e-6)) for s in syms]
    rows_all = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        for chunk in ex.map(lambda rec: scan_symbol(sess, args.api, rec, args.interval, args.window,
                                                    lookbacks, args.min_body, args.breakout_mode,
                                                    args.untouched, args.max_candles_ago, args.target_pct),
                            pairs, chunksize=10):
            if chunk:
                for r in chunk:
                    print(r)

if __name__ == "__main__":
    main()
