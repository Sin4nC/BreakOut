# breakout_scanner.py
# MEXC spot USDT breakout scanner — 4H by default
# Finds green full-body breakouts where close > max(high of prev N) with N in lookbacks
# Window: scan last W candles backward from "now" per symbol
# Untouched rule: default 'next_low'  options: none | next_low | all_low | all_highlow
# Target filter: if --target-pct given, exclude signals that have already hit target after the signal

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

API = "https://api.mexc.com"

def get_exchange_info(session: requests.Session) -> Dict:
    url = f"{API}/api/v3/exchangeInfo"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_symbols(ex_info: Dict, only_usdt_spot: bool = True) -> List[Dict]:
    syms = []
    for s in ex_info.get("symbols", []):
        symbol = s.get("symbol", "")
        status = s.get("status", "TRADING")
        quote = s.get("quoteAsset", "")
        perms = set(s.get("permissions", []))
        if only_usdt_spot:
            if quote != "USDT":
                continue
            # treat as spot if no permissions field present or contains SPOT
            if perms and "SPOT" not in perms:
                continue
        if status not in ("TRADING", "ENABLED"):
            continue
        # price tick
        tick = 1e-06
        for f in s.get("filters", []):
            if f.get("filterType") in ("PRICE_FILTER", "PRICE_FILTER_1"):
                ts = f.get("tickSize")
                if ts is not None:
                    try:
                        tick = float(ts)
                    except Exception:
                        pass
        syms.append({"symbol": symbol, "tick": tick})
    return syms

def get_klines(session: requests.Session, symbol: str, interval: str, limit: int) -> List[Tuple]:
    url = f"{API}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    for row in data:
        # row = [openTime, open, high, low, close, volume, closeTime, ...]
        o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
        ot = int(row[0]); ct = int(row[6])
        out.append((ot, o, h, l, c, ct))
    return out

def fmt_time_utc(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M UTC")

def body_ratio(o: float, h: float, l: float, c: float) -> float:
    tr = max(h - l, 1e-12)
    if c <= o:
        return 0.0
    return (c - o) / tr

def bottom_wick_ticks(o: float, l: float, tick: float, green: bool) -> int:
    if not green:
        return 0
    bw = max(min(o, float('inf')) - l, 0.0)
    return int(round(bw / max(tick, 1e-12)))

def close_above_prev_highs(candles: List[Tuple], idx: int, lookback: int, close_value: float) -> bool:
    start = max(0, idx - lookback)
    prev_highs = [c[2] for c in candles[start:idx]]
    return len(prev_highs) >= lookback and close_value > max(prev_highs)

def untouched_ok(candles: List[Tuple], idx: int, rule: str) -> bool:
    # idx is signal candle index
    sig_h = candles[idx][2]
    sig_l = candles[idx][3]
    if rule == "none":
        return True
    if idx + 1 >= len(candles):
        # no next candle yet, treat as pass
        return True
    if rule == "next_low":
        nxt = candles[idx + 1]
        return nxt[3] > sig_l
    if rule == "all_low":
        for k in range(idx + 1, len(candles)):
            if candles[k][3] <= sig_l:
                return False
        return True
    if rule == "all_highlow":
        for k in range(idx + 1, len(candles)):
            if candles[k][3] <= sig_l or candles[k][2] >= sig_h:
                return False
        return True
    return True

def target_not_hit(candles: List[Tuple], idx: int, target_pct: float, from_price: Optional[float] = None) -> bool:
    if target_pct is None or target_pct <= 0:
        return True
    # target based on close of signal
    sig_close = candles[idx][4] if from_price is None else from_price
    tgt = sig_close * (1.0 + target_pct)
    # did any later candle reach target high
    for k in range(idx + 1, len(candles)):
        if candles[k][2] >= tgt:
            return False
    return True

def scan_symbol(session: requests.Session,
                sym: str,
                tick: float,
                interval: str,
                window: int,
                lookbacks: List[int],
                min_body: float,
                untouched_rule: str,
                max_candles_ago: int,
                target_pct: Optional[float],
                limit_pad: int = 5) -> List[str]:

    max_lb = max(lookbacks)
    # fetch enough to cover lookback plus window
    limit = window + max_lb + limit_pad
    kl = get_klines(session, sym, interval, limit)
    if len(kl) < max_lb + 2:
        return []

    out_lines = []
    last_idx = len(kl) - 1
    # scan last `window` candles: j ranges [last_idx - window + 1 .. last_idx]
    start_idx = max(0, last_idx - window + 1)
    for j in range(start_idx, len(kl)):
        ot, o, h, l, c, ct = kl[j]
        green = c > o
        if not green:
            continue
        br = body_ratio(o, h, l, c)
        if br < min_body:
            continue
        # breakout on any lookback
        n_used = None
        for n in lookbacks:
            if j - n < 0:
                continue
            if close_above_prev_highs(kl, j, n, c):
                n_used = n
                break
        if n_used is None:
            continue
        if not untouched_ok(kl, j, untouched_rule):
            continue
        if target_pct is not None and target_pct > 0:
            if not target_not_hit(kl, j, target_pct):
                continue
        candles_ago = last_idx - j
        if candles_ago > max_candles_ago:
            continue

        bw_ticks = bottom_wick_ticks(o, l, tick, green=True)
        line = (
            f"{sym},{fmt_time_utc(ct)},{c:.15g},{br:.2f},{n_used},"
            f"{h:.15g},{l:.15g},{bw_ticks},{tick:.15g},{candles_ago}"
        )
        out_lines.append(line)
    return out_lines

def read_symbols_file(path: str) -> List[str]:
    symbols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            symbols.append(t)
    return symbols

def main():
    ap = argparse.ArgumentParser(description="MEXC 4H breakout scanner")
    ap.add_argument("--interval", default="4h", help="kline interval default 4h")
    ap.add_argument("--window", type=int, default=45, help="how many recent candles to scan")
    ap.add_argument("--lookbacks", default="10,15", help="comma list of lookbacks any-pass e.g. 10,15")
    ap.add_argument("--min-body", type=float, default=0.70, help="min body ratio 0..1")
    ap.add_argument("--untouched", default="next_low", choices=["none", "next_low", "all_low", "all_highlow"],
                    help="untouched rule default next_low")
    ap.add_argument("--max-candles-ago", type=int, default=45, help="filter to fresh signals candles_ago <= this")
    ap.add_argument("--target-pct", type=float, default=None, help="exclude signals that already hit target pct after signal")
    ap.add_argument("--symbols-file", default=None, help="optional path with one symbol per line")
    ap.add_argument("--max-workers", type=int, default=10, help="concurrency for klines fetch")
    ap.add_argument("--timeout-s", type=int, default=30, help="http timeout seconds")

    args = ap.parse_args()
    lookbacks = sorted({int(x) for x in args.lookbacks.split(",") if x.strip()})
    session = requests.Session()

    # universe
    if args.symbols_file:
        base_symbols = [{"symbol": s, "tick": 1e-06} for s in read_symbols_file(args.symbols_file)]
    else:
        ex = get_exchange_info(session)
        base_symbols = parse_symbols(ex, only_usdt_spot=True)

    symbols = base_symbols
    sys.stderr.write(f"# universe symbols={len(symbols)}\n")
    sys.stderr.flush()

    header = "symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago"
    print(header)

    # worker
    def task(rec):
        sym = rec["symbol"]
        tick = rec["tick"]
        tries = 0
        while True:
            try:
                return scan_symbol(
                    session=session,
                    sym=sym,
                    tick=tick,
                    interval=args.interval,
                    window=args.window,
                    lookbacks=lookbacks,
                    min_body=args.min_body,
                    untouched_rule=args.untouched,
                    max_candles_ago=args.max_candles_ago,
                    target_pct=args.target_pct,
                )
            except requests.HTTPError as e:
                tries += 1
                if e.response is not None and e.response.status_code in (418, 429, 451, 500, 502, 503, 504):
                    time.sleep(min(1 + tries * 0.5, 5))
                    continue
                # log and give up
                sys.stderr.write(f"! {sym} http {e}\n")
                return []
            except requests.RequestException as e:
                tries += 1
                if tries <= 3:
                    time.sleep(min(1 + tries * 0.5, 3))
                    continue
                sys.stderr.write(f"! {sym} net {e}\n")
                return []
            except Exception as e:
                sys.stderr.write(f"! {sym} err {e}\n")
                return []

    hits = 0
    scanned = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for lines in ex.map(task, symbols, chunksize=20):
            scanned += 1
            if not lines:
                continue
            hits += len(lines)
            for ln in lines:
                print(ln)

    sys.stderr.write(f"# scanned={scanned} hits={hits} window={args.window} max_candles_ago={args.max_candles_ago}\n")
    sys.stderr.flush()

if __name__ == "__main__":
    main()
