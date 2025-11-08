#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEXC USDT spot breakout scanner

Defaults for Forz4crypto
- timeframe: 4h
- universe: همه جفتهای اسپات USDT از exchangeInfo
- CANDLE_WINDOW=180  تقریبا یک ماه 4h
- MAX_CANDLES_AGO=1  فقط سیگنال تازه
- قوانین هسته
  1) کندل سبز full-body با body_ratio >= 0.70
  2) close > max(high) از 15 یا 20 کندل قبل  یکی کافیست  با quantization روی tick
  3) bottom wick ≤ 1 tick
  4) untouched rule پیشفرض next_low  یعنی فقط کندل بعدی نباید low سیگنال را لمس کند
- HIPPOPUSDT و AIAUSDT هرگز حذف دستی ندارند
- --watch برای دیباگ چند سیمبل
خروجی TSV: symbol, signal_utc, close, body_ratio, lookbackN, high, low,
bottom_wick_ticks, tick_size, candles_ago
"""

import os
import sys
import math
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List

import requests

API_BASE = "https://api.mexc.com/api/v3"

# ---------- utils ----------

def ts_ms_to_iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def floor_to_tick(value: float, tick: float) -> float:
    if not tick or tick <= 0:
        return value
    return math.floor(value / tick) * tick

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def http_get(url: str, params: dict = None, timeout: int = 25):
    last_err = None
    for i in range(4):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_err = RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
        except Exception as e:
            last_err = e
        time.sleep(0.7 * (i + 1))
    if last_err:
        raise last_err

# ---------- data ----------

def get_exchange_info() -> Dict[str, dict]:
    """symbol -> {tick_size, status} فقط USDT اسپات و TRADING"""
    data = http_get(f"{API_BASE}/exchangeInfo")
    out = {}
    for s in data.get("symbols", []):
        sym = s.get("symbol", "")
        quote = s.get("quoteAsset", "")
        status = s.get("status", "TRADING")
        if quote != "USDT" or status != "TRADING":
            continue
        tick = None
        for f in s.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick = safe_float(f.get("tickSize"))
        if not tick:
            tick = 0.00000001
        out[sym] = {"tick_size": tick, "status": status}
    return out

def get_klines(symbol: str, interval: str, limit: int) -> List[dict]:
    params = {"symbol": symbol, "interval": interval, "limit": min(max(200, limit), 1000)}
    data = http_get(f"{API_BASE}/klines", params=params)
    candles = []
    for row in data:
        candles.append(
            {
                "open_time": int(row[0]),
                "open": safe_float(row[1]),
                "high": safe_float(row[2]),
                "low": safe_float(row[3]),
                "close": safe_float(row[4]),
            }
        )
    return candles

# ---------- logic ----------

def body_ratio(c: dict) -> float:
    rng = c["high"] - c["low"]
    if rng <= 0:
        return 0.0
    return abs(c["close"] - c["open"]) / rng

def is_green_full_body(c: dict, min_ratio: float) -> bool:
    return c["close"] > c["open"] and body_ratio(c) >= min_ratio

def bottom_wick_ticks(c: dict, tick: float) -> int:
    bw = min(c["open"], c["close"]) - c["low"]
    if tick <= 0:
        return 0 if bw <= 0 else 1
    return int(round(bw / tick))

def breaks_previous_highs(candles: List[dict], i: int, lookback: int, tick: float) -> bool:
    if i - lookback < 0:
        return False
    prev_high = max(c["high"] for c in candles[i - lookback : i])
    thr = floor_to_tick(prev_high, tick)
    return candles[i]["close"] > thr

def untouched_next_low(candles: List[dict], i: int) -> bool:
    if i + 1 >= len(candles):
        return False
    return candles[i + 1]["low"] > candles[i]["low"]

def untouched_all(candles: List[dict], i: int) -> bool:
    base_low = candles[i]["low"]
    for j in range(i + 1, len(candles)):
        if candles[j]["low"] <= base_low:
            return False
    return True

def scan_symbol(
    symbol: str,
    klines: List[dict],
    tick_size: float,
    max_candles_ago: int,
    window: int,
    min_body_ratio: float,
    untouched_mode: str,
):
    hits = []
    if len(klines) < 25:
        return hits
    last_idx = len(klines) - 1
    start = max(0, last_idx - window + 1)

    for i in range(last_idx, start - 1, -1):
        c = klines[i]
        if not is_green_full_body(c, min_body_ratio):
            continue
        if bottom_wick_ticks(c, tick_size) > 1:
            continue

        lb_used = None
        for N in (15, 20):
            if breaks_previous_highs(klines, i, N, tick_size):
                lb_used = N
                break
        if lb_used is None:
            continue

        ok = untouched_next_low(klines, i) if untouched_mode == "next" else untouched_all(klines, i)
        if not ok:
            continue

        candles_ago = last_idx - i
        if candles_ago > max_candles_ago:
            continue

        hits.append(
            {
                "symbol": symbol,
                "signal_utc": ts_ms_to_iso_utc(c["open_time"]),
                "close": c["close"],
                "body_ratio": round(body_ratio(c), 4),
                "lookbackN": lb_used,
                "high": c["high"],
                "low": c["low"],
                "bottom_wick_ticks": bottom_wick_ticks(c, tick_size),
                "tick_size": tick_size,
                "candles_ago": candles_ago,
            }
        )
    return hits

# ---------- cli ----------

def parse_args():
    p = argparse.ArgumentParser(description="MEXC Breakout Scanner")
    p.add_argument("--timeframe", default="4h", help="kline interval eg 4h")
    p.add_argument("--window", type=int, default=int(os.environ.get("CANDLE_WINDOW", "180")))
    p.add_argument("--max-candles-ago", type=int, default=int(os.environ.get("MAX_CANDLES_AGO", "1")))
    p.add_argument("--min-body", type=float, default=0.70)
    p.add_argument("--watch", default="", help="CSV eg BTCUSDT,ETHUSDT")
    p.add_argument(
        "--untouched",
        choices=["next", "all"],
        default=os.environ.get("UNTOUCHED", "next"),
        help="next = فقط کندل بعدی  all = همه کندلهای بعدی",
    )
    return p.parse_args()

def main():
    args = parse_args()

    info = get_exchange_info()
    universe = sorted([s for s in info if s.endswith("USDT")])

    if args.watch.strip():
        watch = {s.strip().upper() for s in args.watch.split(",") if s.strip()}
        # نگه داشتن فقط موجودها  و اضافه کردن غایبها برای دیباگ
        uni = [s for s in universe if s in watch]
        for s in watch:
            if s not in info:
                info[s] = {"tick_size": 0.00000001, "status": "UNKNOWN"}
                uni.append(s)
        universe = uni

    print("symbol\tsignal_utc\tclose\tbody_ratio\tlookbackN\thigh\tlow\tbottom_wick_ticks\ttick_size\tcandles_ago")
    for sym in universe:
        try:
            kl = get_klines(sym, args.timeframe, limit=args.window + 25)
            hits = scan_symbol(
                sym,
                kl,
                info[sym]["tick_size"],
                max_candles_ago=args.max_candles_ago,
                window=args.window,
                min_body_ratio=args.min_body,
                untouched_mode=args.untouched,
            )
            for h in hits:
                print(
                    f"{h['symbol']}\t{h['signal_utc']}\t{h['close']}\t{h['body_ratio']}\t"
                    f"{h['lookbackN']}\t{h['high']}\t{h['low']}\t{h['bottom_wick_ticks']}\t"
                    f"{h['tick_size']}\t{h['candles_ago']}"
                )
        except Exception as e:
            print(f"# ERROR {sym}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
