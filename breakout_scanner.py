# breakout_scanner.py
# Full replacement file — copy paste over your repo

import os
import sys
import time
import math
import json
import argparse
from typing import Dict, List, Tuple, Optional

import requests
from datetime import datetime, timezone

MEXC_BASE = "https://api.mexc.com"

# -----------------------------
# Helpers
# -----------------------------
def env_default(key: str, fallback: Optional[str] = None):
    v = os.environ.get(key)
    return v if v is not None else fallback

def parse_args():
    p = argparse.ArgumentParser(description="MEXC USDT Breakout Scanner")

    # core
    p.add_argument("--timeframe", default=env_default("TIMEFRAME", "4h"))
    p.add_argument("--candle_window", type=int, default=int(env_default("CANDLE_WINDOW", "180")))
    p.add_argument("--max_candles_ago", type=int, default=int(env_default("MAX_CANDLES_AGO", "1")))
    p.add_argument("--min_body", type=float, default=float(env_default("MIN_BODY", "0.70")))
    p.add_argument("--lookbacks", default=env_default("LOOKBACKS", "15,20"))
    p.add_argument("--untouched", choices=["next_low", "next_highlow", "all", "none"],
                   default=env_default("UNTOUCHED", "next_low"))

    # universe control
    p.add_argument("--watch", default=env_default("WATCH", "").strip())
    p.add_argument("--universe", default=env_default("UNIVERSE", "mexc_usdt_spot"))

    # misc
    p.add_argument("--timeout", type=int, default=int(env_default("HTTP_TIMEOUT", "15")))
    p.add_argument("--retries", type=int, default=int(env_default("HTTP_RETRIES", "3")))
    p.add_argument("--debug", action="store_true" if env_default("DEBUG", "0") == "1" else "store_false")

    return p.parse_args()

def http_get(url: str, params=None, timeout=15, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            # small backoff on non-200 as well
        except Exception:
            pass
        time.sleep(0.5 * (i + 1))
    raise RuntimeError(f"HTTP GET failed for {url} params={params}")

def load_exchange_info(timeout=15, retries=3):
    data = http_get(f"{MEXC_BASE}/api/v3/exchangeInfo", timeout=timeout, retries=retries)
    symbols = data.get("symbols", [])
    result = {}
    for s in symbols:
        try:
            if s.get("status") != "TRADING":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            # spot only if permissions present
            perms = s.get("permissions") or s.get("permissionsList") or []
            if perms and "SPOT" not in perms:
                continue

            symbol = s["symbol"]

            tick_size = None
            step_size = None
            for f in s.get("filters", []):
                n = f.get("filterType") or f.get("filter_type") or ""
                if n == "PRICE_FILTER":
                    tick_size = float(f.get("tickSize") or f.get("tick_size") or "0.0001")
                if n == "LOT_SIZE":
                    step_size = float(f.get("stepSize") or f.get("step_size") or "1")

            if tick_size is None:
                # safe fallback
                tick_size = 0.0001
            if step_size is None:
                step_size = 1.0

            result[symbol] = {
                "base": s.get("baseAsset"),
                "quote": s.get("quoteAsset"),
                "tick_size": tick_size,
                "step_size": step_size,
            }
        except Exception:
            # ignore malformed entries
            continue
    return result

def klines(symbol: str, interval: str, limit: int, timeout=15, retries=3):
    # MEXC Klines format like Binance
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = http_get(f"{MEXC_BASE}/api/v3/klines", params=params, timeout=timeout, retries=retries)
    out = []
    for row in data:
        # [ openTime, open, high, low, close, volume, closeTime, ... ]
        o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
        ot = int(row[0]); ct = int(row[6])
        out.append({
            "open": o, "high": h, "low": l, "close": c,
            "open_time": ot, "close_time": ct
        })
    return out

def tick_floor(x: float, tick: float) -> int:
    # stable floor to integer ticks
    return math.floor((x + 1e-12) / tick)

def utc_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

# -----------------------------
# Signal logic
# -----------------------------
def body_ratio(c):
    rng = c["high"] - c["low"]
    if rng <= 0:
        return 0.0
    return (c["close"] - c["open"]) / rng

def bottom_wick_ticks(c, tick):
    bw = min(c["open"], c["close"]) - c["low"]
    if bw < 0:
        return 0
    return tick_floor(bw, tick)

def close_above_prev_highs(candles: List[dict], i: int, tick: float, lookbacks: List[int]) -> Tuple[bool, Optional[int]]:
    c = candles[i]
    c_tick = tick_floor(c["close"], tick)
    for n in lookbacks:
        if i - n < 0:
            continue
        prev_high = max(candles[i - n:i], key=lambda x: x["high"])["high"]
        prev_tick = tick_floor(prev_high, tick)
        if c_tick > prev_tick:
            return True, n
    return False, None

def has_untouched(candles: List[dict], i: int, tick: float, mode: str) -> bool:
    if mode == "none":
        return True

    # need at least 1 subsequent candle
    if i >= len(candles) - 1:
        return False

    sig = candles[i]
    nextc = candles[i + 1]

    sig_low_tick = tick_floor(sig["low"], tick)
    sig_high_tick = tick_floor(sig["high"], tick)

    if mode == "next_low":
        return tick_floor(nextc["low"], tick) > sig_low_tick

    if mode == "next_highlow":
        return (tick_floor(nextc["low"], tick) > sig_low_tick) and (tick_floor(nextc["high"], tick) < sig_high_tick)

    if mode == "all":
        for k in range(i + 1, len(candles)):
            if tick_floor(candles[k]["low"], tick) <= sig_low_tick:
                return False
            if tick_floor(candles[k]["high"], tick) >= sig_high_tick:
                return False
        return True

    return True

def find_signals_for_symbol(sym: str,
                            info: Dict,
                            tf: str,
                            candle_window: int,
                            max_candles_ago: int,
                            min_body: float,
                            lookbacks: List[int],
                            untouched: str,
                            timeout: int,
                            retries: int,
                            debug: bool = False) -> List[Dict]:
    out = []
    data = klines(sym, tf, limit=candle_window + 30, timeout=timeout, retries=retries)
    if len(data) < max(lookbacks) + 2:
        return out

    tick = info["tick_size"]

    # iterate over recent candles as candidates
    for ago in range(0, max_candles_ago):
        i = len(data) - 1 - ago  # candidate index
        if i <= 0 or i >= len(data):
            continue

        c = data[i]
        if c["close"] <= c["open"]:
            if debug:
                print(f"[{sym}] skip idx {i} not green")
            continue

        br = body_ratio(c)
        if br < min_body:
            if debug:
                print(f"[{sym}] skip idx {i} body_ratio={br:.3f} < {min_body}")
            continue

        bwt = bottom_wick_ticks(c, tick)
        if bwt > 1:
            if debug:
                print(f"[{sym}] skip idx {i} bottom_wick_ticks={bwt} > 1")
            continue

        ok, used_lb = close_above_prev_highs(data, i, tick, lookbacks)
        if not ok:
            if debug:
                print(f"[{sym}] skip idx {i} close not above prev highs {lookbacks}")
            continue

        if not has_untouched(data, i, tick, untouched):
            if debug:
                print(f"[{sym}] skip idx {i} untouched rule failed mode={untouched}")
            continue

        out.append({
            "symbol": sym,
            "signal_utc": utc_from_ms(c["close_time"]),
            "close": c["close"],
            "body_ratio": br,
            "lookbackN": used_lb,
            "high": c["high"],
            "low": c["low"],
            "bottom_wick_ticks": bwt,
            "tick_size": tick,
            "candles_ago": ago
        })

    return out

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # lookbacks parse and ensure ascending unique ints
    lbs = sorted({int(x) for x in args.lookbacks.split(",") if x.strip()})

    ex = load_exchange_info(timeout=args.timeout, retries=args.retries)

    # universe
    if args.universe != "mexc_usdt_spot":
        print(f"# Unknown universe {args.universe}, defaulting to mexc_usdt_spot", file=sys.stderr)

    symbols = sorted(ex.keys())

    # optional watch filter
    if args.watch:
        watch = {s.strip().upper() for s in args.watch.split(",") if s.strip()}
        symbols = [s for s in symbols if s in watch]

    # HIPPOPUSDT and AIAUSDT must never be hard-excluded — no-op here

    # header
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago")

    for sym in symbols:
        try:
            sigs = find_signals_for_symbol(
                sym=sym,
                info=ex[sym],
                tf=args.timeframe,
                candle_window=args.candle_window,
                max_candles_ago=args.max_candles_ago,
                min_body=args.min_body,
                lookbacks=lbs,
                untouched=args.untouched,
                timeout=args.timeout,
                retries=args.retries,
                debug=args.debug and (not args.watch or sym in {s.strip().upper() for s in args.watch.split(",")})
            )
            for s in sigs:
                print("{symbol},{signal_utc},{close:.8f},{body_ratio:.3f},{lookbackN},{high:.8f},{low:.8f},{bottom_wick_ticks},{tick_size:.8f},{candles_ago}".format(**s))
        except Exception as e:
            # keep the run alive even if one symbol fails
            print(f"# error {sym} {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
