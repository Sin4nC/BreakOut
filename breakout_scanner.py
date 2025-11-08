#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import json
import time
from typing import List, Tuple, Dict
import requests
from datetime import datetime, timezone

# =========================
# Config via ENV (defaults)
# =========================
TIMEFRAME          = os.environ.get("TIMEFRAME", "4h").lower()
CANDLE_WINDOW      = int(os.environ.get("CANDLE_WINDOW", "180"))
MAX_CANDLES_AGO    = int(os.environ.get("MAX_CANDLES_AGO", "1"))   # fresh by default
MIN_BODY           = float(os.environ.get("MIN_BODY", "0.70"))
LOOKBACKS          = [int(x.strip()) for x in os.environ.get("LOOKBACKS", "15,20").split(",") if x.strip()]
BOTTOM_WICK_TICKS  = int(os.environ.get("BOTTOM_WICK_TICKS", "1"))
UNTOUCHED          = os.environ.get("UNTOUCHED", "next_low").lower()  # none | next_low | next_highlow | all
WATCH              = [x.strip().upper() for x in os.environ.get("WATCH", "").split(",") if x.strip()]
DEBUG              = int(os.environ.get("DEBUG", "0"))

# Constants
MEXC_BASE = "https://api.mexc.com"
INTERVAL  = "4h"  # scanner default per user
if TIMEFRAME != "4h":
    print(f"# WARNING: Only 4h is supported now, forcing 4h", file=sys.stderr)

# =========================
# Helpers
# =========================
def debug(msg: str):
    if DEBUG:
        print(msg, file=sys.stderr)

def iso_utc(ms: int) -> str:
    return datetime.utcfromtimestamp(ms/1000).replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

def floor_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return x
    return math.floor(x / tick) * tick

def get_exchange_info() -> Dict:
    url = f"{MEXC_BASE}/api/v3/exchangeInfo"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def list_mexc_spot_usdt_symbols() -> Dict[str, Dict]:
    """Return {symbol: {'tickSize': ..., 'status': ...}} filtered to USDT quote, spot, trading"""
    info = get_exchange_info()
    out = {}
    for s in info.get("symbols", []):
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        # Find tickSize from filters
        tick_size = None
        for f in s.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", "0.00000001"))
                break
        if not tick_size:
            tick_size = 0.00000001
        out[s["symbol"]] = {"tickSize": tick_size}
    return out

def fetch_klines(symbol: str, limit: int) -> List[List]:
    url = f"{MEXC_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def body_ratio(open_, high, low, close) -> float:
    rng = max(high - low, 1e-12)
    body = abs(close - open_)
    return body / rng

def bottom_wick_ticks(open_, low, close, tick_size) -> int:
    lower_body = min(open_, close)
    bw = max(lower_body - low, 0.0)
    if tick_size <= 0:
        return 0
    return int(round(bw / tick_size))

def passes_breakout(close, highs_window, tick_size) -> Tuple[bool, int]:
    """
    Returns (ok, lookbackN) where ok is True if close > max(highs of any lookback in LOOKBACKS) after tick-floor.
    """
    for N in LOOKBACKS:
        if len(highs_window) < N:
            continue
        ref = max(highs_window[-N:])
        ref_q = floor_to_tick(ref, tick_size)
        if close > ref_q:
            return True, N
    return False, 0

def check_untouched(rule: str, signal_idx: int, candles: List[List]) -> bool:
    """
    rule:
      - none: skip check
      - next_low: next candle's low must be > signal low
      - next_highlow: next candle's low > signal low AND next candle's high < signal high
      - all: for all subsequent candles up to last closed, lows > signal low AND highs < signal high
    """
    if rule == "none":
        return True
    if signal_idx >= len(candles) - 1:
        return True  # no next candle yet

    sig = candles[signal_idx]
    sig_high = float(sig[2])
    sig_low  = float(sig[3])

    nxt = candles[signal_idx + 1]
    next_low = float(nxt[3])
    next_high = float(nxt[2])

    if rule == "next_low":
        return next_low > sig_low
    if rule == "next_highlow":
        return (next_low > sig_low) and (next_high < sig_high)
    if rule == "all":
        for i in range(signal_idx + 1, len(candles)):
            c = candles[i]
            h = float(c[2]); l = float(c[3])
            if (l <= sig_low) or (h >= sig_high):
                return False
        return True
    return True

# =========================
# Main
# =========================
def main():
    # CSV header
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago")

    universe = list_mexc_spot_usdt_symbols()
    symbols = sorted(universe.keys())

    if WATCH:
        watch_set = set([s.upper() for s in WATCH])
        symbols = [s for s in symbols if s.upper() in watch_set]
        if DEBUG:
            debug(f"# WATCH mode active for {len(symbols)} symbols: {symbols}")

    # how many klines to request
    need = max(CANDLE_WINDOW, max(LOOKBACKS) + 5)
    limit = min(max(need, 50), 1000)

    for sym in symbols:
        tick = universe[sym]["tickSize"]
        try:
            kl = fetch_klines(sym, limit=limit)
        except Exception as e:
            debug(f"[{sym}] fetch_klines error: {e}")
            continue

        if len(kl) < 25:
            debug(f"[{sym}] not enough candles")
            continue

        # kline fields: [openTime, open, high, low, close, volume, closeTime, ...]
        # We only use closed candles (MEXC returns only closed for past intervals)
        # Choose candidate indices within last MAX_CANDLES_AGO candles (1 = only last closed candle)
        max_back = min(MAX_CANDLES_AGO, len(kl) - 1)
        # indices of candidate candles (from the end)
        cand_indices = list(range(len(kl) - max_back - 1, len(kl) - 1)) if MAX_CANDLES_AGO > 1 else [len(kl) - 2]

        # If MAX_CANDLES_AGO == 1 → just last closed candle index = -2 (because -1 is partially next?)
        # MEXC returns fully closed candles; still we keep -2 to be safe

        found_any = False

        for idx in reversed(cand_indices):
            c = kl[idx]
            ts = int(c[0])
            o = float(c[1]); h = float(c[2]); l = float(c[3]); cl = float(c[4])

            if cl <= o:
                debug(f"[{sym}] {iso_utc(ts)} fail: not green")
                continue

            br = body_ratio(o, h, l, cl)
            if br < MIN_BODY:
                debug(f"[{sym}] {iso_utc(ts)} fail: body_ratio {br:.3f} < {MIN_BODY}")
                continue

            bwt = bottom_wick_ticks(o, l, cl, tick)
            if bwt > BOTTOM_WICK_TICKS:
                debug(f"[{sym}] {iso_utc(ts)} fail: bottom_wick_ticks {bwt} > {BOTTOM_WICK_TICKS}")
                continue

            # highs window before the signal candle
            highs_before = [float(x[2]) for x in kl[:idx]]
            ok, usedN = passes_breakout(cl, highs_before, tick)
            if not ok:
                debug(f"[{sym}] {iso_utc(ts)} fail: close not > max high of lookbacks {LOOKBACKS}")
                continue

            if not check_untouched(UNTOUCHED, idx, kl):
                debug(f"[{sym}] {iso_utc(ts)} fail: untouched rule '{UNTOUCHED}' violated")
                continue

            # how many candles ago from the last closed
            candles_ago = (len(kl) - 1) - idx
            print(f"{sym},{iso_utc(ts)},{cl:.8f},{br:.3f},{usedN},{h:.8f},{l:.8f},{bwt},{tick:.8f},{candles_ago}")
            found_any = True

        if not found_any:
            # keep silent to avoid huge logs unless DEBUG
            if DEBUG:
                debug(f"[{sym}] no signal in last {MAX_CANDLES_AGO} candles within window {CANDLE_WINDOW}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
