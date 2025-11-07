#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, math, sys, csv
from datetime import datetime, timezone
import requests

API = "https://api.mexc.com"

def get_exchange_info():
    r = requests.get(f"{API}/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    data = r.json()
    symbols = []
    for s in data.get("symbols", []):
        # keep only real spot USDT pairs that are trading
        quote = s.get("quoteAsset")
        status = s.get("status") == "TRADING"
        perms = set(s.get("permissions", []))
        is_spot = ("SPOT" in perms) or s.get("isSpotTradingAllowed", True)
        if quote == "USDT" and status and is_spot:
            # pull tick size
            tick = None
            for f in s.get("filters", []):
                if f.get("filterType") in ("PRICE_FILTER", "MIN_PRICE"):
                    ts = f.get("tickSize") or f.get("minPrice")
                    if ts is not None:
                        try:
                            tick = float(ts)
                        except:
                            pass
            if tick is None:
                # fallback from quote precision
                qp = s.get("quotePrecision", 6)
                tick = 10 ** (-int(qp))
            symbols.append((s["symbol"], tick))
    return symbols

def get_klines(symbol, interval="4h", limit=220):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(f"{API}/api/v3/klines", params=params, timeout=20)
    r.raise_for_status()
    rows = r.json()
    # each row: [openTime, open, high, low, close, volume, closeTime, ...]
    out = []
    for k in rows:
        o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4])
        ct = int(k[6]) // 1000
        out.append((ct, o, h, l, c))
    return out

def body_ratio(o,h,l,c):
    rng = h - l
    if rng <= 0: return 0.0
    return max(0.0, (c - o) / rng)

def fmt_float(x):
    # stable printing like samples
    if x == 0: return "0"
    s = f"{x:.10g}"
    return s

def scan_symbol(sym, tick, args):
    kl = get_klines(sym, "4h", limit=max(args.candle_window + 20, 60))
    if len(kl) < 25:
        return []
    hits = []
    highs = [row[2] for row in kl]
    lows  = [row[3] for row in kl]
    for i in range(len(kl)):
        # only look inside the last window
        if i < len(kl) - args.candle_window: 
            continue
        ts,o,h,l,c = kl[i]
        # green full body
        if c <= o: 
            continue
        br = body_ratio(o,h,l,c)
        if br < args.min_body:
            continue
        # bottom wick in ticks
        bw_ticks = max(0.0, (o - l) / tick)
        # quantize to nearest tick floor
        if bw_ticks > 1.0000001:
            continue
        # breakout above max of last 15 or last 20
        ok15 = False
        ok20 = False
        if i >= 15:
            prev_max15 = max(highs[i-15:i]) if i>=15 else -math.inf
            ok15 = c > prev_max15
        if i >= 20:
            prev_max20 = max(highs[i-20:i]) if i>=20 else -math.inf
            ok20 = c > prev_max20
        if not (ok15 or ok20):
            continue
        n_used = 15 if ok15 else 20
        # untouched rule next_low only
        if i+1 < len(kl):
            next_low = kl[i+1][3]
            if next_low <= l + 1e-15:
                continue
        # fresh filter by candles_ago
        candles_ago = (len(kl) - 1) - i
        if candles_ago > args.max_candles_ago:
            continue
        # optional target not hit after signal
        if args.target_pct > 0 and i+1 < len(kl):
            tgt = c * (1 + args.target_pct)
            fut_highs = [row[2] for row in kl[i+1:]]
            if any(hh >= tgt for hh in fut_highs):
                continue
        # keep
        hits.append({
            "symbol": sym,
            "time_utc": datetime.fromtimestamp(kl[i][0], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "close": c,
            "body": br,
            "n_used": n_used,
            "hi": h,
            "lo": l,
            "bottom_wick_ticks": int(round(bw_ticks)),
            "tick": tick,
            "candles_ago": candles_ago
        })
    # keep only most recent hit per symbol
    if not hits:
        return []
    hits.sort(key=lambda x: x["candles_ago"])
    return [hits[0]]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candle-window", type=int, default=180, dest="candle_window",
                   help="how many recent candles to search")
    p.add_argument("--max-candles-ago", type=int, default=1, dest="max_candles_ago",
                   help="only signals within this many candles ago")
    p.add_argument("--min-body", type=float, default=0.70, dest="min_body",
                   help="minimum full body ratio")
    p.add_argument("--target-pct", type=float, default=0.05, dest="target_pct",
                   help="filter out signals that already hit target after signal")
    p.add_argument("--limit-symbols", type=int, default=0, help="debug limit")
    p.add_argument("--include", type=str, default="", help="comma separated whitelist symbols like HIPPOPUSDT,AIAUSDT")
    args = p.parse_args()

    # universe
    syms = get_exchange_info()
    # optional include priority to be safe for HIPPO AIA
    include = {s.strip().upper() for s in args.include.split(",") if s.strip()}
    if include:
        keep = []
        for s,t in syms:
            if s in include:
                keep.append((s,t))
        # also add the rest after included
        for s,t in syms:
            if s not in include:
                keep.append((s,t))
        syms = keep
    if args.limit_symbols > 0:
        syms = syms[:args.limit_symbols]

    # scan
    hits_total = []
    scanned = 0
    for sym,tick in syms:
        try:
            out = scan_symbol(sym, tick, args)
            if out:
                hits_total.extend(out)
        except Exception as e:
            # keep going
            pass
        scanned += 1

    # sort newest first then by body desc
    hits_total.sort(key=lambda x: (x["candles_ago"], -x["body"]))

    # header and summary
    print("# universe symbols=%d" % len(syms))
    print("# scanned=%d hits=%d window=%d max_candles_ago=%d" % (
        scanned, len(hits_total), args.candle_window, args.max_candles_ago
    ))
    print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
    w = csv.writer(sys.stdout)
    for h in hits_total:
        w.writerow([
            h["symbol"],
            h["time_utc"],
            fmt_float(h["close"]),
            f"{h['body']:.2f}",
            h["n_used"],
            fmt_float(h["hi"]),
            fmt_float(h["lo"]),
            h["bottom_wick_ticks"],
            fmt_float(h["tick"]),
            h["candles_ago"],
        ])

if __name__ == "__main__":
    main()
