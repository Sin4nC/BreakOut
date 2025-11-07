#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, math, sys, csv, os
from datetime import datetime, timezone
import requests

DEF_API = os.environ.get("MEXC_API_BASE", "https://api.mexc.com")
HDR = {"User-Agent": "BreakoutScanner/1.0 (+GitHub Actions)"}

def get_exchange_info(base):
    # try full exchangeInfo first
    try:
        r = requests.get(f"{base}/api/v3/exchangeInfo", timeout=25, headers=HDR)
        r.raise_for_status()
        data = r.json()
        syms_raw = data.get("symbols") or (data.get("data") or {}).get("symbols") or []
    except Exception:
        syms_raw = []
    symbols = []
    for s in syms_raw:
        try:
            quote = s.get("quoteAsset")
            status = s.get("status") == "TRADING"
            perms = set(s.get("permissions", []))
            is_spot = ("SPOT" in perms) or s.get("isSpotTradingAllowed", True)
            if quote == "USDT" and status and is_spot:
                tick = None
                for f in s.get("filters", []):
                    if f.get("filterType") in ("PRICE_FILTER", "MIN_PRICE"):
                        ts = f.get("tickSize") or f.get("minPrice")
                        if ts is not None:
                            tick = float(ts)
                            break
                if tick is None:
                    qp = s.get("quotePrecision", 6)
                    try:
                        tick = 10 ** (-int(qp))
                    except Exception:
                        tick = 1e-6
                symbols.append((s["symbol"], float(tick)))
        except Exception:
            continue
    if symbols:
        print(f"# debug exchangeInfo_symbols={len(symbols)} api_base={base}")
        return symbols

    # fallback to ticker/price if exchangeInfo failed or empty
    try:
        r = requests.get(f"{base}/api/v3/ticker/price", timeout=25, headers=HDR)
        r.raise_for_status()
        arr = r.json()
        tick_default = 1e-6
        symbols = []
        for it in arr:
            sym = it.get("symbol", "")
            if sym.endswith("USDT"):
                symbols.append((sym, tick_default))
        print(f"# debug exchangeInfo_symbols=0 ticker_price_symbols={len(symbols)} api_base={base}")
        return symbols
    except Exception:
        print(f"# debug both_endpoints_failed api_base={base}")
        return []

def get_klines(base, symbol, interval="4h", limit=220):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(f"{base}/api/v3/klines", params=params, timeout=25, headers=HDR)
    r.raise_for_status()
    rows = r.json()
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
    if x == 0: return "0"
    return f"{x:.10g}"

def scan_symbol(base, sym, tick, args):
    kl = get_klines(base, sym, "4h", limit=max(args.candle_window + 20, 60))
    if len(kl) < 25:
        return []
    hits = []
    highs = [row[2] for row in kl]
    lows  = [row[3] for row in kl]
    for i in range(len(kl)):
        if i < len(kl) - args.candle_window:
            continue
        ts,o,h,l,c = kl[i]
        if c <= o:
            continue
        br = body_ratio(o,h,l,c)
        if br < args.min_body:
            continue
        bw_ticks = max(0.0, (o - l) / tick)
        if bw_ticks > 1.0000001:
            continue
        ok15 = i >= 15 and c > max(highs[i-15:i])
        ok20 = i >= 20 and c > max(highs[i-20:i])
        if not (ok15 or ok20):
            continue
        n_used = 15 if ok15 else 20
        if i+1 < len(kl):
            next_low = kl[i+1][3]
            if next_low <= l + 1e-15:
                continue
        candles_ago = (len(kl) - 1) - i
        if candles_ago > args.max_candles_ago:
            continue
        if args.target_pct > 0 and i+1 < len(kl):
            tgt = c * (1 + args.target_pct)
            fut_highs = [row[2] for row in kl[i+1:]]
            if any(hh >= tgt for hh in fut_highs):
                continue
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
    if not hits:
        return []
    hits.sort(key=lambda x: x["candles_ago"])
    return [hits[0]]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", type=str, default=DEF_API, help="MEXC api base")
    p.add_argument("--candle-window", type=int, default=180, dest="candle_window")
    p.add_argument("--max-candles-ago", type=int, default=1, dest="max_candles_ago")
    p.add_argument("--min-body", type=float, default=0.70, dest="min_body")
    p.add_argument("--target-pct", type=float, default=0.05, dest="target_pct")
    p.add_argument("--limit-symbols", type=int, default=0)
    p.add_argument("--include", type=str, default="")
    args = p.parse_args()

    syms = get_exchange_info(args.api_base)

    include = {s.strip().upper() for s in args.include.split(",") if s.strip()}
    if include:
        keep = []
        for s,t in syms:
            if s in include: keep.append((s,t))
        for s,t in syms:
            if s not in include: keep.append((s,t))
        syms = keep

    if args.limit_symbols > 0:
        syms = syms[:args.limit_symbols]

    hits_total = []
    scanned = 0
    for sym,tick in syms:
        try:
            out = scan_symbol(args.api_base, sym, tick, args)
            if out: hits_total.extend(out)
        except Exception:
            pass
        scanned += 1

    hits_total.sort(key=lambda x: (x["candles_ago"], -x["body"]))

    print("# universe symbols=%d" % len(syms))
    print("# scanned=%d hits=%d window=%d max_candles_ago=%d" % (
        scanned, len(hits_total), args.candle_window, args.max_candles_ago
    ))
    print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
    w = csv.writer(sys.stdout)
    for h in hits_total:
        w.writerow([
            h["symbol"], h["time_utc"], fmt_float(h["close"]), f"{h['body']:.2f}",
            h["n_used"], fmt_float(h["hi"]), fmt_float(h["lo"]), h["bottom_wick_ticks"],
            fmt_float(h["tick"]), h["candles_ago"],
        ])

if __name__ == "__main__":
    main()
