#!/usr/bin/env python3
import argparse, sys, time, math, json, random, threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--api", default="https://api.mexc.com", help="REST base url")
p.add_argument("--interval", default="4h")
p.add_argument("--window", type=int, default=45, help="lookback window candles to scan")
p.add_argument("--lookbacks", default="10,15", help="breakout over max high of these N")
p.add_argument("--min-body", type=float, default=0.70, help="full-body threshold [0..1]")
p.add_argument("--untouched", choices=["none","next_low","all"], default="next_low",
               help="low untouched rule after signal")
p.add_argument("--max-candles-ago", type=int, default=180, help="freshness filter")
p.add_argument("--target-pct", type=float, default=None, help="require target not hit yet")
p.add_argument("--symbols-file", default=None, help="file with symbols one per line")
p.add_argument("--workers", type=int, default=4, help="max concurrent symbol scans")
p.add_argument("--sleep", type=float, default=0.25, help="sleep seconds before EACH HTTP call")
p.add_argument("--max-retries", type=int, default=6, help="HTTP retry attempts for 429/5xx")
p.add_argument("--universe-limit", type=int, default=None, help="cap number of symbols scanned")
args = p.parse_args()
lookbacks = [int(x) for x in args.lookbacks.split(",")]

# ---------- HTTP with throttle + backoff ----------
http_lock = threading.Lock()
session = requests.Session()
session.headers.update({"User-Agent": "BreakOutScanner/1.0"})

def get(url, **params):
    # throttle
    if args.sleep > 0:
        time.sleep(args.sleep)
    backoff = 0.5
    for attempt in range(args.max_retries + 1):
        r = session.get(url, params=params, timeout=20)
        if r.status_code < 400:
            return r
        # retry for rate limit or transient
        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            delay = float(ra) if ra else backoff * (2 ** attempt)
            delay = min(delay, 8.0)  # cap
            # jitter
            delay = delay * (0.75 + 0.5*random.random())
            time.sleep(delay)
            continue
        r.raise_for_status()
    # last try failed
    r.raise_for_status()

# ---------- helpers ----------
def to_utc(ts_ms):
    return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def fetch_exchange_symbols(api):
    try:
        data = get(f"{api}/api/v3/exchangeInfo").json()
        syms = []
        for s in data.get("symbols", []):
            sym = s.get("symbol")
            if not sym or not sym.endswith("USDT"): 
                continue
            st = s.get("status","TRADING")
            if st != "TRADING": 
                continue
            syms.append(sym)
        return sorted(set(syms))
    except Exception:
        return []

def fetch_ticker_symbols(api):
    data = get(f"{api}/api/v3/ticker/price").json()
    syms = [x["symbol"] for x in data if x["symbol"].endswith("USDT")]
    return sorted(set(syms))

def get_universe():
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            syms = [ln.strip() for ln in f if ln.strip()]
        print(f"# universe source=symbols_file symbols={len(syms)}", file=sys.stderr)
        return syms
    syms = fetch_exchange_symbols(args.api)
    if not syms:
        print("# exchangeInfo returned zero symbols for USDT, falling back", file=sys.stderr)
        syms = fetch_ticker_symbols(args.api)
        print(f"# universe source=ticker/price symbols={len(syms)}", file=sys.stderr)
    if args.universe_limit:
        syms = syms[:args.universe_limit]
        print(f"# universe limited to {len(syms)}", file=sys.stderr)
    return syms

def get_klines(symbol, limit):
    # MEXC v3 format like Binance
    data = get(f"{args.api}/api/v3/klines", symbol=symbol, interval=args.interval, limit=limit).json()
    # each item: [openTime, open, high, low, close, volume, closeTime, ...]
    kl = []
    for it in data:
        o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4])
        kl.append((int(it[0]), o, h, l, c, int(it[6])))
    return kl

def tick_from_price(p):
    # try to guess a tick size from price magnitude
    if p == 0: 
        return 1e-6
    s = f"{p:.12f}".rstrip("0")
    if "." in s:
        dec = len(s.split(".")[1])
        return 10 ** (-min(dec, 6))
    return 1e-6

# ---------- strategy rules ----------
def breakout_records(symbol):
    limit = args.window + max(lookbacks) + 5  # history + a few after-candles
    kl = get_klines(symbol, limit)
    if len(kl) < (max(lookbacks) + 2):
        return []

    recs = []
    # use only CLOSED candles (ignore the last if still open)
    # in MEXC, last item is the just closed for 4h when requested slightly after
    for idx in range(max(lookbacks)+1, len(kl)):
        ts,o,h,l,c,cts = kl[idx]
        rng = (h - l)
        if rng <= 0: 
            continue
        body = (c - o) / rng
        if c <= o: 
            continue  # green full-body only
        if body < args.min_body:
            continue

        # breakout over prior N highs
        window_hi = [kl[idx-j-1][2] for j in range(max(lookbacks))]
        ok_lb = []
        for N in lookbacks:
            prev_max = max(window_hi[:N])
            if c > prev_max:
                ok_lb.append(N)
        if not ok_lb:
            continue

        # untouched rule
        if args.untouched != "none":
            sig_low = l
            if args.untouched == "next_low":
                if idx+1 < len(kl):
                    next_low = kl[idx+1][3]
                    if next_low <= sig_low:
                        continue
            elif args.untouched == "all":
                good = True
                for j in range(idx+1, len(kl)):
                    if kl[j][3] <= sig_low:
                        good = False; break
                if not good:
                    continue

        # freshness
        candles_ago = (len(kl)-1) - idx
        if candles_ago > args.max_candles_ago:
            continue

        # target not hit yet
        if args.target_pct is not None and idx+1 < len(kl):
            tgt = c * (1 + args.target_pct)
            high_after = max(kl[j][2] for j in range(idx+1, len(kl)))
            if high_after >= tgt:
                continue

        # bottom wick ticks (reported only)
        tick = tick_from_price(c)
        bottom_wick_ticks = int(round((min(o,c) - l) / tick))

        # choose the *smallest* N that passed (more strict)
        n_used = min(ok_lb)
        recs.append({
            "symbol": symbol,
            "time_utc": to_utc(kl[idx][0]),
            "close": c,
            "body": round(body, 2),
            "n_used": n_used,
            "hi": h,
            "lo": l,
            "bottom_wick_ticks": bottom_wick_ticks,
            "tick": tick,
            "candles_ago": candles_ago,
        })
    return recs

# ---------- main ----------
def main():
    syms = get_universe()
    print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
    hits = 0
    scanned = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(breakout_records, s): s for s in syms}
        for fut in as_completed(futs):
            scanned += 1
            try:
                for r in fut.result():
                    hits += 1
                    print("{symbol},{time_utc},{close},{body:.2f},{n_used},{hi},{lo},{bottom_wick_ticks},{tick},{candles_ago}"
                          .format(**r), flush=True)
            except Exception as e:
                # keep running on errors (e.g., persistent 429 on a symbol)
                print(f"# warn symbol={futs[fut]} err={type(e).__name__} {e}", file=sys.stderr)

    print(f"# scanned={scanned} hits={hits} window={args.window} max_candles_ago={args.max_candles_ago}", file=sys.stderr)

if __name__ == "__main__":
    main()
