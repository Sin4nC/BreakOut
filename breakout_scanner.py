# breakout_scanner.py
# MEXC Spot USDT 4H breakout scanner — strict fixed rules per Forz4crypto
#
# Fixed rules  no switches
# 1) Green full body  body_ratio >= 0.70
# 2) Breakout  close_q > prev_high_q over previous 15 OR 20 CLOSED candles  either qualifies
#    Quantization by tick size  x_q = floor(x / tick) * tick
# 3) Untouched  ALL subsequent CLOSED candles must NOT touch or break the signal LOW
#    Touch means low_q <= signal_low_q  equality counts as touch
# 4) Suppression ON  if any later CLOSED candle makes a strictly higher HIGH than the signal  older signal is dropped
# 5) Bottom wick threshold  <= 1 tick
# 6) Freshness OFF  we allow old signals if they still satisfy 3 and 4
#
# Output  one latest surviving signal per symbol
# CSV header
# symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago

import argparse, concurrent.futures as cf, time, math, requests
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

# ---------- CLI minimal ----------
ap = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
ap.add_argument("--api", default="https://api.mexc.com")
ap.add_argument("--interval", default="4h")
ap.add_argument("--window", type=int, default=180)     # how many CLOSED candles to scan backward
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--symbols-file", default=None)
ap.add_argument("--quote", default="USDT")
ap.add_argument("--sleep", type=float, default=0.16)
ap.add_argument("--debug-symbol", default=None)        # optional single symbol debug
args = ap.parse_args()

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/forz4crypto-3.0"})

LOOKBACKS = [15, 20]
MIN_BODY = 0.70
MAX_BOTTOM_WICK_TICKS = 1
DEFAULT_TICK = 1e-6

# ---------- utils ----------
def http_get(url: str, params: Dict = None, max_retries: int = 6):
    params = params or {}
    back = max(args.sleep, 0.05)
    for _ in range(max_retries):
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(back)
            back = min(back * 1.8, 5.0)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r

def qfloor(x: float, tick: float) -> float:
    t = tick if tick and tick > 0 else DEFAULT_TICK
    return math.floor(x / t) * t

def to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def last_closed_index(kl) -> int:
    now_ms = int(time.time() * 1000)
    i = len(kl) - 1
    while i >= 0 and kl[i][5] > now_ms:
        i -= 1
    return i

# ---------- universe ----------
def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s:
                out.append((s, DEFAULT_TICK))
    return out

def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    ticks: Dict[str, float] = {}
    quote = quote.upper()
    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        for s in ei.get("symbols", []):
            sym = (s.get("symbol") or s.get("symbolName") or "").upper()
            if not sym or not sym.endswith(quote):
                continue
            # no status or permissions filter  keep all USDT spot style symbols
            tick = DEFAULT_TICK
            for f in s.get("filters", []):
                if str(f.get("filterType") or "").upper() == "PRICE_FILTER":
                    ts = f.get("tickSize") or f.get("tick_size")
                    if ts:
                        try:
                            tick = float(ts)
                        except Exception:
                            pass
            ticks[sym] = tick
        if ticks:
            out = sorted((k, ticks[k]) for k in ticks)
    except Exception:
        pass

    if not out:
        # fallback  use ticker price and default tick
        try:
            tp = http_get(f"{api}/api/v3/ticker/price").json()
            symbols = sorted(
                {row["symbol"].upper() for row in tp if row.get("symbol", "").upper().endswith(quote)}
            )
            out = [(s, DEFAULT_TICK) for s in symbols]
        except Exception:
            out = []
    return out

# ---------- data ----------
def get_klines(symbol: str, interval: str, limit: int):
    time.sleep(args.sleep)
    data = http_get(
        f"{args.api}/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    ).json()
    kl = []
    for r in data:
        try:
            ot = int(r[0]); o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4]); ct = int(r[6])
            kl.append((ot, o, h, l, c, ct))
        except Exception:
            pass
    return kl

# ---------- logic ----------
def passes_fixed_rules(kl, idx: int, tick: float) -> Optional[Tuple[float, float, int, int]]:
    # returns body_ratio, bottom_wick_ticks, n_used, candles_ago if ok else None
    ts, o, h, l, c, _ = kl[idx]
    rng = h - l
    if rng <= 0:
        return None
    if c <= o:
        return None

    body_ratio = (c - o) / rng
    if body_ratio < MIN_BODY:
        return None

    denom = tick if tick and tick > 0 else DEFAULT_TICK

    # bottom wick ticks
    bottom_wick = max(0.0, min(o, c) - l)
    bottom_wick_ticks = int(math.floor(bottom_wick / denom + 1e-12))
    if bottom_wick_ticks > MAX_BOTTOM_WICK_TICKS:
        return None

    last = last_closed_index(kl)

    # breakout with tick quantization
    passed_N = []
    for N in LOOKBACKS:
        if idx - N < 0:
            continue
        prev_slice = kl[idx - N: idx]
        prev_high = max(prev_slice, key=lambda x: x[2])[2]
        prev_high_q = qfloor(prev_high, denom)
        close_q = qfloor(c, denom)
        if close_q > prev_high_q:
            passed_N.append(N)
    if not passed_N:
        return None
    n_used = min(passed_N)

    # untouched low  ALL subsequent CLOSED candles
    sig_low_q = qfloor(l, denom)
    for j in range(idx + 1, last + 1):
        low_q = qfloor(kl[j][3], denom)
        if low_q <= sig_low_q:
            return None

    # suppression ON  any later higher HIGH kills this signal
    for j in range(idx + 1, last + 1):
        if kl[j][2] > h:
            return None

    candles_ago = last - idx
    return body_ratio, bottom_wick_ticks, n_used, candles_ago

def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    limit = max(args.window + max(LOOKBACKS) + 8, 120)
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 5:
        return None

    last = last_closed_index(kl)
    if last < max(LOOKBACKS):
        return None
    start = max(last - args.window + 1, max(LOOKBACKS))

    # newest to older  pick first that survives  this is the latest surviving signal
    for idx in range(last, start - 1, -1):
        res = passes_fixed_rules(kl, idx, tick if tick > 0 else DEFAULT_TICK)
        if res is None:
            continue
        body_ratio, bottom_wick_ticks, n_used, candles_ago = res
        ts, o, h, l, c, _ = kl[idx]
        row = [
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{body_ratio:.2f}",
            f"{n_used}",
            f"{h:g}",
            f"{l:g}",
            f"{bottom_wick_ticks}",
            f"{(tick if tick and tick > 0 else DEFAULT_TICK):g}",
            f"{candles_ago}",
        ]
        return ",".join(row)
    return None

# ---------- main ----------
def main():
    # universe
    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        src = "symbols_file"
    else:
        universe = load_universe(args.api, args.quote)
        src = "exchangeInfo" if universe else "ticker/price"

    print(f"# universe source={src} symbols={len(universe)}")
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago")
    if not universe:
        return

    # optional single symbol debug
    if args.debug_symbol:
        debug_sym = args.debug_symbol.upper()
        for sym, tick in universe:
            if sym == debug_sym:
                line = scan_symbol((sym, tick))
                if line:
                    print(line)
                return
        return

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(scan_symbol, universe, chunksize=32):
            if out:
                print(out, flush=True)

if __name__ == "__main__":
    main()
