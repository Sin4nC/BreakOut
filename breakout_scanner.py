# breakout_scanner_perp.py
# MEXC USDT-M Perpetual 4H breakout scanner — Forz4crypto rules, SUPPRESSION OFF

import argparse, concurrent.futures as cf, time, math, requests
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

# ----- fixed rules -----
LOOKBACKS = [15, 20]           # pass if close > max high of last 15 OR 20 closed candles
MIN_BODY = 0.70                # full-body green
MAX_BOTTOM_WICK_TICKS = 1      # bottom wick <= 1 tick
SUPPRESSION = False            # keep historical signals even if later highs print
DEFAULT_TICK = 1e-6

# ----- CLI -----
ap = argparse.ArgumentParser("MEXC Perp 4H Breakout Scanner")
ap.add_argument("--api", default="https://contract.mexc.com")
ap.add_argument("--interval", default="Hour4")       # futures intervals: Min1 Min5 Min15 Min30 Min60 Hour4 Hour8 Day1 Week1 Month1
ap.add_argument("--window", type=int, default=180)   # how many CLOSED candles back to search
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--symbols-file", default=None)      # optional custom universe one symbol per line e.g. BTC_USDT
ap.add_argument("--sleep", type=float, default=0.16)
ap.add_argument("--allow-fallback", action="store_true", help="if contract/detail fails use ticker list with default tick")
args = ap.parse_args()

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/perp-forz4crypto-1.0"})

# ----- helpers -----
PERIOD_SECS = {
    "Min1": 60, "Min5": 300, "Min15": 900, "Min30": 1800, "Min60": 3600,
    "Hour4": 14400, "Hour8": 28800, "Day1": 86400, "Week1": 604800, "Month1": 2592000
}

def http_get(url: str, params: Dict = None, max_retries: int = 6):
    params = params or {}
    back = max(args.sleep, 0.05)
    for _ in range(max_retries):
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(back); back = min(back * 1.8, 5.0); continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r

def qfloor(x: float, tick: float) -> float:
    t = tick if tick and tick > 0 else DEFAULT_TICK
    return math.floor(x / t) * t

def to_utc(sec: int) -> str:
    return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def last_closed_index(kl, period_sec: int) -> int:
    """kl rows carry start time in seconds; treat candle closed if start_time + period <= now"""
    now = int(time.time())
    i = len(kl) - 1
    while i >= 0 and (kl[i][0] + period_sec) > now:
        i -= 1
    return i

# ----- universe -----
def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if not s or not s.endswith("_USDT"):
                continue
            out.append((s, DEFAULT_TICK))
    return out

def load_universe_from_detail(api: str) -> List[Tuple[str, float]]:
    """Prefer this path to get BOTH symbol list and priceUnit tick in one shot."""
    data = http_get(f"{api}/api/v1/contract/detail").json().get("data")
    items = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
    out: List[Tuple[str, float]] = []
    for d in items:
        try:
            if str(d.get("settleCoin","")).upper() != "USDT":
                continue
            if int(d.get("state", 0)) != 0:  # 0 enabled
                continue
            sym = str(d.get("symbol","")).upper()
            if not sym.endswith("_USDT"):
                continue
            tick = float(d.get("priceUnit") or d.get("price_unit") or 0.0) or DEFAULT_TICK
            out.append((sym, tick))
        except:
            continue
    return sorted(out)

def load_universe_fallback(api: str) -> List[Tuple[str, float]]:
    """If detail fails and --allow-fallback set, use ticker to list symbols with default tick."""
    tj = http_get(f"{api}/api/v1/contract/ticker").json()
    arr = tj.get("data") or []
    symbols = sorted({str(x.get("symbol","")).upper() for x in arr if str(x.get("symbol","")).upper().endswith("_USDT")})
    return [(s, DEFAULT_TICK) for s in symbols]

def load_universe(api: str) -> Tuple[List[Tuple[str, float]], str]:
    try:
        u = load_universe_from_detail(api)
        if u:
            return u, "contract/detail"
    except Exception:
        pass
    if args.allow_fallback:
        try:
            return load_universe_fallback(api), "contract/ticker_fallback"
        except Exception:
            return [], "fallback_error"
    return [], "detail_error"

# ----- data -----
def get_klines(symbol: str, interval: str, need: int):
    """
    Futures kline returns arrays under data
    We request enough window using start end seconds
    """
    period = PERIOD_SECS.get(interval, 14400)
    end = int(time.time())
    start = end - (need + 10) * period
    time.sleep(args.sleep)
    j = http_get(f"{args.api}/api/v1/contract/kline/{symbol}", {
        "interval": interval, "start": start, "end": end
    }).json()
    d = j.get("data") or {}
    t = d.get("time") or []
    o = d.get("open") or []
    h = d.get("high") or []
    l = d.get("low") or []
    c = d.get("close") or []
    kl = []
    n = min(len(t), len(o), len(h), len(l), len(c))
    for i in range(n):
        try:
            ts = int(t[i])         # seconds
            oo = float(o[i]); hh = float(h[i]); ll = float(l[i]); cc = float(c[i])
            kl.append((ts, oo, hh, ll, cc))
        except:
            continue
    return kl, period

# ----- rules -----
def passes_rules(kl, idx: int, tick: float, last: int) -> Optional[Tuple[float, int, int, int]]:
    ts, o, h, l, c = kl[idx]
    rng = h - l
    if rng <= 0 or c <= o:
        return None

    body_ratio = (c - o) / rng
    if body_ratio < MIN_BODY:
        return None

    denom = tick if tick and tick > 0 else DEFAULT_TICK
    bottom_wick = max(0.0, min(o, c) - l)
    bottom_wick_ticks = int(math.floor(bottom_wick / denom + 1e-12))
    if bottom_wick_ticks > MAX_BOTTOM_WICK_TICKS:
        return None

    # strict breakout vs highs of last N CLOSED candles using tick-floor
    passed_N = []
    close_q = qfloor(c, denom)
    for N in LOOKBACKS:
        if idx - N < 0:
            continue
        prev_high = max(x[2] for x in kl[idx - N: idx])
        if close_q > qfloor(prev_high, denom):
            passed_N.append(N)
    if not passed_N:
        return None
    n_used = min(passed_N)

    # untouched ALL
    sig_low_q = qfloor(l, denom)
    for j in range(idx + 1, last + 1):
        if qfloor(kl[j][3], denom) <= sig_low_q:
            return None

    # suppression OFF by default
    if SUPPRESSION:
        for j in range(idx + 1, last + 1):
            if kl[j][2] > h:
                return None

    candles_ago = last - idx
    return body_ratio, bottom_wick_ticks, n_used, candles_ago

def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    need = max(args.window + max(LOOKBACKS) + 8, 120)
    try:
        kl, period = get_klines(symbol, args.interval, need)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 5:
        return None

    last = last_closed_index(kl, period)
    if last < max(LOOKBACKS):
        return None
    start = max(last - args.window + 1, max(LOOKBACKS))

    # newest to oldest return only the latest surviving signal
    for idx in range(last, start - 1, -1):
        res = passes_rules(kl, idx, tick if tick > 0 else DEFAULT_TICK, last)
        if res is None:
            continue
        body_ratio, bottom_wick_ticks, n_used, candles_ago = res
        ts, o, h, l, c = kl[idx]
        return ",".join([
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
        ])
    return None

# ----- main -----
def main():
    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        src = "symbols_file"
    else:
        universe, src = load_universe(args.api)

    print(f"# universe source={src} symbols={len(universe)} suppression={SUPPRESSION} interval={args.interval}")
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago")

    if not universe:
        return

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(scan_symbol, universe, chunksize=32):
            if out:
                print(out, flush=True)

if __name__ == "__main__":
    main()
