# MEXC 4H breakout scanner — full-body green + close above max high of prev 15 OR 20
# Untouched LOW rule = ALL later CLOSED candles must not touch the signal LOW
# Suppression rule default ON = if any later CLOSED candle prints HIGH > signal HIGH, drop the older signal
# Universe = MEXC spot USDT pairs from exchangeInfo
# Output = CSV rows, one LATEST surviving signal per symbol

import argparse, concurrent.futures as cf, time, sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import requests

# ---------- CLI ----------
p = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
p.add_argument("--api", default="https://api.mexc.com", help="MEXC REST base URL")
p.add_argument("--interval", default="4h", help="Kline interval")
p.add_argument("--window", type=int, default=180, help="search window in CLOSED candles")
p.add_argument("--lookbacks", default="15,20", help="comma list of N values for breakout over max HIGH of previous N CLOSED candles")
p.add_argument("--min-body", type=float, default=0.70, help="full-body threshold in 0..1")
p.add_argument("--untouched", choices=["none", "next", "all"], default="all",
               help="untouched LOW rule scope")
p.add_argument("--max-candles-ago", type=int, default=-1,
               help="freshness filter; -1 disables")
p.add_argument("--max-bottom-wick-ticks", type=int, default=1,
               help="require bottom wick <= this many price ticks")
p.add_argument("--symbols-file", default=None, help="optional file with symbols one per line")
p.add_argument("--workers", type=int, default=10, help="thread workers")
p.add_argument("--sleep", type=float, default=0.25, help="sleep between API calls to avoid 429")
p.add_argument("--quote", default="USDT", help="quote asset filter for universe")
# suppression ON by default, but allow disabling via --no-suppress
group = p.add_mutually_exclusive_group()
group.add_argument("--suppress", dest="suppress", action="store_true", default=True,
                   help="drop older signals if any later CLOSED candle prints higher HIGH than that signal")
group.add_argument("--no-suppress", dest="suppress", action="store_false",
                   help="keep older signals even if later HIGH is higher")
args = p.parse_args()

LOOKBACKS = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.0"})
DEFAULT_TICK = 1e-6

# ---------- HTTP helpers ----------
def http_get(url: str, params: Dict = None, max_retries: int = 6) -> requests.Response:
    params = params or {}
    backoff = args.sleep
    for _ in range(max_retries):
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r  # never

# ---------- Universe & tick size ----------
def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    ticks: Dict[str, float] = {}
    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        for s in ei.get("symbols", []):
            sym = s.get("symbol") or s.get("symbolName")
            if not sym or not sym.endswith(quote):
                continue
            status = (s.get("status") or "TRADING").upper()
            if status not in ("TRADING", "ENABLED"):
                continue
            tick = DEFAULT_TICK
            for f in s.get("filters", []):
                if (f.get("filterType") or "").upper() in ("PRICE_FILTER",):
                    ts = f.get("tickSize") or f.get("tick_size")
                    if ts is not None:
                        try:
                            tick = float(ts)
                        except Exception:
                            tick = DEFAULT_TICK
            ticks[sym] = tick
        out = sorted([(k, ticks[k]) for k in ticks])
    except Exception:
        out = []

    if not out:
        # fallback when exchangeInfo is unavailable
        try:
            tp = http_get(f"{api}/api/v3/ticker/price").json()
            symbols = sorted({row["symbol"] for row in tp if row.get("symbol", "").endswith(quote)})
            out = [(s, DEFAULT_TICK) for s in symbols]
        except Exception:
            out = []
    return out

def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s:
                out.append((s, DEFAULT_TICK))
    return out

# ---------- Data ----------
def get_klines(symbol: str, interval: str, limit: int):
    # friendly throttle
    time.sleep(args.sleep)
    data = http_get(f"{args.api}/api/v3/klines", {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }).json()
    # item = [openTime, open, high, low, close, volume, closeTime, ...]
    out = []
    for r in data:
        out.append((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), int(r[6])))
    return out  # (ts, o, h, l, c, close_ts)

def to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------- Core checks ----------
def latest_closed_index(kl) -> int:
    now_ms = int(time.time() * 1000)
    last = len(kl) - 1
    if last < 0:
        return -1
    return last if kl[last][5] <= now_ms else last - 1

def full_body_green(o: float, h: float, l: float, c: float, min_body: float) -> bool:
    rng = h - l
    if rng <= 0:
        return False
    if c <= o:
        return False
    body_ratio = (c - o) / rng
    return body_ratio >= min_body

def bottom_wick_in_ticks(o: float, l: float, tick: float) -> int:
    if tick is None or tick <= 0:
        tick = DEFAULT_TICK
    bw = max(0.0, min(o, o) - l)  # since c>o for green, min(o,c)=o
    return int(round(bw / tick))

def close_above_prev_high(kl, idx: int, lookbacks: List[int]) -> Optional[int]:
    # return the smallest N that passes (any N qualifies)
    passed: List[int] = []
    for N in lookbacks:
        if idx - N < 0:
            continue
        prev_high = max(kl[j][2] for j in range(idx - N, idx))
        if kl[idx][4] > prev_high:
            passed.append(N)
    if not passed:
        return None
    return min(passed)

def untouched_low_rule(kl, idx: int, scope: str, last_closed: int) -> bool:
    l = kl[idx][3]
    if scope == "none":
        return True
    if scope == "next":
        if idx + 1 <= last_closed:
            return kl[idx + 1][3] > l
        return True
    # scope == "all"
    for j in range(idx + 1, last_closed + 1):
        if kl[j][3] <= l:
            return False
    return True

def later_higher_high_exists(kl, idx: int, last_closed: int) -> bool:
    sig_high = kl[idx][2]
    for j in range(idx + 1, last_closed + 1):
        if kl[j][2] > sig_high:
            return True
    return False

# ---------- Scan one symbol ----------
def scan_symbol(item: Tuple[str, float]) -> Optional[str]:
    symbol, tick = item
    # need enough candles for window + max lookback + some room
    limit = max(args.window + max(LOOKBACKS) + 10, 250)
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 2:
        return None

    last_closed = latest_closed_index(kl)
    if last_closed < max(LOOKBACKS):
        return None

    start = max(0, last_closed - args.window + 1)

    candidates: List[Tuple[int, str]] = []  # (idx, csv_row)
    for idx in range(start, last_closed + 1):
        ts, o, h, l, c, ct = kl[idx]

        # 1) must be green full-body
        if not full_body_green(o, h, l, c, args.min_body):
            continue

        # 2) bottom wick <= threshold ticks
        bw_ticks = bottom_wick_in_ticks(o, l, tick)
        if bw_ticks > args.max_bottom_wick_ticks:
            continue

        # 3) breakout over max HIGH of previous N CLOSED candles
        n_used = close_above_prev_high(kl, idx, LOOKBACKS)
        if n_used is None:
            continue

        # 4) untouched LOW rule
        if not untouched_low_rule(kl, idx, args.untouched, last_closed):
            continue

        # 5) freshness if enabled
        candles_ago = last_closed - idx
        if args.max_candles_ago >= 0 and candles_ago > args.max_candles_ago:
            continue

        # 6) suppression check against later CLOSED candles (if ON)
        if args.suppress and later_higher_high_exists(kl, idx, last_closed):
            continue

        # Passed — prepare CSV row
        row = ",".join([
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{((c - o) / (h - l)):.2f}",
            f"{n_used}",
            f"{h:g}",
            f"{l:g}",
            f"{bw_ticks}",
            f"{(tick if tick and tick > 0 else DEFAULT_TICK):g}",
            f"{candles_ago}",
        ])
        candidates.append((idx, row))

    if not candidates:
        return None

    # Keep only the LATEST surviving candidate for this symbol
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

# ---------- Main ----------
def main():
    # build universe
    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        source_note = "symbols_file"
    else:
        universe = load_universe(args.api, args.quote)
        source_note = "exchangeInfo" if universe else "ticker/price"

    print(f"# universe source={source_note} symbols={len(universe)}")
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago")

    if not universe:
        return

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(scan_symbol, universe, chunksize=25):
            if res:
                print(res, flush=True)

if __name__ == "__main__":
    main()
