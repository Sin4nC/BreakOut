# breakout_scanner.py
# MEXC 4H breakout scanner — latest surviving signal per symbol (USDT spot)
#
# Core rules (ALL must hold):
# 1) Green full-body candle with body_ratio >= 0.70
# 2) Close strictly ABOVE the max HIGH of the previous 15 OR 20 CLOSED candles (either qualifies)
# 3) Untouched rule = ALL subsequent CLOSED candles must NOT touch/break the signal candle’s LOW
# 4) Suppression ON by default: if any later CLOSED candle prints a higher HIGH than the signal candle,
#    drop/ignore that older signal  only the latest surviving signal remains
# 5) Bottom wick threshold = <= 1 tick (floor to tick)
# 6) Freshness OFF by default (no max-candles-ago constraint)
#
# Extras:
# --only CAKEUSDT,0GUSDT         limit universe
# --explain CAKEUSDT             print accept/reject reasons for each candidate at that symbol
# --suppress on|off              toggle suppression (default on)
# --untouched all|next           untouched mode (default all)
#
# Output CSV header:
# symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago

import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Set
import time, math, requests

# ---------- CLI ----------
ap = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
ap.add_argument("--api", default="https://api.mexc.com", help="MEXC REST base URL")
ap.add_argument("--interval", default="4h", help="kline interval (default 4h)")
ap.add_argument("--window", type=int, default=180, help="how many CLOSED candles back to scan (default 180)")
ap.add_argument("--lookbacks", default="15,20", help="comma list of lookbacks (default 15,20)")
ap.add_argument("--min-body", type=float, default=0.70, help="min full-body ratio (default 0.70)")
ap.add_argument("--max-bottom-wick-ticks", type=int, default=1, help="max bottom wick in ticks, -1 disables (default 1)")
ap.add_argument("--untouched", choices=["all", "next"], default="all", help="untouched rule (default all)")
ap.add_argument("--max-candles-ago", type=int, default=-1, help="-1 disables freshness (default -1)")
ap.add_argument("--suppress", choices=["on", "off"], default="on", help="higher-high suppression (default on)")
ap.add_argument("--symbols-file", default=None, help="optional file with symbols (one per line)")
ap.add_argument("--quote", default="USDT", help="universe filter by quote asset (default USDT)")
ap.add_argument("--only", default="", help="comma list to limit universe e.g. CAKEUSDT,0GUSDT")
ap.add_argument("--explain", default="", help="print reasons for this symbol e.g. CAKEUSDT")
ap.add_argument("--workers", type=int, default=12, help="thread workers (default 12)")
ap.add_argument("--sleep", type=float, default=0.16, help="sleep between API calls to avoid 429 (default 0.16)")
args = ap.parse_args()

LOOKBACKS: List[int] = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.5"})
DEFAULT_TICK = 1e-6
SUPPRESS_ON = (args.suppress.lower() == "on")

ONLY: Set[str] = set([s.strip().upper() for s in args.only.split(",") if s.strip()])
EXPLAIN = args.explain.strip().upper()

# ---------- HTTP ----------
def http_get(url: str, params: Dict = None, max_retries: int = 6) -> requests.Response:
    params = params or {}
    backoff = max(args.sleep, 0.05)
    for _ in range(max_retries):
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 5.0)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r

# ---------- Universe & tick ----------
def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    ticks: Dict[str, float] = {}
    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        for s in ei.get("symbols", []):
            sym = (s.get("symbol") or s.get("symbolName") or "").upper()
            if not sym.endswith(quote):
                continue
            status = (s.get("status") or "TRADING").upper()
            if status not in ("TRADING", "ENABLED"):
                continue
            # keep spot only if permissions present
            perms = [str(p).upper() for p in (s.get("permissions") or s.get("permissionList") or [])]
            if perms and all(p not in ("SPOT", "SPOT_TRADING") for p in perms):
                continue
            tick = DEFAULT_TICK
            for f in s.get("filters", []):
                ftype = str(f.get("filterType") or "").upper()
                if ftype == "PRICE_FILTER":
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
        try:
            tp = http_get(f"{api}/api/v3/ticker/price").json()
            symbols = sorted({row["symbol"].upper() for row in tp if row.get("symbol", "").upper().endswith(quote)})
            out = [(s, DEFAULT_TICK) for s in symbols]
        except Exception:
            out = []
    if ONLY:
        out = [t for t in out if t[0] in ONLY]
    return out

def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s:
                res.append((s, DEFAULT_TICK))
    if ONLY:
        res = [t for t in res if t[0] in ONLY]
    return res

# ---------- Data ----------
def get_klines(symbol: str, interval: str, limit: int):
    time.sleep(args.sleep)
    data = http_get(f"{args.api}/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit}).json()
    kl = []
    for r in data:
        try:
            ot = int(r[0]); o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4]); ct = int(r[6])
            kl.append((ot, o, h, l, c, ct))
        except Exception:
            continue
    return kl

def last_closed_index(kl) -> int:
    now_ms = int(time.time() * 1000)
    idx = len(kl) - 1
    # drop any trailing not-closed bars if present
    while idx >= 0 and kl[idx][5] > now_ms:
        idx -= 1
    return idx

def to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------- Checks ----------
def check_signal_at(kl, idx: int, tick_size: float, reasons: List[str]) -> Tuple[bool, Optional[Dict]]:
    ts, o, h, l, c, _ = kl[idx]
    rng = h - l
    if rng <= 0:
        reasons.append(f"idx {idx} reject rng<=0")
        return False, None
    if c <= o:
        reasons.append(f"idx {idx} reject not_green c<=o")
        return False, None

    body_ratio = (c - o) / rng
    if body_ratio < args.min_body:
        reasons.append(f"idx {idx} reject body_ratio<{args.min_body:.2f} got {body_ratio:.3f}")
        return False, None

    # bottom wick ticks (floor to tick)
    if args.max_bottom_wick_ticks >= 0:
        bottom_wick = max(0.0, min(o, c) - l)
        denom = tick_size if tick_size > 0 else DEFAULT_TICK
        bottom_wick_ticks = int(math.floor(bottom_wick / denom + 1e-12))
        if bottom_wick_ticks > args.max_bottom_wick_ticks:
            reasons.append(f"idx {idx} reject bottom_wick_ticks>{args.max_bottom_wick_ticks} got {bottom_wick_ticks}")
            return False, None
    else:
        bottom_wick_ticks = 0

    # breakout: close strictly above max high of last N CLOSED candles (N in LOOKBACKS)
    passed_N: List[int] = []
    eps = max(tick_size, DEFAULT_TICK) * 1e-3  # tiny epsilon for float stability
    for N in LOOKBACKS:
        if idx - N < 0:
            continue
        prev_high = max(kl[idx - N: idx], key=lambda x: x[2])[2]
        if c > prev_high + eps:
            passed_N.append(N)
    if not passed_N:
        reasons.append(f"idx {idx} reject close_not_above_prev_highs")
        return False, None
    n_used = min(passed_N)

    last = last_closed_index(kl)

    # freshness (optional, default OFF)
    candles_ago = last - idx
    if args.max_candles_ago >= 0 and candles_ago > args.max_candles_ago:
        reasons.append(f"idx {idx} reject too_old candles_ago={candles_ago}")
        return False, None

    # untouched rule
    if args.untouched == "next":
        if idx + 1 <= last and kl[idx + 1][3] <= l:
            reasons.append(f"idx {idx} reject untouched_next low_touched")
            return False, None
    else:
        for j in range(idx + 1, last + 1):
            if kl[j][3] <= l:
                reasons.append(f"idx {idx} reject untouched_all touched_at_j={j}")
                return False, None

    # suppression (default ON): if any later CLOSED candle makes a higher HIGH, reject this older signal
    if SUPPRESS_ON:
        for j in range(idx + 1, last + 1):
            if kl[j][2] > h:
                reasons.append(f"idx {idx} reject suppressed higher_high_at_j={j}")
                return False, None

    return True, {
        "ts": ts, "o": o, "h": h, "l": l, "c": c,
        "body_ratio": body_ratio,
        "n_used": n_used,
        "bottom_wick_ticks": bottom_wick_ticks,
        "candles_ago": candles_ago,
        "tick": tick_size
    }

# ---------- Scan ----------
def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    limit = max(args.window + max(LOOKBACKS) + 8, 80)
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

    # walk from newest to older; first valid is THE latest surviving signal
    for idx in range(last, start - 1, -1):
        reasons: List[str] = []
        ok, info = check_signal_at(kl, idx, tick if tick > 0 else DEFAULT_TICK, reasons)
        if EXPLAIN == symbol:
            # print reasons for each candidate bar
            stamp = to_utc(kl[idx][0])
            for r in reasons:
                print(f"# {symbol} {stamp} {r}")
        if ok:
            inf = info
            row = [
                symbol,
                to_utc(inf["ts"]),
                f"{inf['c']:g}",
                f"{inf['body_ratio']:.2f}",
                f"{inf['n_used']}",
                f"{kl[idx][2]:g}",
                f"{kl[idx][3]:g}",
                f"{inf['bottom_wick_ticks']}",
                f"{inf['tick']:g}",
                f"{inf['candles_ago']}",
            ]
            if EXPLAIN == symbol:
                print(f"# {symbol} SELECTED {row[1]}")
            return ",".join(row)
    return None

# ---------- Main ----------
def main():
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

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(scan_symbol, universe, chunksize=25):
            if res:
                print(res, flush=True)

if __name__ == "__main__":
    main()
