# breakout_scanner.py
# MEXC Spot USDT 4H breakout scanner — latest surviving signal per symbol
#
# Core rules (ALL):
# 1) Green full-body candle with body_ratio >= 0.70
# 2) Close strictly ABOVE the max HIGH of the previous 15 OR 20 CLOSED candles
# 3) Untouched = ALL subsequent CLOSED candles must NOT touch the signal LOW
# 4) Suppression ON: if any later CLOSED candle prints a strictly higher HIGH than the signal,
#    drop that older signal — only the latest surviving one can remain
# 5) Bottom wick threshold <= 1 tick (floor to tick). Set --max-bottom-wick-ticks -1 to disable
# 6) Freshness OFF by default. You can enable via --max-candles-ago N
#
# Extras:
# --only CAKEUSDT,0GUSDT         limit universe
# --explain CAKEUSDT             print accept/reject reasons for that symbol
# --suppress on|off              toggle suppression (default on)
# --untouched all|next           untouched mode (default all)
# --mock-dir ./mocks             read klines from JSON files instead of API for offline test
# --selftest                     run a built-in offline CAKE test that should PASS
#
# Output CSV:
# symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago

import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Set
import time, math, json, os, requests

# ---------- CLI ----------
ap = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
ap.add_argument("--api", default="https://api.mexc.com", help="MEXC REST base URL")
ap.add_argument("--interval", default="4h", help="kline interval (default 4h)")
ap.add_argument("--window", type=int, default=180, help="how many CLOSED candles back to scan")
ap.add_argument("--lookbacks", default="15,20", help="comma list of lookbacks")
ap.add_argument("--min-body", type=float, default=0.70, help="min full-body ratio")
ap.add_argument("--max-bottom-wick-ticks", type=int, default=1, help="max bottom wick in ticks, -1 disables")
ap.add_argument("--untouched", choices=["all", "next"], default="all", help="untouched rule mode")
ap.add_argument("--max-candles-ago", type=int, default=-1, help="-1 disables freshness")
ap.add_argument("--suppress", choices=["on", "off"], default="on", help="higher-high suppression")
ap.add_argument("--symbols-file", default=None, help="optional file with symbols (one per line)")
ap.add_argument("--quote", default="USDT", help="universe filter by quote asset")
ap.add_argument("--only", default="", help="comma list to limit universe e.g. CAKEUSDT,0GUSDT")
ap.add_argument("--explain", default="", help="print reasons for this symbol e.g. CAKEUSDT")
ap.add_argument("--workers", type=int, default=12, help="thread workers")
ap.add_argument("--sleep", type=float, default=0.16, help="sleep between API calls to avoid 429")
ap.add_argument("--mock-dir", default="", help="read klines from mock JSON files instead of API")
ap.add_argument("--selftest", action="store_true", help="run built-in offline test and exit")
args = ap.parse_args()

LOOKBACKS: List[int] = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.6"})
DEFAULT_TICK = 1e-6
SUPPRESS_ON = (args.suppress.lower() == "on")

ONLY: Set[str] = set([s.strip().upper() for s in args.only.split(",") if s.strip()])
EXPLAIN = args.explain.strip().upper()
MOCK = args.mock_dir.strip()

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
    if MOCK:
        # when mocking, build universe from filenames
        syms = []
        for fn in os.listdir(MOCK):
            if fn.upper().endswith(".JSON"):
                sym = fn[:-5].upper()
                if not quote or sym.endswith(quote.upper()):
                    syms.append(sym)
        syms = sorted(set(syms))
        return [(s, DEFAULT_TICK) for s in syms if not ONLY or s in ONLY]

    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        for s in ei.get("symbols", []):
            sym = (s.get("symbol") or s.get("symbolName") or "").upper()
            if not sym.endswith(quote.upper()):
                continue
            status = (s.get("status") or "TRADING").upper()
            if status not in ("TRADING", "ENABLED"):
                continue
            perms = [str(p).upper() for p in (s.get("permissions") or s.get("permissionList") or [])]
            if perms and all(p not in ("SPOT", "SPOT_TRADING") for p in perms):
                continue
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
        try:
            tp = http_get(f"{api}/api/v3/ticker/price").json()
            symbols = sorted({row["symbol"].upper() for row in tp if row.get("symbol", "").upper().endswith(quote.upper())})
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
    if MOCK:
        pth = os.path.join(MOCK, f"{symbol.upper()}.json")
        with open(pth, "r", encoding="utf-8") as f:
            data = json.load(f)
        # expect list of [openTime, open, high, low, close, closeTime]
        kl = []
        for r in data:
            ot = int(r[0]); o=float(r[1]); h=float(r[2]); l=float(r[3]); c=float(r[4]); ct=int(r[5])
            kl.append((ot,o,h,l,c,ct))
        return kl

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

    # bottom wick ticks
    if args.max_bottom_wick_ticks >= 0:
        bottom_wick = max(0.0, min(o, c) - l)
        denom = tick_size if tick_size > 0 else DEFAULT_TICK
        bottom_wick_ticks = int(math.floor(bottom_wick / denom + 1e-12))
        if bottom_wick_ticks > args.max_bottom_wick_ticks:
            reasons.append(f"idx {idx} reject bottom_wick_ticks>{args.max_bottom_wick_ticks} got {bottom_wick_ticks}")
            return False, None
    else:
        bottom_wick_ticks = 0

    # breakout vs max high of previous N CLOSED candles
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

    # suppression
    if SUPPRESS_ON:
        for j in range(idx + 1, last + 1):
            # strictly higher high only
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

    # newest -> older, first valid is the latest surviving signal
    for idx in range(last, start - 1, -1):
        reasons: List[str] = []
        ok, info = check_signal_at(kl, idx, tick if tick > 0 else DEFAULT_TICK, reasons)
        if EXPLAIN == symbol:
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

# ---------- Selftest (offline) ----------
def selftest() -> int:
    # Build a minimal CAKE 4H series where a signal at 2025-11-07 16:00 UTC must pass
    # Format per bar: [openTime, open, high, low, close, closeTime]
    bars = []
    base = 1730400000000  # arbitrary
    step = 4 * 60 * 60 * 1000
    # 24 older bars with highs below 2.90
    px = 2.40
    for i in range(24):
        ot = base + i * step
        ct = ot + step
        o = px
        h = px + 0.20
        l = px - 0.10
        c = px + 0.05
        bars.append([ot, o, h, l, c, ct])
        px += 0.01
    # Signal bar at index 24  pretend lookback highs ~2.60 so we close above them
    ot = base + 24 * step
    ct = ot + step
    o = 2.55; l = 2.55; h = 3.10; c = 3.05  # full-ish body, bottom wick 0
    bars.append([ot, o, h, l, c, ct])
    # 3 later bars that do NOT touch low and do NOT make higher high
    for k in range(3):
        ot = base + (25 + k) * step
        ct = ot + step
        o = 2.98; h = 3.04; l = 2.70; c = 2.90
        bars.append([ot, o, h, l, c, ct])

    # feed through checker
    kl = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), int(r[5])) for r in bars]
    last = last_closed_index(kl)
    idx = last - 3  # signal index
    reasons = []
    ok, info = check_signal_at(kl, idx, 1e-4, reasons)
    if not ok:
        print("# SELFTEST FAIL reasons:")
        for r in reasons: print("# ", r)
        return 1
    row = [
        "CAKEUSDT",
        to_utc(info["ts"]),
        f"{info['c']:g}",
        f"{info['body_ratio']:.2f}",
        f"{info['n_used']}",
        f"{kl[idx][2]:g}",
        f"{kl[idx][3]:g}",
        f"{info['bottom_wick_ticks']}",
        f"{info['tick']:g}",
        f"{info['candles_ago']}",
    ]
    print("# SELFTEST PASS expected CAKE-like signal")
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago")
    print(",".join(row))
    return 0

# ---------- Main ----------
def main():
    if args.selftest:
        exit(selftest())

    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        src = "symbols_file"
    else:
        universe = load_universe(args.api, args.quote)
        src = "mock_dir" if MOCK else ("exchangeInfo" if universe else "ticker/price")

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
