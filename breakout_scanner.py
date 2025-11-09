# breakout_scanner.py
# MEXC 4H breakout scanner — USDT spot only, full-body green, strict breakout, untouched low, suppression rule

import argparse, concurrent.futures as cf, math, time, sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import requests

# ---------- CLI ----------
p = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
p.add_argument("--api", default="https://api.mexc.com")
p.add_argument("--interval", default="4h")
p.add_argument("--window", type=int, default=180)                    # recent search window
p.add_argument("--lookbacks", default="15,20")                       # N set
p.add_argument("--min-body", type=float, default=0.70)               # full-body threshold
p.add_argument("--bottom-wick-max-ticks", type=int, default=1)       # <= 1 tick
p.add_argument("--untouched", choices=["all","next","none"], default="all")
p.add_argument("--freshness-max", type=int, default=-1)              # -1 means OFF
p.add_argument("--no-suppress", action="store_true")                 # disable suppression
p.add_argument("--symbols-file", default=None)
p.add_argument("--workers", type=int, default=12)
p.add_argument("--sleep", type=float, default=0.20)
p.add_argument("--quote", default="USDT")
p.add_argument("--min-atr-ticks", type=int, default=3)               # ignore dead charts
args = p.parse_args()

LOOKBACKS = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.0"})
DEFAULT_TICK = 1e-6

# ---------- helpers ----------
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
    return r

def floor_to_tick(x: float, tick: float) -> float:
    if tick and tick > 0:
        return math.floor(x / tick) * tick
    return x

def last_closed_index(kl) -> int:
    """Return index of the last CLOSED candle (guard against a still-forming last bar)."""
    now_ms = int(time.time() * 1000)
    i = len(kl) - 1
    while i >= 0 and kl[i][6] > now_ms:
        i -= 1
    return i

# ---------- universe ----------
def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()
        for s in ei.get("symbols", []):
            sym = s.get("symbol") or s.get("symbolName")
            if not sym:
                continue
            if s.get("quoteAsset") != quote:
                continue
            status = s.get("status", "TRADING")
            if status not in ("TRADING", "ENABLED"):
                continue
            perms = set(s.get("permissions", []) or s.get("permissionList", []) or [])
            spot_allowed = True if not perms else ("SPOT" in perms)
            if not spot_allowed:
                continue
            tick = DEFAULT_TICK
            for f in s.get("filters", []):
                if f.get("filterType") in ("PRICE_FILTER", "price_filter"):
                    ts = f.get("tickSize") or f.get("tick_size")
                    if ts:
                        try:
                            tick = float(ts)
                        except:
                            pass
            out.append((sym, tick))
    except Exception:
        pass
    # fallback guarded by quote filter
    if not out:
        tp = http_get(f"{api}/api/v3/ticker/price").json()
        symbols = sorted({row["symbol"] for row in tp if str(row.get("symbol","")).endswith(quote)})
        out = [(s, DEFAULT_TICK) for s in symbols]
    return sorted(out)

def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s:
                res.append((s, DEFAULT_TICK))
    return res

# ---------- data ----------
def get_klines(symbol: str, interval: str, limit: int):
    time.sleep(args.sleep)
    data = http_get(f"{args.api}/api/v3/klines", {
        "symbol": symbol, "interval": interval, "limit": limit
    }).json()
    kl = []
    for r in data:
        ts = int(r[0]); o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4])
        v = float(r[5]); ct = int(r[6])
        kl.append((ts, o, h, l, c, v, ct))
    return kl

def to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------- logic ----------
def atr_ticks(kl, tick: float, look: int = 10) -> float:
    last = last_closed_index(kl)
    if last < 0 or last < look:
        return 0.0
    rngs = [abs(kl[i][2] - kl[i][3]) for i in range(last - look + 1, last + 1)]
    atr = sum(rngs) / look
    if tick <= 0:
        tick = DEFAULT_TICK
    return atr / tick

def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    last_need = max(LOOKBACKS) + 5
    limit = max(args.window + last_need, last_need + 5)
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 2:
        return None

    # dead chart filter
    if atr_ticks(kl, tick, 10) < args.min_atr_ticks:
        return None

    last = last_closed_index(kl)                                  # last CLOSED candle
    if last < 0:
        return None
    start = max(last - args.window, max(LOOKBACKS))

    for idx in range(last, start - 1, -1):
        ts, o, h, l, c, v, ct = kl[idx]
        rng = h - l
        if rng <= 0:
            continue
        if c <= o:
            continue

        body = (c - o) / rng
        if body < args.min_body:
            continue

        # bottom wick ticks (use tick quantization)
        tick_size = tick if tick and tick > 0 else DEFAULT_TICK
        bottom_wick = max(0.0, min(o, c) - l)
        bottom_wick_ticks = int(round(bottom_wick / tick_size))
        if bottom_wick_ticks > args.bottom_wick_max_ticks:
            continue

        # strict breakout using tick floor
        passed_N = []
        for N in LOOKBACKS:
            if idx - N < 0:
                continue
            prev_high = max(x[2] for x in kl[idx - N: idx])
            if floor_to_tick(c, tick_size) > floor_to_tick(prev_high, tick_size):
                passed_N.append(N)
        if not passed_N:
            continue
        n_used = min(passed_N)

        # freshness optional
        candles_ago = last - idx
        if args.freshness_max >= 0 and candles_ago > args.freshness_max:
            break

        # untouched rule
        if args.untouched == "next":
            if idx + 1 <= last and kl[idx + 1][3] <= l:
                continue
        elif args.untouched == "all":
            if any(kl[j][3] <= l for j in range(idx + 1, last + 1)):
                continue

        # suppression rule: keep only the latest surviving signal
        if not args.no_suppress:
            if any(kl[j][2] > h for j in range(idx + 1, last + 1)):
                continue

        row = [
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{body:.2f}",
            f"{n_used}",
            f"{h:g}",
            f"{l:g}",
            f"{bottom_wick_ticks}",
            f"{tick_size:g}",
            f"{candles_ago}",
        ]
        return ",".join(row)

    return None

# ---------- main ----------
def main():
    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        source_note = "file"
    else:
        universe = load_universe(args.api, args.quote)
        source_note = "exchangeInfo"
    print(f"# universe source={source_note} symbols={len(universe)}")
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago")
    if not universe:
        return
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(scan_symbol, universe, chunksize=32):
            if res:
                print(res, flush=True)

if __name__ == "__main__":
    main()
