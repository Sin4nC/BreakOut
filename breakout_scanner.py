# breakout_scanner.py
# MEXC Spot USDT 4H full-body momentum scanner — shape only, no breakout rules (Forz4crypto v4.0)
#
# هدف: پیدا کردن آخرین کندل‌های 4H سبز با بدنه خیلی بزرگ و ویک‌های خیلی کوچک،
# در بازه N کندل قبلی (پارامتر --window)، روی تمام جفت‌های اسپات USDT در MEXC.
#
# Rules (fixed defaults)
# 1) Green body: close > open
# 2) Full body:
#       body_ratio = (close - open) / (high - low)  >= 0.80
# 3) Small wicks relative to full range:
#       bottom_wick_ratio = (min(open, close) - low) / (high - low) <= 0.20
#       top_wick_ratio    = (high - max(open, close)) / (high - low) <= 0.20
# 4) Meaningful move up:
#       body_pct = (close - open) / open >= 0.03   (حداقل +۳٪ روی کندل)
# 5) فقط کندل‌های بسته شده بررسی می‌شوند (هیچ کندل در حال تشکیل پذیرفته نمی‌شود)
#
# خروجی: برای هر سیمبل حداکثر یک ردیف، آخرین کندل 4H که همه شرایط بالا را پاس کرده.
#
# CSV header:
# symbol,signal_utc,close,body_ratio,body_pct,high,low,bottom_wick_ticks,tick_size,candles_ago

import argparse
import concurrent.futures as cf
import time
import math
import requests
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

# ---------- CLI ----------
ap = argparse.ArgumentParser("MEXC 4H Full-Body Momentum Scanner")
ap.add_argument("--api", default="https://api.mexc.com")
ap.add_argument("--interval", default="4h")
ap.add_argument(
    "--window",
    type=int,
    default=180,     # how many CLOSED candles to scan backward
)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--symbols-file", default=None)
ap.add_argument("--quote", default="USDT")
ap.add_argument("--sleep", type=float, default=0.16)
ap.add_argument("--debug-symbol", default=None)        # optional single symbol debug
args = ap.parse_args()

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fullbody-scanner/forz4crypto-4.0"})

MIN_BODY_RATIO = 0.80            # حداقل ۸۰٪ رنج باید بدنه باشد
MAX_BOTTOM_WICK_RATIO = 0.20     # حداکثر ۲۰٪ رنج برای ویک پایین
MAX_TOP_WICK_RATIO = 0.20        # حداکثر ۲۰٪ رنج برای ویک بالا
MIN_BODY_PCT = 0.03              # حداقل +۳٪ حرکت رو به بالا روی بدنه نسبت به اوپن
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
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def last_closed_index(kl) -> int:
    """
    آخرین کندل *بسته شده* را بر اساس close time نسبت به now پیدا می‌کند.
    """
    now_ms = int(time.time() * 1000)
    i = len(kl) - 1
    while i >= 0 and kl[i][5] > now_ms:
        i -= 1
    return i


# ---------- universe ----------
def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    """
    Optional file format:
      SYMBOL
      or
      SYMBOL, tick_size
    Lines starting with # are ignored.
    """
    out: List[Tuple[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p.strip() for p in raw.split(",")]
            sym = parts[0].upper()
            tick = DEFAULT_TICK
            if len(parts) >= 2:
                try:
                    tick = float(parts[1])
                except Exception:
                    pass
            out.append((sym, tick))
    return out


def _extract_tick_from_filters(filters) -> Optional[float]:
    if not isinstance(filters, list):
        return None
    for f in filters:
        try:
            ft = str(f.get("filterType") or f.get("filter_type") or "").upper()
        except AttributeError:
            continue
        if "PRICE" not in ft:
            continue
        ts = f.get("tickSize") or f.get("tick_size")
        if ts is None:
            continue
        try:
            val = float(ts)
            if val > 0:
                return val
        except Exception:
            continue
    return None


def _extract_tick_from_precisions(sym_info) -> Optional[float]:
    # Fallback if filters do not expose a tick size.
    for key in ("quotePrecision", "quoteAssetPrecision"):
        if key in sym_info:
            try:
                prec = int(sym_info[key])
                if 0 < prec < 12:
                    return 10.0 ** (-prec)
            except Exception:
                continue
    return None


def load_universe(api: str, quote: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    ticks: Dict[str, float] = {}
    quote = quote.upper()

    try:
        ei = http_get(f"{api}/api/v3/exchangeInfo").json()

        if isinstance(ei, dict):
            symbols_raw = ei.get("symbols")
            if symbols_raw is None and "symbol" in ei and "baseAsset" in ei:
                symbols_raw = [ei]
        elif isinstance(ei, list):
            symbols_raw = ei
        else:
            symbols_raw = []

        for s in symbols_raw or []:
            try:
                sym = (s.get("symbol") or s.get("symbolName") or "").upper().strip()
            except AttributeError:
                continue
            if not sym or not sym.endswith(quote):
                continue

            # spot only via permissions
            perms = s.get("permissions")
            if isinstance(perms, list):
                if "SPOT" not in {str(p).upper() for p in perms}:
                    continue
            elif isinstance(perms, str):
                if "SPOT" not in perms.upper():
                    continue

            if s.get("isSpotTradingAllowed") is False:
                continue

            st_flag = s.get("st")
            if isinstance(st_flag, bool) and st_flag:
                continue

            tick = (
                _extract_tick_from_filters(s.get("filters"))
                or _extract_tick_from_precisions(s)
                or DEFAULT_TICK
            )
            ticks[sym] = tick

        if ticks:
            out = sorted(ticks.items())

    except Exception:
        pass

    if not out:
        # Fallback: approximate tick from current prices.
        try:
            tp = http_get(f"{api}/api/v3/ticker/price").json()
            seen = set()
            approx: List[Tuple[str, float]] = []
            for row in tp:
                sym = str(row.get("symbol") or "").upper()
                if not sym or not sym.endswith(quote):
                    continue
                if sym in seen:
                    continue
                price_str = str(row.get("price") or "")
                tick = DEFAULT_TICK
                if "." in price_str:
                    frac = price_str.split(".", 1)[1].rstrip("0")
                    if frac:
                        dec = len(frac)
                        if 0 < dec < 12:
                            tick = 10.0 ** (-dec)
                seen.add(sym)
                approx.append((sym, tick))
            approx.sort()
            out = approx
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
            ot = int(r[0])
            o = float(r[1])
            h = float(r[2])
            l = float(r[3])
            c = float(r[4])
            ct = int(r[6])
            kl.append((ot, o, h, l, c, ct))
        except Exception:
            pass
    return kl


# ---------- logic ----------
def passes_fixed_rules(
    kl,
    idx: int,
    tick: float,
) -> Optional[Tuple[float, float, float, int]]:
    """
    Returns (body_ratio, bottom_wick_ticks, body_pct, candles_ago) if all rules pass.
    Otherwise returns None.
    """
    ts, o, h, l, c, _ = kl[idx]
    rng = h - l
    if rng <= 0:
        return None
    if o <= 0:
        return None
    if c <= o:
        return None  # must be green

    body = c - o
    body_ratio = body / rng
    if body_ratio < MIN_BODY_RATIO:
        return None

    body_pct = body / o
    if body_pct < MIN_BODY_PCT:
        return None

    denom = tick if tick and tick > 0 else DEFAULT_TICK

    bottom_wick = max(0.0, min(o, c) - l)
    top_wick = max(0.0, h - max(o, c))

    bottom_wick_ratio = bottom_wick / rng
    top_wick_ratio = top_wick / rng

    if bottom_wick_ratio > MAX_BOTTOM_WICK_RATIO:
        return None
    if top_wick_ratio > MAX_TOP_WICK_RATIO:
        return None

    bottom_wick_ticks = int(math.floor(bottom_wick / denom + 1e-12))

    last = last_closed_index(kl)
    candles_ago = last - idx
    return body_ratio, bottom_wick_ticks, body_pct, candles_ago


def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    limit = max(args.window + 8, 120)
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < 5:
        return None

    last = last_closed_index(kl)
    if last < 0:
        return None
    start = max(last - args.window + 1, 0)

    denom_tick = tick if tick and tick > 0 else DEFAULT_TICK

    # newest to older: pick first that survives → latest full-body momentum candle
    for idx in range(last, start - 1, -1):
        res = passes_fixed_rules(kl, idx, denom_tick)
        if res is None:
            continue
        body_ratio, bottom_wick_ticks, body_pct, candles_ago = res
        ts, o, h, l, c, _ = kl[idx]
        row = [
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{body_ratio:.2f}",
            f"{body_pct:.4f}",
            f"{h:g}",
            f"{l:g}",
            f"{bottom_wick_ticks}",
            f"{denom_tick:g}",
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
    print(
        "symbol,signal_utc,close,body_ratio,body_pct,high,low,"
        "bottom_wick_ticks,tick_size,candles_ago"
    )
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
