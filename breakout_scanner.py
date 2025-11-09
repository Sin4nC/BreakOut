# breakout_scanner.py
# MEXC 4H breakout scanner — latest valid signal per symbol
#
# پیشفرض های دقیقا مطابق خواسته تو
# 1) کندل سبز فول بادی با body_ratio >= 0.70
# 2) کلوز بزرگتر از بیشترین HIGH بین 15 یا 20 کندل بسته قبلی  هر کدام که پاس شد قبول است
# 3) قانون untouched = همهٔ کندل های بسته بعد از سیگنال نباید LOW کندل سیگنال را لمس یا بشکنند
# 4) سرکوب خاموش به صورت پیشفرض  قدیمی را فقط وقتی کنار میگذاریم که خودت روشنش کنی
# 5) کف سایه پایین حداکثر 1 تیک  میشه تغییرش داد
# 6) فیلتر تازگی خاموش  هر سیگنال معتبر در پنجره جستجو قابل گزارش است
#
# خروجی یک ردیف برای هر نماد  فقط آخرین سیگنال معتبر همان نماد
# ستون ها
# symbol,signal_utc,close,body_ratio,lookbackN,high,low,bottom_wick_ticks,tick_size,candles_ago

import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import time
import requests

# ---------- CLI ----------
ap = argparse.ArgumentParser("MEXC 4H Breakout Scanner")
ap.add_argument("--api", default="https://api.mexc.com", help="MEXC REST base URL")
ap.add_argument("--interval", default="4h", help="kline interval")
ap.add_argument("--window", type=int, default=180, help="تعداد کندل های بسته برای جستجو به عقب")
ap.add_argument("--lookbacks", default="15,20", help="لیست lookback ها با کاما")
ap.add_argument("--min-body", type=float, default=0.70, help="حداقل نسبت فول بادی")
ap.add_argument("--max-bottom-wick-ticks", type=int, default=1, help="حداکثر سایه پایین بر حسب تیک")
ap.add_argument("--untouched", choices=["all", "next"], default="all",
                help="all = همهٔ کندل های بسته بعدی نباید LOW را لمس کنند  next = فقط کندل بعدی")
ap.add_argument("--max-candles-ago", type=int, default=-1,
                help="-1 یعنی خاموش  در غیر اینصورت فقط سیگنال با candles_ago <= مقدار")
ap.add_argument("--suppress", action="store_true",
                help="اگر ست شود سرکوب روشن میشود  هر سیگنال قدیمی که high کوچکتری از یک کندل بعدی داشته باشد حذف میشود")
ap.add_argument("--symbols-file", default=None, help="فایل لیست نمادها  هر خط یک نماد")
ap.add_argument("--quote", default="USDT", help="فیلتر یونیورس بر اساس USDT")
ap.add_argument("--workers", type=int, default=12, help="تعداد تردها")
ap.add_argument("--sleep", type=float, default=0.16, help="تاخیر بین درخواست ها برای جلوگیری از 429")
args = ap.parse_args()

LOOKBACKS: List[int] = [int(x) for x in args.lookbacks.split(",") if x.strip()]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/1.3"})
DEFAULT_TICK = 1e-6
SUPPRESS = bool(args.suppress)  # پیشفرض خاموش است

# ---------- HTTP ----------
def http_get(url: str, params: Dict = None, max_retries: int = 6) -> requests.Response:
    backoff = max(args.sleep, 0.05)
    params = params or {}
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
            sym = (s.get("symbol") or "").upper()
            if not sym.endswith(quote):
                continue
            status = (s.get("status") or "TRADING").upper()
            if status not in ("TRADING", "ENABLED"):
                continue
            perms = [p.upper() for p in (s.get("permissions") or s.get("permissionList") or [])]
            if perms and all(p not in ("SPOT", "SPOT_TRADING") for p in perms):
                continue
            tick = DEFAULT_TICK
            for f in s.get("filters", []):
                if str(f.get("filterType") or "").upper() == "PRICE_FILTER":
                    ts = f.get("tickSize")
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
    return out

def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s:
                res.append((s, DEFAULT_TICK))
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
    while idx >= 0 and kl[idx][5] > now_ms:
        idx -= 1
    return idx

def to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ---------- Logic ----------
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

    # از جدید به قدیم  اولین سیگنال معتبر را برمیگردانیم
    for idx in range(last, start - 1, -1):
        ts, o, h, l, c, _ = kl[idx]

        rng = h - l
        if rng <= 0:
            continue
        if c <= o:
            continue  # سبز الزام

        body_ratio = (c - o) / rng
        if body_ratio < args.min_body:
            continue

        # سایه پایین بر حسب تیک  بدون کوانتایز کردن قیمت ها
        tick_size = tick if (tick and tick > 0) else DEFAULT_TICK
        bottom_wick = max(0.0, min(o, c) - l)
        bottom_wick_ticks = int(round(bottom_wick / tick_size))
        if bottom_wick_ticks > args.max_bottom_wick_ticks:
            continue

        # شکست  کلوز باید بزرگتر از بیشترین HIGH بین N کندل بسته قبلی باشد
        passed_N: List[int] = []
        for N in LOOKBACKS:
            if idx - N < 0:
                continue
            prev_high = max(kl[idx - N: idx], key=lambda x: x[2])[2]
            if c > prev_high:
                passed_N.append(N)
        if not passed_N:
            continue
        n_used = min(passed_N)

        # تازگی اختیاری
        candles_ago = last - idx
        if args.max_candles_ago >= 0 and candles_ago > args.max_candles_ago:
            break

        # untouched rule
        if args.untouched == "next":
            if idx + 1 <= last and kl[idx + 1][3] <= l:
                continue
        else:
            if any(kl[j][3] <= l for j in range(idx + 1, last + 1)):
                continue

        # suppression اختیاری
        if SUPPRESS and any(kl[j][2] > h for j in range(idx + 1, last + 1)):
            continue

        row = [
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{body_ratio:.2f}",
            f"{n_used}",
            f"{h:g}",
            f"{l:g}",
            f"{bottom_wick_ticks}",
            f"{tick_size:g}",
            f"{candles_ago}",
        ]
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
