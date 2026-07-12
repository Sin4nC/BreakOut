# binance_spot_4h_bi_directional_scanner.py
# Binance SPOT 4H Breakout Scanner (Bullish & Bearish) with Telegram Alerts

import argparse, concurrent.futures as cf, time, math, requests, os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

# ----- fixed rules -----
LOOKBACKS = [15, 20]           # pass if breakout happens beyond last 15 OR 20 closed candles
MIN_BODY = 0.70                # full-body candle ratio
MAX_WICK_TICKS = 150           # max bottom wick for Bullish / max top wick for Bearish
SUPPRESSION = False            
DEFAULT_TICK = 1e-8

# ----- CLI -----
ap = argparse.ArgumentParser("Binance Spot 4H Bi-Directional Scanner")
ap.add_argument("--api", default="https://api.binance.com")
ap.add_argument("--interval", default="4h")          
ap.add_argument("--window", type=int, default=30)     
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--symbols-file", default=None)      
ap.add_argument("--sleep", type=float, default=0.1)
args = ap.parse_args()

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "breakout-scanner/binance-spot-2.0"})

# ----- telegram sender -----
def send_telegram_message(text: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("# Telegram environment variables are missing.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
        r.raise_for_status()
    except Exception as e:
        print(f"# Failed to send Telegram alert: {e}")

# ----- helpers -----
def http_get(url: str, params: Dict = None, max_retries: int = 5):
    params = params or {}
    back = max(args.sleep, 0.05)
    for _ in range(max_retries):
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(back); back = min(back * 2.0, 5.0); continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r

def qfloor(x: float, tick: float) -> float:
    t = tick if tick and tick > 0 else DEFAULT_TICK
    return math.floor(x / t) * t

def to_utc(sec: int) -> str:
    return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ----- universe -----
def load_universe_from_file(path: str) -> List[Tuple[str, float]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if not s or not s.endswith("USDT"):
                continue
            out.append((s, DEFAULT_TICK))
    return out

def load_universe_from_api(api: str) -> List[Tuple[str, float]]:
    res = http_get(f"{api}/api/v3/exchangeInfo").json()
    symbols_data = res.get("symbols") or []
    out: List[Tuple[str, float]] = []
    
    for s in symbols_data:
        try:
            if s.get("status") != "TRADING":
                continue
            sym = s.get("symbol", "").upper()
            if not sym.endswith("USDT"):
                continue
                
            tick = DEFAULT_TICK
            for f in s.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    tick = float(f.get("tickSize") or DEFAULT_TICK)
                    break
            out.append((sym, tick))
        except:
            continue
    return sorted(out)

# ----- data -----
def get_klines(symbol: str, interval: str, limit: int):
    time.sleep(args.sleep)
    res = http_get(f"{args.api}/api/v3/klines", {
        "symbol": symbol, "interval": interval, "limit": limit
    }).json()
    
    kl = []
    for item in res:
        try:
            ts = int(item[0]) // 1000  
            oo = float(item[1])
            hh = float(item[2])
            ll = float(item[3])
            cc = float(item[4])
            kl.append((ts, oo, hh, ll, cc))
        except:
            continue
    return kl

# ----- rules -----
def passes_rules(kl, idx: int, tick: float, last: int) -> Optional[Tuple[str, float, int, int, int]]:
    ts, o, h, l, c = kl[idx]
    rng = h - l
    if rng <= 0:
        return None

    denom = tick if tick and tick > 0 else DEFAULT_TICK
    close_q = qfloor(c, denom)

    # ------ 1. BULLISH CHECK (Green Breakout) ------
    if c > o:
        body_ratio = (c - o) / rng
        if body_ratio < MIN_BODY:
            return None

        bottom_wick = max(0.0, o - l)
        bottom_wick_ticks = int(math.floor(bottom_wick / denom + 1e-12))
        if bottom_wick_ticks > MAX_WICK_TICKS:
            return None

        passed_N = []
        for N in LOOKBACKS:
            if idx - N < 0:
                continue
            prev_high = max(x[2] for x in kl[idx - N: idx])
            if close_q > qfloor(prev_high, denom):
                passed_N.append(N)
        if not passed_N:
            return None
        n_used = min(passed_N)

        # Protection check: Low must not be breached by later candle lows
        sig_low_q = qfloor(l, denom)
        for j in range(idx + 1, last + 1):
            if qfloor(kl[j][3], denom) <= sig_low_q:
                return None

        if SUPPRESSION:
            for j in range(idx + 1, last + 1):
                if kl[j][2] > h:
                    return None

        candles_ago = last - idx
        return "BULLISH", body_ratio, bottom_wick_ticks, n_used, candles_ago

    # ------ 2. BEARISH CHECK (Red Breakout) ------
    elif c < o:
        body_ratio = (o - c) / rng
        if body_ratio < MIN_BODY:
            return None

        top_wick = max(0.0, h - o)
        top_wick_ticks = int(math.floor(top_wick / denom + 1e-12))
        if top_wick_ticks > MAX_WICK_TICKS:
            return None

        passed_N = []
        for N in LOOKBACKS:
            if idx - N < 0:
                continue
            prev_low = min(x[3] for x in kl[idx - N: idx])
            if close_q < qfloor(prev_low, denom):
                passed_N.append(N)
        if not passed_N:
            return None
        n_used = min(passed_N)

        # Protection check: High must not be breached by later candle highs
        sig_high_q = qfloor(h, denom)
        for j in range(idx + 1, last + 1):
            if qfloor(kl[j][2], denom) >= sig_high_q:
                return None

        if SUPPRESSION:
            for j in range(idx + 1, last + 1):
                if kl[j][3] < l:
                    return None

        candles_ago = last - idx
        return "BEARISH", body_ratio, top_wick_ticks, n_used, candles_ago

    return None

def scan_symbol(rec: Tuple[str, float]) -> Optional[str]:
    symbol, tick = rec
    limit = max(args.window + max(LOOKBACKS) + 5, 100)
    try:
        kl = get_klines(symbol, args.interval, limit)
    except Exception:
        return None
    if len(kl) < max(LOOKBACKS) + 5:
        return None

    last = len(kl) - 2 
    if last < max(LOOKBACKS):
        return None
    start = max(last - args.window + 1, max(LOOKBACKS))

    for idx in range(last, start - 1, -1):
        res = passes_rules(kl, idx, tick if tick > 0 else DEFAULT_TICK, last)
        if res is None:
            continue
        direction, body_ratio, wick_ticks, n_used, candles_ago = res
        ts, o, h, l, c = kl[idx]
        return ",".join([
            symbol,
            to_utc(ts),
            f"{c:g}",
            f"{body_ratio:.2f}",
            f"{n_used}",
            f"{h:g}",
            f"{l:g}",
            f"{wick_ticks}",
            f"{(tick if tick and tick > 0 else DEFAULT_TICK):g}",
            f"{candles_ago}",
            direction
        ])
    return None

# ----- main -----
def main():
    if args.symbols_file:
        universe = load_universe_from_file(args.symbols_file)
        src = "symbols_file"
    else:
        universe = load_universe_from_api(args.api)
        src = "binance_spot_api"

    print(f"# universe source={src} symbols={len(universe)} suppression={SUPPRESSION} interval={args.interval}")
    print("symbol,signal_utc,close,body_ratio,lookbackN,high,low,wick_ticks,tick_size,candles_ago,direction")

    if not universe:
        return

    signals = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(scan_symbol, universe, chunksize=32):
            if out:
                print(out, flush=True)
                signals.append(out)

    if signals:
        alert_text = "🚨 *Binance Breakout Alert (4H)* 🚨\n\n"
        for sig in signals:
            parts = sig.split(",")
            direction = parts[10]
            icon = "🟢" if direction == "BULLISH" else "🔴"
            
            alert_text += f"{icon} *{parts[0]}* ({direction})\n"
            alert_text += f"  • Time: {parts[1]}\n"
            alert_text += f"  • Close: {parts[2]}\n"
            alert_text += f"  • Lookback: {parts[4]} candles\n"
            alert_text += f"  • Candles Ago: {parts[9]}\n\n"
        send_telegram_message(alert_text)

if __name__ == "__main__":
    main()
