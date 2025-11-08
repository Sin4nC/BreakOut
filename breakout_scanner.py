# breakout_scanner.py
# MEXC spot USDT breakout scanner — resilient symbol discovery
import argparse, concurrent.futures as cf, datetime as dt, sys, time, math
from typing import Dict, List, Tuple, Optional
import requests

def fmt_time_utc(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M UTC")

def get(session: requests.Session, url: str, **params):
    r = session.get(url, params=params or None, timeout=30)
    r.raise_for_status()
    return r

def get_exchange_info(session: requests.Session, api: str) -> Dict:
    return get(session, f"{api}/api/v3/exchangeInfo").json()

def parse_symbols_from_exchange_info(ex_info: Dict) -> List[Dict]:
    out = []
    for s in ex_info.get("symbols", []):
        symbol = s.get("symbol", "")
        quote = s.get("quoteAsset", "")
        status = s.get("status", "TRADING")
        # شرط‌های مینیمال تا صفر نشود
        if not symbol or quote != "USDT":
            continue
        if status not in ("TRADING", "ENABLED"):
            continue
        tick = 1e-06
        for f in s.get("filters", []):
            if f.get("filterType") in ("PRICE_FILTER", "PRICE_FILTER_1", "PRICE_FILTER_2"):
                ts = f.get("tickSize")
                if ts:
                    try: tick = float(ts)
                    except: pass
        out.append({"symbol": symbol, "tick": tick})
    return out

def fallback_symbols_from_ticker(session: requests.Session, api: str) -> List[Dict]:
    data = get(session, f"{api}/api/v3/ticker/price").json()
    out = []
    for row in data if isinstance(data, list) else []:
        sym = row.get("symbol", "")
        if sym.endswith("USDT"):
            out.append({"symbol": sym, "tick": 1e-06})
    return out

def fallback_symbols_from_default(session: requests.Session, api: str) -> List[Dict]:
    # این اندپوینت گاهی رشته کاما جدا می‌دهد
    r = get(session, f"{api}/api/v3/defaultSymbols")
    try:
        payload = r.json()
        if isinstance(payload, dict):
            raw = payload.get("symbols", "")
        elif isinstance(payload, list):
            raw = ",".join(payload)
        else:
            raw = str(payload)
    except ValueError:
        raw = r.text
    toks = [t.strip().upper().replace('"', '') for t in raw.replace("[","").replace("]","").split(",")]
    out = []
    for t in toks:
        if t.endswith("USDT") and t:
            out.append({"symbol": t, "tick": 1e-06})
    return out

def build_universe(session: requests.Session, api: str) -> List[Dict]:
    # 1 exchangeInfo
    try:
        ex = get_exchange_info(session, api)
        syms = parse_symbols_from_exchange_info(ex)
        if syms:
            sys.stderr.write(f"# universe source=exchangeInfo symbols={len(syms)}\n")
            return syms
        else:
            sys.stderr.write("# exchangeInfo returned zero symbols for USDT, falling back\n")
    except Exception as e:
        sys.stderr.write(f"# exchangeInfo error {e}, falling back\n")

    # 2 ticker/price
    try:
        syms = fallback_symbols_from_ticker(session, api)
        if syms:
            sys.stderr.write(f"# universe source=ticker/price symbols={len(syms)}\n")
            return syms
    except Exception as e:
        sys.stderr.write(f"# ticker/price error {e}\n")

    # 3 defaultSymbols
    try:
        syms = fallback_symbols_from_default(session, api)
        sys.stderr.write(f"# universe source=defaultSymbols symbols={len(syms)}\n")
        return syms
    except Exception as e:
        sys.stderr.write(f"# defaultSymbols error {e}\n")
        return []

def body_ratio(o,h,l,c):
    tr = max(h - l, 1e-12)
    return (c - o) / tr if c > o else 0.0

def bottom_wick_ticks(o,l,tick,green):
    if not green: return 0
    return int(round(max(o - l, 0.0) / max(tick,1e-12)))

def close_above_prev_highs(candles, idx, lookback, close_value):
    start = max(0, idx - lookback)
    prev_highs = [c[2] for c in candles[start:idx]]
    return len(prev_highs) >= lookback and close_value > max(prev_highs)

def untouched_ok(candles, idx, rule):
    sig_h = candles[idx][2]; sig_l = candles[idx][3]
    if rule == "none": return True
    if idx + 1 >= len(candles): return True
    if rule == "next_low":
        return candles[idx+1][3] > sig_l
    if rule == "all_low":
        return all(c[3] > sig_l for c in candles[idx+1:])
    if rule == "all_highlow":
        return all((c[3] > sig_l and c[2] < sig_h) for c in candles[idx+1:])
    return True

def target_not_hit(candles, idx, target_pct, from_price=None):
    if not target_pct or target_pct <= 0: return True
    sig_close = candles[idx][4] if from_price is None else from_price
    tgt = sig_close * (1 + target_pct)
    return all(c[2] < tgt for c in candles[idx+1:])

def get_klines(session, api, symbol, interval, limit):
    data = get(session, f"{api}/api/v3/klines", symbol=symbol, interval=interval, limit=limit).json()
    out=[]
    for r in data:
        o=float(r[1]); h=float(r[2]); l=float(r[3]); c=float(r[4])
        ot=int(r[0]); ct=int(r[6])
        out.append((ot,o,h,l,c,ct))
    return out

def scan_symbol(session, api, rec, interval, window, lookbacks, min_body, untouched_rule, max_candles_ago, target_pct):
    sym = rec["symbol"]; tick = rec["tick"]
    max_lb = max(lookbacks); limit = window + max_lb + 5
    kl = get_klines(session, api, sym, interval, limit)
    if len(kl) < max_lb + 2: return []
    out=[]; last_idx=len(kl)-1; start_idx=max(0,last_idx - window + 1)
    for j in range(start_idx, len(kl)):
        ot,o,h,l,c,ct = kl[j]
        if c <= o: continue
        br = body_ratio(o,h,l,c)
        if br < min_body: continue
        n_used=None
        for n in lookbacks:
            if j-n < 0: continue
            if close_above_prev_highs(kl, j, n, c):
                n_used=n; break
        if n_used is None: continue
        if not untouched_ok(kl, j, untouched_rule): continue
        if target_pct and not target_not_hit(kl, j, target_pct): continue
        candles_ago = last_idx - j
        if candles_ago > max_candles_ago: continue
        bw = bottom_wick_ticks(o,l,tick,True)
        out.append(f"{sym},{fmt_time_utc(ct)},{c:.15g},{br:.2f},{n_used},{h:.15g},{l:.15g},{bw},{tick:.15g},{candles_ago}")
    return out

def main():
    ap = argparse.ArgumentParser(description="MEXC 4H breakout scanner")
    ap.add_argument("--api", default="https://api.mexc.com", help="API base url")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--window", type=int, default=45)
    ap.add_argument("--lookbacks", default="10,15")
    ap.add_argument("--min-body", type=float, default=0.70)
    ap.add_argument("--untouched", default="next_low", choices=["none","next_low","all_low","all_highlow"])
    ap.add_argument("--max-candles-ago", type=int, default=45)
    ap.add_argument("--target-pct", type=float, default=None)
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--max-workers", type=int, default=10)
    args = ap.parse_args()
    lookbacks = sorted({int(x) for x in args.lookbacks.split(",") if x.strip()})
    sess = requests.Session()

    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            syms=[{"symbol":ln.strip(),"tick":1e-06} for ln in f if ln.strip()]
        sys.stderr.write(f"# universe source=file symbols={len(syms)}\n")
    else:
        syms = build_universe(sess, args.api)

    print("symbol,time_utc,close,body,n_used,hi,lo,bottom_wick_ticks,tick,candles_ago")
    scanned=hits=0
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for lines in ex.map(lambda rec: scan_symbol(sess, args.api, rec, args.interval, args.window, lookbacks, args.min_body, args.untouched, args.max_candles_ago, args.target_pct), syms, chunksize=20):
            scanned += 1
            if not lines: continue
            hits += len(lines)
            for ln in lines: print(ln)
    sys.stderr.write(f"# scanned={scanned} hits={hits} window={args.window} max_candles_ago={args.max_candles_ago}\n")

if __name__ == "__main__":
    main()
