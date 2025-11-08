# --- imports near top ---
import requests, math, time
from collections import defaultdict

MAX_CANDLES_AGO = int(os.environ.get("MAX_CANDLES_AGO", "1"))   # default fresh
TIMEFRAME = "4h"  # scanner default per user
EXCHANGEINFO_URL = "https://api.mexc.com/api/v3/exchangeInfo"

def _parse_tick(filters):
    # mexc uses Binance-like filters sometimes priceFilter tickSize stepSize
    tick = None
    for f in filters or []:
        t = (f.get("filterType") or f.get("filter_type") or "").upper()
        if t in ("PRICE_FILTER","PRICEFILTER"):
            tick = f.get("tickSize") or f.get("tick_size")
            if tick is not None: 
                try: 
                    return float(tick)
                except: 
                    pass
    return 1e-6

def get_mexc_spot_usdt_universe():
    """
    Strict SPOT USDT symbols from exchangeInfo
    """
    try:
        r = requests.get(EXCHANGEINFO_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
        symbols = data.get("symbols") or data.get("data") or []
        out = []
        for s in symbols:
            sym = s.get("symbol") or s.get("symbol_name")
            if not sym or not sym.endswith("USDT"):
                continue
            status = (s.get("status") or s.get("state") or "").upper()
            if status not in ("TRADING","ENABLED","LISTING"):
                continue
            perms = set([p.upper() for p in (s.get("permissions") or [])])
            is_spot = ("SPOT" in perms) or bool(s.get("isSpotTradingAllowed", False)) or (s.get("type") == "SPOT")
            if not is_spot:
                continue
            tick = _parse_tick(s.get("filters"))
            out.append((sym, tick))
        if not out:
            raise RuntimeError("exchangeInfo returned zero USDT spot symbols")
        return dict(out)  # symbol -> tick
    except Exception as e:
        print(f"# exchangeInfo failed using strict USDT endswith fallback error={e}")
        # very last resort fallback only to symbols ending with USDT from ticker/price
        try:
            r = requests.get("https://api.mexc.com/api/v3/ticker/price", timeout=15)
            r.raise_for_status()
            data = r.json()
            uni = {d["symbol"]:1e-6 for d in data if d.get("symbol","").endswith("USDT")}
            return uni
        except Exception as e2:
            print(f"# fatal universe build error {e2}")
            return {}

def build_universe():
    """called by main scanning flow"""
    return get_mexc_spot_usdt_universe()

# later in your scan loop after computing each candidate row dict:
# row fields must include symbol time_utc close body n_used hi lo bottom_wick_ticks tick candles_ago

def passes_core_rules(candles, tick, quant):
    """
    Apply core rules
    - green full body body_ratio >= 0.70
    - bottom wick <= 1 tick
    - close > max(high of previous 15) OR > max(high of previous 20)
    - untouched rule default next_low only next candle low must not touch signal low
    Return (ok, n_used) where n_used is 15 or 20 that actually broke out
    """
    # assume latest candle is c0 and candlesticks list has needed fields
    c0 = candles[-1]
    body_ratio = c0["body_ratio"]
    if body_ratio < 0.70:
        return (False, None)
    # bottom wick in ticks
    bw_ticks = max(0, round((c0["open"] - c0["low"])/tick)) if c0["close"] >= c0["open"] else max(0, round((c0["close"] - c0["low"])/tick))
    if bw_ticks > 1:
        return (False, None)
    # breakout checks
    hi15 = max(x["high"] for x in candles[-16:-1])
    hi20 = max(x["high"] for x in candles[-21:-1])
    ok15 = c0["close"] > math.floor(hi15 / tick) * tick
    ok20 = c0["close"] > math.floor(hi20 / tick) * tick
    if not (ok15 or ok20):
        return (False, None)
    n_used = 15 if ok15 else 20

    # untouched rule next_low
    if len(candles) >= 2:
        nxt = candles[-2]  # next candle after signal in your indexing if you append oldest first adjust accordingly
        if nxt["low"] <= c0["low"]:
            return (False, None)
    return (True, n_used)

# after you collect candidates in a list named 'signals'
def apply_fresh_and_dedupe(signals):
    # filter by freshness
    fresh = [s for s in signals if int(s["candles_ago"]) <= MAX_CANDLES_AGO]
    # keep best per symbol prefer freshest then higher body
    best = {}
    for s in fresh:
        key = s["symbol"]
        if key not in best:
            best[key] = s; continue
        a, b = best[key], s
        if (int(b["candles_ago"]), -float(b["body"]), b["time_utc"]) < (int(a["candles_ago"]), -float(a["body"]), a["time_utc"]):
            best[key] = b
    return list(best.values())
