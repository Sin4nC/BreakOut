# --- tick size from MEXC + precise wick math ---
import math, requests

def load_mexc_universe_and_ticks():
    j = requests.get("https://api.mexc.com/api/v3/exchangeInfo", timeout=15).json()
    ticks = {}
    universe = []
    for s in j["symbols"]:
        if s.get("quoteAsset") != "USDT": 
            continue
        if not s.get("spotTradingAllowed", True):
            continue
        sym = s["symbol"]
        tick = None
        for f in s.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick = float(f["tickSize"])
                break
        if tick:
            ticks[sym] = tick
            universe.append(sym)
    return universe, ticks

def tick_floor(x, tick):
    return math.floor(x / tick) * tick

def wick_ticks(low, open_, close, tick):
    # use quantized values to avoid شبه اعشار
    low_q   = tick_floor(low,   tick)
    open_q  = tick_floor(open_, tick)
    close_q = tick_floor(close, tick)
    bottom  = min(open_q, close_q) - low_q
    return int(round(bottom / tick))

def is_full_body_green(o, h, l, c, min_body_ratio=0.70):
    if c <= o:
        return False, 0.0
    rng = max(h - l, 1e-12)
    body = c - o
    br = body / rng
    return br >= min_body_ratio, br

def broke_prev_high(h_arr, i, lookbacks=(15, 20)):
    # close of i must be above max high of previous N for any N in lookbacks
    ok_any = False
    for N in lookbacks:
        if i - N < 0:
            continue
        prev_hi = max(h_arr[i - N:i])
        if h_arr[i] > prev_hi:
            ok_any = True
            break
    return ok_any

def untouched_next_low(l_arr, i):
    # only کندل بعدی نباید کف سیگنال را لمس کند
    if i + 1 >= len(l_arr):
        return True
    return l_arr[i + 1] > l_arr[i]

def target_hit_in_post_window(h_arr, i, target_pct=0.05, post_window=1, from_high=True, hi_i=None, close_i=None):
    base = hi_i if from_high else close_i
    tgt  = base * (1.0 + target_pct)
    end  = min(len(h_arr) - 1, i + post_window)
    return max(h_arr[i + 1:end + 1]) >= tgt
