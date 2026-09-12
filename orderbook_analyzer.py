#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ стакана Binance: стенки ордеров, дисбаланс, исчезновение/появление
стенок, «намерение» разворота по стакану.

Данные через REST GET /api/v3/depth с кэшем CACHE_TTL секунд.
Новых зависимостей не требует.
"""
import time
import threading
import numpy as np

import crypto_zone_screener as czs

DEPTH_LIMIT = 100            # уровней стакана
WALL_DENSITY = 4.0           # объём уровня / медиана по стакану
WALL_MAX_DIST_PCT = 2.0      # макс. расстояние от цены, %
WALL_MIN_USD = 250_000       # минимальный размер стенки, $
IMBALANCE_WINDOW_PCT = 1.0   # окно дисбаланса bid/ask, %
CACHE_TTL = 5                # кэш стакана, сек
HISTORY_LEN = 20             # снапшотов для динамики стенок

_cache = {}
_lock = threading.Lock()


def fetch_depth(symbol):
    data = czs.get_json(f"{czs.BASE_URL}/api/v3/depth",
                        params={"symbol": symbol, "limit": DEPTH_LIMIT})
    if not data:
        return None
    bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
    if not bids or not asks:
        return None
    mid = (bids[0][0] + asks[0][0]) / 2
    return {"bids": bids, "asks": asks, "mid": mid, "ts": time.time()}


def get_depth(symbol):
    with _lock:
        item = _cache.get(symbol)
        if item and time.time() - item["ts"] < CACHE_TTL:
            return item
    depth = fetch_depth(symbol)
    if depth:
        with _lock:
            _cache[symbol] = depth
    return depth


def find_walls(symbol, depth=None):
    """(bid_walls, ask_walls): плотность >= WALL_DENSITY, дистанция <= 2%, объём >= WALL_MIN_USD."""
    depth = depth or get_depth(symbol)
    if not depth:
        return [], []
    mid = depth["mid"]
    bid_usd = [(p, q, p * q) for p, q in depth["bids"]]
    ask_usd = [(p, q, p * q) for p, q in depth["asks"]]
    all_usd = [u for _, _, u in bid_usd + ask_usd]
    med = float(np.median(all_usd)) if all_usd else 0.0
    if med <= 0:
        return [], []

    def pick(levels, side):
        walls = []
        for p, _q, u in levels:
            dist = abs(p - mid) / mid * 100
            density = u / med
            if density >= WALL_DENSITY and dist <= WALL_MAX_DIST_PCT and u >= WALL_MIN_USD:
                walls.append({"side": side, "price": p, "usd": u,
                              "dist_pct": dist, "density": density})
        walls.sort(key=lambda w: w["dist_pct"])
        return walls[:3]

    return pick(bid_usd, "bid"), pick(ask_usd, "ask")


def imbalance(symbol, depth=None):
    """Отношение bid/ask по объёму в $ в окне ±1% от цены."""
    depth = depth or get_depth(symbol)
    if not depth:
        return 1.0
    mid = depth["mid"]
    lo = mid * (1 - IMBALANCE_WINDOW_PCT / 100)
    hi = mid * (1 + IMBALANCE_WINDOW_PCT / 100)
    bid_sum = sum(p * q for p, q in depth["bids"] if p >= lo)
    ask_sum = sum(p * q for p, q in depth["asks"] if p <= hi)
    if ask_sum <= 0:
        return 9.9
    return round(bid_sum / ask_sum, 2)


class WallTracker:
    """Динамика стенок: исчезновения и появления (по каждому символу)."""

    def __init__(self, max_history=HISTORY_LEN):
        self.history = {}
        self.max_history = max_history

    def update(self, symbol):
        depth = get_depth(symbol)
        if not depth:
            return []
        bid_walls, ask_walls = find_walls(symbol, depth)
        now = time.time()
        st = self.history.setdefault(symbol, {"snap": [], "events": []})
        cur = {(w["side"], round(w["price"], 6)) for w in bid_walls + ask_walls}
        prev = st["snap"][-1]["walls"] if st["snap"] else set()

        events = []
        for side, price in prev - cur:
            events.append({"ts": now, "type": "исчезла", "side": side, "price": price})
        for side, price in cur - prev:
            w = next((w for w in bid_walls + ask_walls
                      if (w["side"], round(w["price"], 6)) == (side, price)), None)
            events.append({"ts": now, "type": "появилась", "side": side,
                           "price": price, "usd": w["usd"] if w else 0})

        st["snap"].append({"ts": now, "walls": cur})
        if len(st["snap"]) > self.max_history:
            st["snap"].pop(0)
        st["events"].extend(events)
        st["events"] = st["events"][-50:]
        return events

    def recent_events(self, symbol, seconds=60):
        st = self.history.get(symbol)
        if not st:
            return []
        now = time.time()
        return [e for e in st["events"] if now - e["ts"] <= seconds]


_tracker = WallTracker()


def intent(symbol):
    """
    Намерение разворота по стакану.
    dir: -1 (готовят снижение), 0 (нейтрально), +1 (готовят рост).
    Логика: исчезновение bid-стенок + новые ask-стенки + дисбаланс < 0.6 => -1;
            исчезновение ask-стенок + новые bid-стенки + дисбаланс > 1.4 => +1.
    """
    depth = get_depth(symbol)
    if not depth:
        return {"dir": 0, "note": "нет данных стакана", "imbalance": 1.0,
                "bid_walls": [], "ask_walls": [], "events": []}

    events = _tracker.update(symbol)
    recent = _tracker.recent_events(symbol, 60)
    imb = imbalance(symbol, depth)
    bid_walls, ask_walls = find_walls(symbol, depth)

    bid_gone = [e for e in recent if e["type"] == "исчезла" and e["side"] == "bid"]
    ask_gone = [e for e in recent if e["type"] == "исчезла" and e["side"] == "ask"]
    bid_new = [e for e in recent if e["type"] == "появилась" and e["side"] == "bid"]
    ask_new = [e for e in recent if e["type"] == "появилась" and e["side"] == "ask"]

    score = 0.0
    notes = []
    if imb < 0.6:
        score -= 0.4; notes.append(f"дисбаланс {imb} — перевес продавцов")
    elif imb > 1.4:
        score += 0.4; notes.append(f"дисбаланс {imb} — перевес покупателей")
    if bid_gone and ask_new:
        score -= 0.4; notes.append("исчезают bid-стенки + новые ask-стенки: намерение раздавать")
    if ask_gone and bid_new:
        score += 0.4; notes.append("исчезают ask-стенки + новые bid-стенки: намерение покупать")
    if not notes:
        notes.append("стакан нейтрален: движения стенок нет")

    d = -1 if score <= -0.4 else (1 if score >= 0.4 else 0)
    return {"dir": d, "note": "; ".join(notes), "imbalance": imb,
            "bid_walls": bid_walls, "ask_walls": ask_walls, "events": recent}


def snapshot_for_ui(symbol):
    """Готовый снимок для окна приложения."""
    res = intent(symbol)
    depth = get_depth(symbol)
    res["symbol"] = symbol
    res["mid"] = depth["mid"] if depth else None
    return res