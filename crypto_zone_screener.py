#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate

# ================== БАЗОВЫЕ НАСТРОЙКИ ==================

BASE_URL = "https://api.binance.com"
STATE_FILE = "alerts_state.json"
ALERTS_CSV = "alerts_history.csv"

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "crypto-zone-screener"
})

# =======================================================


def safe_float(x, default=0.0):
    """
    Безопасное преобразование в float.
    """
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def get_json(url, params=None):
    """
    Запрос к Binance с обработкой лимитов.
    """
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=15)

            if r.status_code in (418, 429):
                retry_after = int(r.headers.get("Retry-After", "5"))
                time.sleep(max(2, retry_after))
                continue

            r.raise_for_status()
            return r.json()

        except requests.exceptions.RequestException as e:
            if attempt == 2:
                print(f"[ОШИБКА] {url}: {e}", file=sys.stderr)
                return None

            time.sleep(1 + attempt)

    return None


def get_top_symbols(top_n, min_quote_volume):
    """
    Получает топ ликвидных пар на Binance по обороту в USDT.
    """
    data = get_json(f"{BASE_URL}/api/v3/ticker/24hr")
    if not data:
        return []

    exclude_base = {
        "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "PAXG", "XUSD",
        "EUR", "GBP", "TRY", "BRL", "AUD", "AEUR", "WBTC", "WBETH",
        "USTC", "PYUSD", "GUSD", "FRAX", "USDD", "EURI"
    }

    out = []

    for t in data:
        symbol = t.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue

        base = symbol[:-4]

        if base in exclude_base:
            continue

        # Грубая фильтрация плечевых токенов.
        if len(base) >= 5 and base.endswith("UP"):
            continue
        if len(base) >= 6 and base.endswith("DOWN"):
            continue

        quote_volume = safe_float(t.get("quoteVolume"), 0.0)
        chg24 = safe_float(t.get("priceChangePercent"), 0.0)

        if quote_volume < min_quote_volume:
            continue

        out.append({
            "symbol": symbol,
            "quote_volume": quote_volume,
            "chg24h": chg24
        })

    out.sort(key=lambda x: x["quote_volume"], reverse=True)
    return out[:top_n]


def fetch_ohlcv(symbol, interval, limit):
    """
    Загружает свечи с Binance Spot.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    raw = get_json(f"{BASE_URL}/api/v3/klines", params=params)
    if not raw:
        return None

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(raw, columns=cols)

    numeric_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_base", "taker_buy_quote"
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    return df.reset_index(drop=True)


def add_indicators(df):
    """
    Добавляет индикаторы и считает score накопления/распределения для каждой свечи.
    """
    close = df["close"]

    # Скользящие
    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()

    # OBV
    df["obv"] = (close.diff().apply(np.sign).fillna(0) * df["volume"]).cumsum()

    # A/D line и CMF
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    clv = clv.fillna(0)

    df["ad"] = (clv * df["volume"]).cumsum()

    df["cmf"] = (
        (clv * df["volume"]).rolling(20).sum() /
        (df["volume"].rolling(20).sum() + 1e-12)
    )

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Доходности
    df["price_ret"] = close.pct_change(10)
    df["ret20"] = close.pct_change(20)

    # Объём
    vol_ma = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["vol_z"] = (df["volume"] - vol_ma) / (vol_std + 1e-12)

    # Taker buy ratio
    df["taker_ratio"] = (
        df["taker_buy_quote"].rolling(10).sum() /
        (df["quote_volume"].rolling(10).sum() + 1e-12)
    )

    # Наклоны объёмных индикаторов
    df["obv_slope"] = df["obv"].diff(10) / (df["obv"].rolling(20).std() + 1e-12)
    df["ad_slope"] = df["ad"].diff(10) / (df["ad"].rolling(20).std() + 1e-12)

    # Заполнение NaN для расчёта скоринга
    obv_slope = df["obv_slope"].fillna(0)
    ad_slope = df["ad_slope"].fillna(0)
    price_ret = df["price_ret"].fillna(0)
    cmf = df["cmf"].fillna(0)
    taker = df["taker_ratio"].fillna(0.5)
    vol_z = df["vol_z"].fillna(0)
    rsi = df["rsi"].fillna(50)

    up = close.diff().fillna(0) > 0

    accum = pd.Series(0.0, index=df.index)
    distrib = pd.Series(0.0, index=df.index)

    # 1. Дивергенция OBV/цены
    accum += 2 * ((obv_slope > 0.30) & (price_ret < 0.05)).astype(float)
    distrib += 2 * ((obv_slope < -0.30) & (price_ret > -0.05)).astype(float)

    # 2. Дивергенция AD/цены
    accum += 2 * ((ad_slope > 0.30) & (price_ret < 0.05)).astype(float)
    distrib += 2 * ((ad_slope < -0.30) & (price_ret > -0.05)).astype(float)

    # 3. CMF
    accum += (cmf > 0.05).astype(float)
    distrib += (cmf < -0.05).astype(float)

    # 4. Taker ratio
    accum += (taker > 0.52).astype(float)
    distrib += (taker < 0.48).astype(float)

    # 5. Повышенный объём по направлению свечи
    accum += ((vol_z > 0.5) & up).astype(float)
    distrib += ((vol_z > 0.5) & (~up)).astype(float)

    # 6. Боковик + RSI в середине диапазона
    sideways = (price_ret.abs() < 0.03) & rsi.between(40, 60)
    accum += sideways.astype(float)
    distrib += sideways.astype(float)

    df["accum_score"] = accum
    df["distrib_score"] = distrib

    return df


def make_zone(df, start, end, avg_score, score_col):
    """
    Создаёт описание зоны.
    """
    zone_type = "накопление" if "accum" in score_col else "распределение"

    return {
        "type": zone_type,
        "start": start,
        "end": end,
        "length": end - start + 1,
        "avg_score": avg_score,
        "start_date": df["dt"].iloc[start],
        "end_date": df["dt"].iloc[end],
        "start_price": safe_float(df["close"].iloc[start], np.nan),
        "end_price": safe_float(df["close"].iloc[end], np.nan),
        "price_low": safe_float(df["low"].iloc[start:end + 1].min(), np.nan),
        "price_high": safe_float(df["high"].iloc[start:end + 1].max(), np.nan),
    }


def find_zones(df, score_col, threshold, min_length):
    """
    Ищет непрерывные зоны, где score выше порога.
    """
    zones = []
    in_zone = False
    zone_start = None
    score_sum = 0.0

    scores = df[score_col].fillna(0).to_numpy()

    for i, score in enumerate(scores):
        if score >= threshold:
            if not in_zone:
                in_zone = True
                zone_start = i
                score_sum = float(score)
            else:
                score_sum += float(score)
        else:
            if in_zone:
                zone_end = i - 1
                length = zone_end - zone_start + 1

                if length >= min_length:
                    avg_score = score_sum / length
                    zones.append(make_zone(df, zone_start, zone_end, avg_score, score_col))

                in_zone = False
                score_sum = 0.0

    # Последняя незакрытая зона
    if in_zone:
        zone_end = len(df) - 1
        length = zone_end - zone_start + 1

        if length >= min_length:
            avg_score = score_sum / length
            zones.append(make_zone(df, zone_start, zone_end, avg_score, score_col))

    return zones


def get_zone_state(zones, last_index, max_age):
    """
    Определяет состояние последней зоны относительно текущей свечи.
    """
    if not zones:
        return {
            "state": "нет",
            "zone": None,
            "age": None
        }

    z = zones[-1]

    if z["end"] == last_index:
        state = "активна"
        age = 0
    else:
        age = last_index - z["end"]
        if age <= max_age:
            state = f"завершена {age} св. назад"
        else:
            state = f"старая {age} св. назад"

    return {
        "state": state,
        "zone": z,
        "age": age
    }


def choose_signal(df, accum_state, distrib_state):
    """
    Выбирает сигнал на основе активных и недавно завершённых зон.
    """
    last = df.iloc[-1]
    candidates = []

    # Активная зона накопления
    if accum_state["state"] == "активна" and accum_state["zone"] is not None:
        candidates.append({
            "signal": "ЛОНГ",
            "phase": "накопление",
            "state": "активна",
            "zone": accum_state["zone"],
            "score": accum_state["zone"]["avg_score"],
            "priority": 3
        })

    # Активная зона распределения
    if distrib_state["state"] == "активна" and distrib_state["zone"] is not None:
        candidates.append({
            "signal": "ШОРТ",
            "phase": "распределение",
            "state": "активна",
            "zone": distrib_state["zone"],
            "score": distrib_state["zone"]["avg_score"],
            "priority": 3
        })

    # Недавно завершённая зона накопления
    if accum_state["state"].startswith("завершена") and accum_state["zone"] is not None:
        z = accum_state["zone"]

        breakout_up = bool(last["close"] > z["price_high"])
        trend_up = bool(last["close"] > last["sma20"]) if pd.notna(last["sma20"]) else False
        vol_ok = bool(safe_float(last["vol_z"], 0.0) > 0.2)

        if breakout_up and (vol_ok or trend_up):
            candidates.append({
                "signal": "ЛОНГ",
                "phase": "накопление завершено",
                "state": accum_state["state"],
                "zone": z,
                "score": z["avg_score"],
                "priority": 2
            })
        else:
            candidates.append({
                "signal": "НАБЛЮДЕНИЕ",
                "phase": "накопление завершено",
                "state": accum_state["state"],
                "zone": z,
                "score": z["avg_score"],
                "priority": 1
            })

    # Недавно завершённая зона распределения
    if distrib_state["state"].startswith("завершена") and distrib_state["zone"] is not None:
        z = distrib_state["zone"]

        breakdown = bool(last["close"] < z["price_low"])
        trend_down = bool(last["close"] < last["sma20"]) if pd.notna(last["sma20"]) else False
        vol_ok = bool(safe_float(last["vol_z"], 0.0) > 0.2)

        if breakdown and (vol_ok or trend_down):
            candidates.append({
                "signal": "ШОРТ",
                "phase": "распределение завершено",
                "state": distrib_state["state"],
                "zone": z,
                "score": z["avg_score"],
                "priority": 2
            })
        else:
            candidates.append({
                "signal": "НАБЛЮДЕНИЕ",
                "phase": "распределение завершено",
                "state": distrib_state["state"],
                "zone": z,
                "score": z["avg_score"],
                "priority": 1
            })

    if candidates:
        best = max(candidates, key=lambda c: (c["priority"], c["score"]))
        z = best["zone"]

        reasons = []

        if best["state"] == "активна":
            reasons.append("активная зона")

        if "завершена" in best["state"]:
            reasons.append("зона завершена")

        if best["signal"] == "ЛОНГ":
            if last["close"] > z["price_high"]:
                reasons.append("пробой верхней границы")
            if safe_float(last["cmf"], 0.0) > 0.05:
                reasons.append("CMF положит.")
            if safe_float(last["taker_ratio"], 0.5) > 0.52:
                reasons.append("преобладание покупок")
            if safe_float(last["vol_z"], 0.0) > 0.3:
                reasons.append("повышенный объём")

        elif best["signal"] == "ШОРТ":
            if last["close"] < z["price_low"]:
                reasons.append("пробой нижней границы")
            if safe_float(last["cmf"], 0.0) < -0.05:
                reasons.append("CMF отрицат.")
            if safe_float(last["taker_ratio"], 0.5) < 0.48:
                reasons.append("преобладание продаж")
            if safe_float(last["vol_z"], 0.0) > 0.3:
                reasons.append("повышенный объём")

        else:
            if best["phase"].startswith("накопление"):
                reasons.append("ожидание пробоя вверх")
            if best["phase"].startswith("распределение"):
                reasons.append("ожидание пробоя вниз")

        return {
            "signal": best["signal"],
            "phase": best["phase"],
            "zone_state": best["state"],
            "zone": z,
            "score": best["score"],
            "reasons": "; ".join(dict.fromkeys(reasons))
        }

    # Нейтральный режим: активных/свежих зон нет, но показываем последнюю старую зону для контекста
    z = None
    state = "нет"

    if accum_state["zone"] is not None and distrib_state["zone"] is not None:
        if accum_state["zone"]["end"] >= distrib_state["zone"]["end"]:
            z = accum_state["zone"]
            state = f"накопление: {accum_state['state']}"
        else:
            z = distrib_state["zone"]
            state = f"распределение: {distrib_state['state']}"

    elif accum_state["zone"] is not None:
        z = accum_state["zone"]
        state = f"накопление: {accum_state['state']}"

    elif distrib_state["zone"] is not None:
        z = distrib_state["zone"]
        state = f"распределение: {distrib_state['state']}"

    return {
        "signal": "НЕЙТРАЛЬНО",
        "phase": "-",
        "zone_state": state,
        "zone": z,
        "score": z["avg_score"] if z is not None else 0.0,
        "reasons": "нет активного сигнала"
    }


def analyze_symbol_zones(item, args):
    """
    Полный анализ одной пары.
    """
    symbol = item["symbol"]

    df = fetch_ohlcv(symbol, args.interval, args.limit)
    if df is None or len(df) < 80:
        return None

    df = add_indicators(df)

    accum_zones = find_zones(df, "accum_score", args.accum_threshold, args.min_zone)
    distrib_zones = find_zones(df, "distrib_score", args.distrib_threshold, args.min_zone)

    last_index = len(df) - 1

    accum_state = get_zone_state(accum_zones, last_index, args.zone_age)
    distrib_state = get_zone_state(distrib_zones, last_index, args.zone_age)

    sig = choose_signal(df, accum_state, distrib_state)

    last = df.iloc[-1]
    z = sig["zone"]

    zone_id = None
    if z is not None:
        zone_id = f"{z['type']}-{z['start_date'].strftime('%Y%m%d%H%M')}"

    zone_active = sig["zone_state"] == "активна"

    return {
        "symbol": symbol,
        "signal": sig["signal"],
        "phase": sig["phase"],
        "zone_state": sig["zone_state"],
        "zone_len": z["length"] if z is not None else 0,
        "score": round(sig["score"], 3),
        "price": safe_float(last["close"], np.nan),
        "chg24h": safe_float(item.get("chg24h"), np.nan),
        "ret20_pct": safe_float(last["ret20"], 0.0) * 100,
        "vol_z": safe_float(last["vol_z"], 0.0),
        "cmf": safe_float(last["cmf"], 0.0),
        "taker_ratio": safe_float(last["taker_ratio"], 0.5),
        "rsi": safe_float(last["rsi"], 50.0),
        "zone_range": f"{z['price_low']:.2f}-{z['price_high']:.2f}" if z is not None else "-",
        "reasons": sig["reasons"],
        "zone_id": zone_id,
        "zone_type": z["type"] if z is not None else None,
        "zone_active": zone_active,
        "accum_zones": len(accum_zones),
        "distrib_zones": len(distrib_zones)
    }


def run_scan(args):
    """
    Сканирует топ пар.
    """
    top = get_top_symbols(args.top, args.min_quote)

    if not top:
        print("Не удалось получить список пар. Проверьте доступ к Binance.")
        return pd.DataFrame()

    print(f"Сканирую {len(top)} пар: интервал={args.interval}, свечей={args.limit}.")

    rows = []

    for i, item in enumerate(top, start=1):
        try:
            row = analyze_symbol_zones(item, args)
            if row is not None:
                rows.append(row)

        except Exception as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] {item['symbol']}: {e}", file=sys.stderr)

        time.sleep(args.sleep)

        if i % 10 == 0:
            print(f"Обработано {i}/{len(top)}", file=sys.stderr)

    df = pd.DataFrame(rows)

    if not df.empty:
        df.sort_values(
            by=["zone_active", "score", "vol_z"],
            ascending=False,
            inplace=True
        )

        df.to_csv("top100_zones_scan.csv", index=False)

    return df


def print_scan_table(df, display=30):
    """
    Печатает таблицу на русском.
    """
    if df.empty:
        print("Нет данных для отображения.")
        return

    promising = df[df["signal"].isin(["ЛОНГ", "ШОРТ", "НАБЛЮДЕНИЕ"])]

    if promising.empty:
        promising = df.head(display)
    else:
        promising = promising.head(display)

    columns = [
        "symbol", "signal", "phase", "zone_state", "zone_len", "score",
        "price", "chg24h", "ret20_pct", "vol_z", "cmf", "taker_ratio",
        "rsi", "zone_range", "reasons"
    ]

    headers = {
        "symbol": "Пара",
        "signal": "Сигнал",
        "phase": "Фаза",
        "zone_state": "Состояние зоны",
        "zone_len": "Длина зоны",
        "score": "Сила зоны",
        "price": "Цена",
        "chg24h": "24ч %",
        "ret20_pct": "20 св. %",
        "vol_z": "Z-объём",
        "cmf": "CMF",
        "taker_ratio": "Покупки",
        "rsi": "RSI",
        "zone_range": "Диапазон зоны",
        "reasons": "Причины"
    }

    display_df = promising[columns].rename(columns=headers)

    print("\n=== Автоматический скринер зон накопления и распределения ===")
    print(tabulate(display_df, headers="keys", tablefmt="github", showindex=False, floatfmt=".3f"))
    print("\nПолный результат сохранён в: top100_zones_scan.csv")


def send_telegram_message(text):
    """
    Опциональная отправка алертов в Telegram.

    Нужно задать переменные окружения:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text[:4000]
    }

    try:
        requests.post(url, json=payload, timeout=10)
        return True
    except Exception:
        return False


def run_alerts(df):
    """
    Сравнивает текущие зоны с предыдущим состоянием и создаёт алерты.
    """
    if df.empty:
        print("Нет данных для алертов.")
        return

    state = {}

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}

    alerts = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in df.iterrows():
        symbol = row["symbol"]
        old = state.get(symbol, {})

        new_zone_id = row.get("zone_id")
        old_zone_id = old.get("zone_id")

        new_active = bool(row.get("zone_active", False))
        old_active = bool(old.get("zone_active", False))

        event = None

        if new_zone_id is not None and new_zone_id != old_zone_id:
            if new_active:
                event = "Новая активная зона"
            else:
                event = "Новая/обновлённая зона"

        elif new_zone_id is not None and new_zone_id == old_zone_id and new_active != old_active:
            if new_active:
                event = "Зона стала активной"
            else:
                event = "Активная зона перестала быть активной"

        elif new_zone_id is None and old_zone_id is not None and old_active:
            event = "Активная зона исчезла"

        if event is not None:
            alerts.append({
                "Время": now,
                "Пара": symbol,
                "Событие": event,
                "Сигнал": row["signal"],
                "Фаза": row["phase"],
                "Состояние": row["zone_state"],
                "Цена": row["price"],
                "Причины": row["reasons"]
            })

        state[symbol] = {
            "zone_id": new_zone_id,
            "signal": row["signal"],
            "phase": row["phase"],
            "zone_state": row["zone_state"],
            "zone_active": new_active,
            "updated": now
        }

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    if alerts:
        alerts_df = pd.DataFrame(alerts)

        alert_columns = [
            "Время", "Пара", "Событие", "Сигнал", "Фаза",
            "Состояние", "Цена", "Причины"
        ]

        alerts_df = alerts_df[alert_columns]

        if os.path.exists(ALERTS_CSV):
            alerts_df.to_csv(ALERTS_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            alerts_df.to_csv(ALERTS_CSV, index=False, encoding="utf-8-sig")

        print("\n=== АЛЕРТЫ ===")
        print(tabulate(alerts_df, headers="keys", tablefmt="github", showindex=False))
        print(f"\nАлерты сохранены в: {ALERTS_CSV}")

        for alert in alerts:
            text = (
                f"{alert['Время']} | {alert['Пара']}\n"
                f"Событие: {alert['Событие']}\n"
                f"Сигнал: {alert['Сигнал']}\n"
                f"Фаза: {alert['Фаза']}\n"
                f"Состояние: {alert['Состояние']}\n"
                f"Цена: {alert['Цена']}\n"
                f"Причины: {alert['Причины']}"
            )
            send_telegram_message(text)

    else:
        print("\nНовых алертов нет.")


def prepare_symbol_analysis(symbol, interval, limit, args):
    """
    Загружает данные и находит зоны для одного символа.
    """
    df = fetch_ohlcv(symbol, interval, limit)

    if df is None or len(df) < 80:
        return None, [], []

    df = add_indicators(df)

    accum_zones = find_zones(df, "accum_score", args.accum_threshold, args.min_zone)
    distrib_zones = find_zones(df, "distrib_score", args.distrib_threshold, args.min_zone)

    return df, accum_zones, distrib_zones


def calculate_volume_profile(df, bins=60):
    """
    Упрощённый Volume Profile: распределяет объём свечи по ценовым корзинам.
    """
    if bins <= 0:
        bins = 60

    low = float(df["low"].min())
    high = float(df["high"].max())

    if high <= low:
        high = low + 1.0

    edges = np.linspace(low, high, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    volumes = np.zeros(bins)

    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    vols = df["volume"].to_numpy()

    for i in range(len(df)):
        l = lows[i]
        h = highs[i]
        v = vols[i]

        if v <= 0:
            continue

        if h <= l:
            mid = (h + l) / 2
            idx = int(np.searchsorted(edges, mid) - 1)
            if 0 <= idx < bins:
                volumes[idx] += v
            continue

        mask = (edges[1:] > l) & (edges[:-1] < h)
        cnt = int(mask.sum())

        if cnt > 0:
            volumes[mask] += v / cnt

    return centers, volumes


def add_zone_shapes(fig, zones, row=1, col=1, max_annotations=8):
    """
    Добавляет зоны на график.
    """
    for z in zones:
        if z["type"] == "накопление":
            fill_color = "rgba(38, 166, 154, 0.16)"
            line_color = "rgba(38, 166, 154, 0.55)"
        else:
            fill_color = "rgba(239, 83, 80, 0.16)"
            line_color = "rgba(239, 83, 80, 0.55)"

        fig.add_shape(
            type="rect",
            x0=z["start_date"],
            y0=z["price_low"],
            x1=z["end_date"],
            y1=z["price_high"],
            fillcolor=fill_color,
            line=dict(color=line_color, width=1),
            layer="below",
            row=row,
            col=col
        )

    if zones:
        for z in zones[-max_annotations:]:
            text = "Н" if z["type"] == "накопление" else "Р"
            color = "#26a69a" if z["type"] == "накопление" else "#ef5350"

            fig.add_annotation(
                x=z["end_date"],
                y=z["price_high"],
                text=text,
                showarrow=False,
                font=dict(size=10, color=color),
                row=row,
                col=col
            )


def run_volume_profile(symbol, interval, limit, args):
    """
    Строит график: свечи, зоны, Volume Profile, объём, OBV, CMF.
    """
    df, accum_zones, distrib_zones = prepare_symbol_analysis(symbol, interval, limit, args)

    if df is None:
        print(f"Не удалось загрузить данные для {symbol}.")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Для Volume Profile нужно установить plotly: pip install plotly")
        return

    centers, volumes = calculate_volume_profile(df, args.bins)

    fig = make_subplots(
        rows=4,
        cols=2,
        column_widths=[0.8, 0.2],
        row_heights=[0.5, 0.15, 0.15, 0.2],
        horizontal_spacing=0.03
    )

    # Свечи
    fig.add_trace(
        go.Candlestick(
            x=df["dt"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Цена",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1,
        col=1
    )

    # SMA
    fig.add_trace(
        go.Scatter(
            x=df["dt"],
            y=df["sma20"],
            mode="lines",
            name="SMA 20",
            line=dict(color="orange", width=1)
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["dt"],
            y=df["sma50"],
            mode="lines",
            name="SMA 50",
            line=dict(color="purple", width=1)
        ),
        row=1,
        col=1
    )

    # Зоны
    all_zones = accum_zones + distrib_zones
    add_zone_shapes(fig, all_zones, row=1, col=1)

    # POC — уровень максимального объёма
    if len(volumes) > 0:
        poc_idx = int(np.argmax(volumes))
        poc_price = float(centers[poc_idx])

        fig.add_hline(
            y=poc_price,
            line_dash="dot",
            line_color="gold",
            row=1,
            col=1
        )

    # Volume Profile справа
    fig.add_trace(
        go.Bar(
            x=volumes,
            y=centers,
            orientation="h",
            name="Volume Profile",
            marker_color="rgba(100, 149, 237, 0.55)",
            hovertemplate="Цена: %{y:.2f}<br>Объём: %{x:.2f}<extra></extra>"
        ),
        row=1,
        col=2
    )

    # Объём
    volume_colors = [
        "#26a69a" if close >= open else "#ef5350"
        for close, open in zip(df["close"], df["open"])
    ]

    fig.add_trace(
        go.Bar(
            x=df["dt"],
            y=df["volume"],
            name="Объём",
            marker_color=volume_colors,
            opacity=0.7
        ),
        row=2,
        col=1
    )

    # OBV
    fig.add_trace(
        go.Scatter(
            x=df["dt"],
            y=df["obv"],
            mode="lines",
            name="OBV",
            line=dict(color="#2196f3", width=1.5)
        ),
        row=3,
        col=1
    )

    # CMF
    fig.add_trace(
        go.Scatter(
            x=df["dt"],
            y=df["cmf"],
            mode="lines",
            name="CMF",
            line=dict(color="#ff9800", width=1.5)
        ),
        row=4,
        col=1
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        row=4,
        col=1
    )

    price_low = float(df["low"].min())
    price_high = float(df["high"].max())
    price_pad = (price_high - price_low) * 0.05

    fig.update_yaxes(
        range=[price_low - price_pad, price_high + price_pad],
        title_text="Цена",
        row=1,
        col=1
    )

    fig.update_yaxes(
        range=[price_low - price_pad, price_high + price_pad],
        showticklabels=False,
        row=1,
        col=2
    )

    fig.update_xaxes(title_text="Объём профиля", row=1, col=2)
    fig.update_yaxes(title_text="Объём", row=2, col=1)
    fig.update_yaxes(title_text="OBV", row=3, col=1)
    fig.update_yaxes(title_text="CMF", row=4, col=1)
    fig.update_xaxes(title_text="Дата", row=4, col=1)

    # Синхронизация нижних графиков по цене
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=3, col=1)
    fig.update_xaxes(matches="x", row=4, col=1)

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    fig.update_layout(
        height=1000,
        template="plotly_dark",
        title_text=f"{symbol}: Volume Profile, зоны накопления и распределения",
        showlegend=True
    )

    filename = f"{symbol}_volume_profile_zones.html"
    fig.write_html(filename)

    print(f"\n✅ Volume Profile сохранён в: {filename}")


def backtest_direction(symbol, df, zones, direction, stop_pct, take_pct, max_bars):
    """
    Простой бэктест:
    - после завершения зоны накопления вход лонг на следующей свече;
    - после завершения зоны распределения вход шорт на следующей свече;
    - выход по стопу, тейку или таймауту.
    """
    trades = []

    if df is None or df.empty or not zones:
        return pd.DataFrame()

    for z in zones:
        entry_idx = z["end"] + 1

        if entry_idx >= len(df):
            continue

        entry_price = safe_float(df["open"].iloc[entry_idx], 0.0)

        if entry_price <= 0:
            continue

        if direction == "long":
            stop_price = entry_price * (1 - stop_pct / 100)
            take_price = entry_price * (1 + take_pct / 100)
        else:
            stop_price = entry_price * (1 + stop_pct / 100)
            take_price = entry_price * (1 - take_pct / 100)

        exit_price = None
        exit_reason = None
        exit_idx = None

        max_favorable = 0.0
        max_adverse = 0.0

        last_possible = min(entry_idx + max_bars, len(df) - 1)

        for j in range(entry_idx, last_possible + 1):
            high = safe_float(df["high"].iloc[j], entry_price)
            low = safe_float(df["low"].iloc[j], entry_price)

            if direction == "long":
                max_favorable = max(max_favorable, (high - entry_price) / entry_price * 100)
                max_adverse = max(max_adverse, (entry_price - low) / entry_price * 100)

                # Консервативно: сначала проверяем стоп, потом тейк.
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "стоп"
                    exit_idx = j
                    break

                if high >= take_price:
                    exit_price = take_price
                    exit_reason = "тейк"
                    exit_idx = j
                    break

            else:
                max_favorable = max(max_favorable, (entry_price - low) / entry_price * 100)
                max_adverse = max(max_adverse, (high - entry_price) / entry_price * 100)

                if high >= stop_price:
                    exit_price = stop_price
                    exit_reason = "стоп"
                    exit_idx = j
                    break

                if low <= take_price:
                    exit_price = take_price
                    exit_reason = "тейк"
                    exit_idx = j
                    break

        if exit_price is None:
            exit_idx = last_possible
            exit_price = safe_float(df["close"].iloc[exit_idx], entry_price)
            exit_reason = "таймаут"

        if direction == "long":
            profit_pct = (exit_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - exit_price) / entry_price * 100

        trades.append({
            "symbol": symbol,
            "direction": direction,
            "zone_type": z["type"],
            "entry_time": df["dt"].iloc[entry_idx],
            "exit_time": df["dt"].iloc[exit_idx],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_pct": profit_pct,
            "exit_reason": exit_reason,
            "bars_held": exit_idx - entry_idx + 1,
            "max_favorable_pct": max_favorable,
            "max_adverse_pct": max_adverse,
            "zone_start": z["start_date"],
            "zone_end": z["end_date"],
            "zone_strength": z["avg_score"]
        })

    return pd.DataFrame(trades)


def backtest_stats(trades):
    """
    Статистика по сделкам.
    """
    if trades is None or trades.empty:
        return {}

    rets = trades["profit_pct"].astype(float)

    if rets.empty:
        return {}

    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = 999.0 if gross_win > 0 else 0.0

    cum = (1 + rets / 100).cumprod()
    max_dd = float(((cum / cum.cummax()) - 1).min() * 100) if len(cum) > 0 else 0.0

    return {
        "Сделок": int(len(rets)),
        "Винрейт %": float((len(wins) / len(rets)) * 100),
        "Прибыльных": int(len(wins)),
        "Убыточных": int(len(losses)),
        "Средняя %": float(rets.mean()),
        "Медиана %": float(rets.median()),
        "Суммарно %": float(rets.sum()),
        "Лучшая %": float(rets.max()),
        "Худшая %": float(rets.min()),
        "Profit Factor": float(profit_factor),
        "Макс. просадка %": max_dd
    }


def print_backtest_stats(title, stats):
    """
    Печатает статистику бэктеста.
    """
    print(f"\n=== {title} ===")

    if not stats:
        print("Сделок нет.")
        return

    print(tabulate([stats], headers="keys", tablefmt="github", floatfmt=".2f"))


def run_backtest_for_symbol(symbol, interval, limit, args):
    """
    Бэктест по одному символу.
    """
    df, accum_zones, distrib_zones = prepare_symbol_analysis(symbol, interval, limit, args)

    if df is None:
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    long_trades = backtest_direction(
        symbol=symbol,
        df=df,
        zones=accum_zones,
        direction="long",
        stop_pct=args.stop,
        take_pct=args.take,
        max_bars=args.hold
    )

    short_trades = backtest_direction(
        symbol=symbol,
        df=df,
        zones=distrib_zones,
        direction="short",
        stop_pct=args.stop,
        take_pct=args.take,
        max_bars=args.hold
    )

    long_stats = backtest_stats(long_trades)
    short_stats = backtest_stats(short_trades)

    return long_trades, short_trades, long_stats, short_stats


def run_backtest(args):
    """
    Бэктест для одного символа или для списка топ-ликвидных символов.
    """
    symbols = []

    if args.symbol.upper() == "ALL":
        top = get_top_symbols(args.backtest_top, args.min_quote)
        symbols = [t["symbol"] for t in top]
    else:
        symbols = [args.symbol.upper()]

    if not symbols:
        print("Нет символов для бэктеста.")
        return

    all_long = []
    all_short = []
    per_symbol_rows = []

    for symbol in symbols:
        print(f"\nБэктест: {symbol}...")

        long_trades, short_trades, long_stats, short_stats = run_backtest_for_symbol(
            symbol=symbol,
            interval=args.interval,
            limit=args.limit,
            args=args
        )

        if not long_trades.empty:
            all_long.append(long_trades)

        if not short_trades.empty:
            all_short.append(short_trades)

        per_symbol_rows.append({
            "Пара": symbol,
            "Лонг сделок": long_stats.get("Сделок", 0),
            "Лонг винрейт %": long_stats.get("Винрейт %", 0.0),
            "Лонг средняя %": long_stats.get("Средняя %", 0.0),
            "Лонг суммарно %": long_stats.get("Суммарно %", 0.0),
            "Шорт сделок": short_stats.get("Сделок", 0),
            "Шорт винрейт %": short_stats.get("Винрейт %", 0.0),
            "Шорт средняя %": short_stats.get("Средняя %", 0.0),
            "Шорт суммарно %": short_stats.get("Суммарно %", 0.0)
        })

        time.sleep(args.sleep)

    all_long_df = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    all_short_df = pd.concat(all_short, ignore_index=True) if all_short else pd.DataFrame()

    print_backtest_stats(
        "ИТОГ: ЛОНГ после зон накопления",
        backtest_stats(all_long_df)
    )

    print_backtest_stats(
        "ИТОГ: ШОРТ после зон распределения",
        backtest_stats(all_short_df)
    )

    if per_symbol_rows:
        print("\n=== Статистика по парам ===")
        print(tabulate(per_symbol_rows, headers="keys", tablefmt="github", floatfmt=".2f"))

    if not all_long_df.empty:
        all_long_df.to_csv("backtest_long_trades.csv", index=False)
        print("\nЛонг-сделки сохранены в: backtest_long_trades.csv")

    if not all_short_df.empty:
        all_short_df.to_csv("backtest_short_trades.csv", index=False)
        print("Шорт-сделки сохранены в: backtest_short_trades.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Автоматический скринер зон накопления и распределения на Binance."
    )

    parser.add_argument(
        "--mode",
        choices=["scan", "alert", "volume", "backtest", "all"],
        default="all",
        help="Режим работы."
    )

    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Символ для Volume Profile и бэктеста. Для бэктеста топ-списка используйте ALL."
    )

    parser.add_argument("--interval", default="4h", help="Интервал: 15m, 1h, 4h, 1d.")
    parser.add_argument("--limit", type=int, default=300, help="Количество свечей.")
    parser.add_argument("--top", type=int, default=100, help="Сколько пар сканировать.")
    parser.add_argument("--display", type=int, default=30, help="Сколько строк показывать в таблице.")

    parser.add_argument(
        "--min-quote",
        type=float,
        default=10_000_000,
        help="Минимальный суточный оборот в USDT."
    )

    parser.add_argument("--accum-threshold", type=float, default=3.0, help="Порог для зоны накопления.")
    parser.add_argument("--distrib-threshold", type=float, default=3.0, help="Порог для зоны распределения.")
    parser.add_argument("--min-zone", type=int, default=3, help="Минимальная длина зоны в свечах.")
    parser.add_argument("--zone-age", type=int, default=5, help="Сколько свечей считать зону свежей после завершения.")

    parser.add_argument("--bins", type=int, default=60, help="Количество корзин для Volume Profile.")

    parser.add_argument("--stop", type=float, default=3.0, help="Стоп в %% для бэктеста.")
    parser.add_argument("--take", type=float, default=6.0, help="Тейк в %% для бэктеста.")
    parser.add_argument("--hold", type=int, default=30, help="Максимальное время удержания сделки в свечах.")

    parser.add_argument("--backtest-top", type=int, default=10, help="Сколько пар тестировать при --symbol ALL.")
    parser.add_argument("--sleep", type=float, default=0.12, help="Пауза между запросами.")

    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Запускать сканирование и алерты каждые N минут. 0 — без цикла."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Режим постоянного мониторинга алертов
    if args.loop > 0:
        print(f"Режим мониторинга: сканирование и алерты каждые {args.loop} мин.")
        print("Для остановки нажмите Ctrl+C.")

        try:
            while True:
                scan_df = run_scan(args)

                if not scan_df.empty:
                    print_scan_table(scan_df, args.display)
                    run_alerts(scan_df)

                print(f"\nСледующий цикл через {args.loop} мин...\n")
                time.sleep(args.loop * 60)

        except KeyboardInterrupt:
            print("\nМониторинг остановлен пользователем.")

        return

    # Обычный режим
    if args.mode in ["scan", "alert", "all"]:
        scan_df = run_scan(args)

        if not scan_df.empty:
            print_scan_table(scan_df, args.display)

            if args.mode in ["alert", "all"]:
                run_alerts(scan_df)

    if args.mode in ["volume", "all"]:
        symbol = args.symbol.upper()

        if symbol == "ALL":
            top = get_top_symbols(1, args.min_quote)
            symbol = top[0]["symbol"] if top else "BTCUSDT"

        run_volume_profile(symbol, args.interval, args.limit, args)

    if args.mode in ["backtest", "all"]:
        run_backtest(args)


if __name__ == "__main__":
    main()