#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Локальные ИИ-агенты для оценки сигналов.

Работают офлайн, без API-ключей: каждый агент — взвешенная эвристическая модель.
Опционально подключается внешняя LLM (OpenAI-совместимый API) через переменные:
    LLM_API_KEY   — ключ
    LLM_BASE_URL  — например https://api.openai.com/v1 (по умолчанию)
    LLM_MODEL     — например gpt-4o-mini (по умолчанию)
"""
import os
import requests


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _direction(r):
    sig = str(r.get("signal", ""))
    if sig == "ЛОНГ":
        return 1
    if sig == "ШОРТ":
        return -1
    return 0


def _f(r, key, default=0.0):
    try:
        v = float(r.get(key, default))
        if v != v:
            return default
        return v
    except Exception:
        return default


class VolumeAgent:
    """Оценивает, подтверждает ли объём и денежный поток текущий сигнал."""
    name = "Агент объёма"

    def analyze(self, r):
        d = _direction(r)
        if d == 0:
            return 0.0, "нет активного сигнала — объём не оценивается"
        vol_z = _f(r, "vol_z")
        cmf = _f(r, "cmf")
        taker = _f(r, "taker_ratio", 0.5)
        conf = 0.0
        notes = []
        if vol_z > 1.0:
            conf += 0.35; notes.append(f"объём выше нормы (Z={vol_z:.2f})")
        elif vol_z > 0.4:
            conf += 0.20; notes.append("объём умеренно повышен")
        else:
            notes.append("объём не повышен — подтверждение слабое")
        if d * cmf > 0.05:
            conf += 0.35; notes.append(f"денежный поток в сторону сигнала (CMF={cmf:.2f})")
        elif d * cmf < -0.05:
            conf -= 0.25; notes.append("денежный поток против сигнала")
        if d * (taker - 0.5) > 0.02:
            conf += 0.30; notes.append("агрессоры совпадают с сигналом")
        elif d * (taker - 0.5) < -0.02:
            conf -= 0.20; notes.append("агрессоры против сигнала")
        return _clamp(conf), "; ".join(notes)


class TrendAgent:
    """Оценивает здоровье импульса: RSI и удалённость цены от зоны."""
    name = "Агент тренда"

    def analyze(self, r):
        d = _direction(r)
        if d == 0:
            return 0.0, "нет активного сигнала — тренд не оценивается"
        rsi = _f(r, "rsi", 50)
        ret20 = _f(r, "ret20_pct")
        conf = 0.3
        notes = []
        if d == 1:
            if 50 <= rsi <= 70:
                conf += 0.3; notes.append(f"RSI {rsi:.0f} — импульс здоровый")
            elif rsi > 75:
                conf -= 0.3; notes.append(f"RSI {rsi:.0f} — перегрев")
            if 0 < ret20 <= 15:
                conf += 0.2; notes.append("цена ещё не ушла далеко от зоны")
            elif ret20 > 25:
                conf -= 0.3; notes.append("цена уже сильно ушла вверх")
        else:
            if 30 <= rsi <= 50:
                conf += 0.3; notes.append(f"RSI {rsi:.0f} — снижение здоровое")
            elif rsi < 25:
                conf -= 0.3; notes.append(f"RSI {rsi:.0f} — перепроданность, шорт рискован")
            if -15 <= ret20 < 0:
                conf += 0.2; notes.append("снижение не перегрето")
            elif ret20 < -25:
                conf -= 0.3; notes.append("цена уже сильно ушла вниз")
        return _clamp(conf), "; ".join(notes)


class ZoneAgent:
    """Оценивает свежесть и силу зоны накопления/распределения."""
    name = "Агент зон"

    def analyze(self, r):
        d = _direction(r)
        if d == 0:
            return 0.0, "нет зоны в активном состоянии"
        state = str(r.get("zone_state", ""))
        score = _f(r, "score")
        length = _f(r, "zone_len")
        conf = 0.2
        notes = []
        if state == "активна":
            conf += 0.35; notes.append("зона активна сейчас")
        elif state.startswith("завершена"):
            digits = "".join(ch for ch in state.split()[1] if ch.isdigit())
            age = int(digits) if digits else 99
            if age <= 3:
                conf += 0.3; notes.append(f"зона завершена {age} св. назад — свежая")
            else:
                conf -= 0.2; notes.append("зона несвежая")
        if score >= 4:
            conf += 0.25; notes.append(f"высокая сила зоны ({score:.1f})")
        if 4 <= length <= 20:
            conf += 0.2; notes.append("длительность зоны достаточна")
        return _clamp(conf), "; ".join(notes)


class RiskAgent:
    """Ищет факторы риска: аномальные движения, кульминации, экстремумы."""
    name = "Агент риска"

    def analyze(self, r):
        risk = 0.0
        notes = []
        chg = abs(_f(r, "chg24h"))
        vol_z = _f(r, "vol_z")
        rsi = _f(r, "rsi", 50)
        if chg > 12:
            risk += 0.4; notes.append(f"аномальное движение за 24ч ({chg:.1f}%)")
        if vol_z > 3:
            risk += 0.3; notes.append("кульминация объёма — возможно истощение")
        if rsi > 80 or rsi < 20:
            risk += 0.3; notes.append("RSI на экстремуме")
        if not notes:
            notes.append("повышенных рисков не обнаружено")
        return _clamp(risk), "; ".join(notes)


AGENTS = [VolumeAgent(), TrendAgent(), ZoneAgent(), RiskAgent()]
WEIGHTS = [0.35, 0.25, 0.25, 0.15]


def _llm_comment(summary):
    """Опциональный комментарий внешней LLM, если заданы переменные окружения."""
    key = os.getenv("LLM_API_KEY")
    if not key:
        return None
    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "temperature": 0.3,
                "messages": [
                    {"role": "system",
                     "content": "Ты осторожный аналитик крипторынка. Даёшь только оценку качества сигнала и рисков, без торговых рекомендаций."},
                    {"role": "user", "content": summary},
                ],
            },
            timeout=25,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def analyze_row(r):
    """Вердикт координатора по строке таблицы."""
    d = _direction(r)
    reports = []
    for ag, w in zip(AGENTS, WEIGHTS):
        c, note = ag.analyze(r)
        reports.append((ag.name, c, note))
    risk = reports[-1][1]
    conf = sum(c * w for c, w in zip([x[1] for x in reports], WEIGHTS)) - 0.25 * risk
    conf = _clamp(conf)

    if d == 0:
        verdict = "Ждать"
    elif conf >= 0.60:
        verdict = "Сильный кандидат"
    elif conf >= 0.40:
        verdict = "Умеренный кандидат"
    elif conf >= 0.25:
        verdict = "Слабый кандидат"
    else:
        verdict = "Избегать"

    if d == 1 and verdict != "Ждать":
        verdict += " (лонг)"
    elif d == -1 and verdict != "Ждать":
        verdict += " (шорт)"

    return {"verdict": verdict, "conf": conf, "dir": d, "reports": reports}


def analyze_dataframe(df):
    """Добавляет колонки ai_verdict и ai_conf ко всему результату скана."""
    verdicts, confs = [], []
    for _, r in df.iterrows():
        a = analyze_row(r.to_dict())
        verdicts.append(a["verdict"])
        confs.append(round(a["conf"], 2))
    df = df.copy()
    df["ai_verdict"] = verdicts
    df["ai_conf"] = confs
    return df


def build_report(r):
    """Полный текстовый отчёт агентов по выбранной паре."""
    a = analyze_row(r)
    lines = [
        f"Пара: {r.get('symbol')} | Сигнал: {r.get('signal')} | Фаза: {r.get('phase')}",
        f"Вердикт: {a['verdict']} | Уверенность: {a['conf']:.2f}",
        "-" * 70,
    ]
    for name, c, note in a["reports"]:
        lines.append(f"{name}: {c:.2f} — {note}")
    summary = "\n".join(lines)
    llm = _llm_comment(summary)
    if llm:
        lines.append("-" * 70)
        lines.append("Комментарий внешней LLM:")
        lines.append(llm)
    return "\n".join(lines)