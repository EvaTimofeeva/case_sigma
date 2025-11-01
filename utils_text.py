"""
Вспомогательные утилиты:
robust_read_table: надёжное чтение CSV/TSV (разные кодировки и разделители)
load_noise_patterns: загрузка шаблонов "мусора" для чистки текста
clean_text_noise: очистка текста от HTML и служебных фраз
"""
from __future__ import annotations
import os
import re
import pandas as pd
from typing import List

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE       = re.compile(r"\s+")

# Базовые "мусорные" фразы по умолчанию (если файла нет)
_DEFAULT_NOISE: List[str] = [
    r"^добро пожаловать на экзамен$",
    r"^начните диалог\.?$",
    r"^здравствуйте[,! ]*это экзамен\.?$",
    r"^приветствуем на экзамене\.?$",
    r"^говорите после сигнала\.?$",
]

def robust_read_table(path: str) -> pd.DataFrame:
    """
    Пробует прочитать табличный файл несколькими способами.
    Поддерживает ; и , как разделители и популярные кодировки.
    """
    attempts = [
        dict(sep=";", encoding="utf-8-sig"),
        dict(sep=",", encoding="utf-8-sig"),
        dict(sep=";", encoding="cp1251"),
        dict(sep=",", encoding="cp1251"),
        dict(sep=None, encoding="utf-8", engine="python"),  # авто-определение
    ]
    last_err = None
    for a in attempts:
        try:
            return pd.read_csv(path, **a)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Не удалось прочитать файл '{path}': {last_err}")

def load_noise_patterns(path: str) -> List[re.Pattern]:
    """
    Загружает построчно регулярные выражения для удаления "мусора".
    Если файл отсутствует — возвращает набор дефолтных шаблонов.
    """
    patterns: List[str] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    patterns.append(s)
        except Exception:
            pass
    if not patterns:
        patterns = _DEFAULT_NOISE
    # компилируем в regex c флагом IGNORECASE
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]

def _strip_html(s: str) -> str:
    """Удаляет HTML-теги и лишние пробелы."""
    s = _HTML_TAG_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def clean_text_noise(text: str, patterns: List[re.Pattern]) -> str:
    """
    Удаляет HTML-теги + строки, полностью совпадающие с "мусорными" паттернами.
    Паттерны предполагаются как регулярные выражения.
    """
    if not isinstance(text, str):
        return ""
    s = _strip_html(text)
    # удаляем "служебные" строки полностью
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    keep: List[str] = []
    for ln in lines:
        if any(p.fullmatch(ln) for p in patterns):
            continue
        keep.append(ln)
    s2 = " ".join(keep).strip()
    return s2
