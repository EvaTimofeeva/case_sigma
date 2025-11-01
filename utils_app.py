# utils_app.py
# -------------------------------------------------------------
# Утилиты проекта: чтение конфигурации, CSV, подмешивание подписи
# картинок, очистка текста от "мусора".
# -------------------------------------------------------------
import os
import re
import yaml
import codecs
import pandas as pd
from typing import Optional
from urllib.parse import urlparse
import csv

DEFAULT_CFG = {
    "transcription": {
        "engine": "tone",
        "device": "cpu",
        "sample_rate": 16000,
        "overwrite": True,
        "batch_limit": None,
        "tone": {
            "mode": "python",
            "python_import": "tone",
            "python_fn": "asr_transcribe",
            "model_name": "small",
        },
    },
    "captions": {
        "path": os.path.join("image_process", "image_captions.csv"),
        "text_column": "caption_ru",  # используем только ОДНУ колонку
    },
    "training": {
        "model_path": os.path.join("output_data", "model.pkl"),
        "kfold": 3,
        "gbrt": {"n_estimators": 300, "max_depth": 3, "random_state": 42},
    },
}


def _clean_header(s: str) -> str:
    """Нормализуем имя колонки: низкий регистр, пробелы, убираем BOM/непеч.символы."""
    if not isinstance(s, str):
        return s
    s = s.replace("\ufeff", "")  # BOM
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s


def _rename_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _clean_header(c) for c in df.columns})


def _autodetect_image_col(
    cap_df: pd.DataFrame, expected: str = "картинка из вопроса"
) -> Optional[str]:
    if expected in cap_df.columns:
        return expected

    # список частых вариантов имён (уже нормализованных, т.е. lower/trim/bom-removed)
    synonyms = [
        "картинка из вопроса",
        "image",
        "image_url",
        "url",
        "img",
        "picture",
        "link",
        "картинка",
        "путь к изображению",
        "path",
        "file",
        "filepath",
    ]
    for s in synonyms:
        if s in cap_df.columns:
            return s

    def is_image_like(series: pd.Series) -> float:
        """Доля значений, напоминающих URL/путь к картинке."""
        if series.dtype == object:
            s = series.astype(str).str.lower()
            mask = s.str.contains(
                r"http://|https://", regex=True, na=False
            ) | s.str.contains(
                r"\.png$|\.jpg$|\.jpeg$|\.gif$|\.webp$", regex=True, na=False
            )
            return float(mask.mean())
        return 0.0

    scores = {c: is_image_like(cap_df[c]) for c in cap_df.columns}
    if not scores:
        return None
    best_col, best_score = max(scores.items(), key=lambda kv: kv[1])
    return (
        best_col if best_score > 0.15 else None
    )  # небольшой порог, чтобы отсечь мусор


def _norm_img_key(x: str) -> str:
    """
    Преобразует URL/путь в стабильный ключ:
      - берём только basename пути,
      - убираем query/fragment,
      - приводим к lower.
    """
    if not isinstance(x, str) or not x.strip():
        return ""
    x = x.strip()
    try:
        pr = urlparse(x)
        path = pr.path if pr.scheme in ("http", "https") else x
    except Exception:
        path = x
    path = path.replace("\\", "/")
    base = os.path.basename(path) or path.split("/")[-1]
    base = base.split("?", 1)[0].split("#", 1)[0]
    return base.strip().lower()


def load_config(path: str = "models.yml") -> dict:
    """Загрузка конфига с наложением дефолтов."""
    cfg = DEFAULT_CFG.copy()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        # неглубокое слияние — достаточно для наших ключей
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    else:
        print("[cfg] models.yml не найден — использую дефолты.")
    return cfg


def robust_read_table(path: str) -> pd.DataFrame:
    """
    Надёжное чтение CSV/TSV: пробуем ; , \t и кодировки utf-8-sig/cp1251.
    """
    attempts = [
        dict(sep=";", encoding="utf-8-sig"),
        dict(sep=",", encoding="utf-8-sig"),
        dict(sep="\t", encoding="utf-8-sig"),
        dict(sep=";", encoding="cp1251"),
        dict(sep=",", encoding="cp1251"),
    ]
    last_err = None
    for a in attempts:
        try:
            df = pd.read_csv(path, **a)
            if df.shape[1] >= 1:
                return df
        except Exception as e:
            last_err = e
    # универсальный парсер
    try:
        with codecs.open(path, "r", "utf-8") as f:
            sample = f.read(2048)
        sep = csv.Sniffer().sniff(sample).delimiter
        return pd.read_csv(path, sep=sep, encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать {path}: {last_err or e}")


def pick_input_file(input_dir: str) -> str:
    """
    Возвращает путь к первому CSV/XLSX из папки input_data.
    Бросает FileNotFoundError, если папки или файлов нет.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Папка не найдена: {input_dir}")
    files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".csv", ".xlsx", ".xls"))
    ]
    if not files:
        raise FileNotFoundError(f"В {input_dir} нет CSV/XLSX-файлов.")
    files.sort()
    return os.path.join(input_dir, files[0])


def load_noise_patterns(path: str) -> list[str]:
    """
    Загружаем фразы для очистки из текста.
    Если файла нет — используем дефолтный набор.
    """
    path = ""
    if not os.path.exists(path):
        return ["<p>", "</p>", "<br>", "добро пожаловать на экзамен", "начните диалог."]
    pats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                pats.append(s.lower())
    return pats


def _autodetect_image_col(
    cap_df: pd.DataFrame, expected: str = "картинка из вопроса"
) -> Optional[str]:
    if expected in cap_df.columns:
        return expected
    candidates = [
        "картинка из вопроса",
        "image",
        "image_url",
        "url",
        "img",
        "picture",
        "link",
        "картинка",
        "путь к изображению",
        "path",
        "file",
        "filepath",
    ]
    for c in candidates:
        if c in cap_df.columns:
            return c

    # эвристика: колонка, где чаще встречаются .png/.jpg/http
    def is_image_like(series: pd.Series) -> float:
        if series.dtype != object:
            return 0.0
        s = series.astype(str).str.lower()
        mask = s.str.contains(
            r"http://|https://", regex=True, na=False
        ) | s.str.contains(r"\.(png|jpg|jpeg|gif|webp)$", regex=True, na=False)
        return float(mask.mean())

    scores = {c: is_image_like(cap_df[c]) for c in cap_df.columns}
    if not scores:
        return None
    best_col, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_col if best_score > 0.15 else None


def clean_text_noise(text: str, patterns: list[str]) -> str:
    """
    Удаляет из текста заранее известные мусорные подстроки (регистр игнорируем).
    """
    t = str(text)
    low = t.lower()
    for p in patterns:
        low = low.replace(p, " ")
    return " ".join(low.split())


def get_image_captions(df: pd.DataFrame, col: str) -> dict[str, str]:
    """
    Извлекает из датафрейма отображение "ключ картинки" -> "текст подписи".
    Ключ картинки — нормализованный basename пути/URL.
    """
    captions = {}
    for _, row in df.iterrows():
        img_url = row[col]
        caption = str(row["caption_ru"])
        captions[img_url] = caption
    return captions
