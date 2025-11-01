"""
обёртка транскрибации:
engine='none'  -> ничего не считаем, просто гарантим колонку.
engine='tone'  -> вызов T-one для транскрибации аудио.
удаляем "мусор" по шаблонам.
"""

import os
import tempfile
import requests
import pandas as pd
from tqdm import tqdm

try:
    from tone import StreamingCTCPipeline, read_audio
    from tone.decoder import DecoderType
except ImportError:
    print("[warning] Модуль 'tone' не найден, транскрибация через T-one недоступна.")
    StreamingCTCPipeline = None
    read_audio = None
    DecoderType = None


def _download_if_url(path_or_url: str) -> str:
    """Если это URL — скачать во временный файл, вернуть локальный путь."""
    s = str(path_or_url or "").strip()
    if not s:
        return ""
    if s.lower().startswith(("http://", "https://")):
        r = requests.get(s, timeout=30)
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        return tmp
    return s  # локальный путь


def transcribe(
    df: pd.DataFrame,
    col_trans: str = "Транскрибация ответа",
    col_audio: str = "Ссылка на оригинальный файл запис",
    overwrite: bool = False,
    engine: str = "none",  # 'none' | 'tone'
    col_trans_new: str = "Транскрибация ответа",
) -> pd.DataFrame:
    """
    Гарантирует наличие колонки транскрибации и (опц.) очищает её от мусора.

    - Если engine='none': просто создаём/оставляем колонку как есть.
    - Если engine='tone': тут должен быть ваш вызов T-one (HTTP-клиент и т.п.).
    - overwrite=True перезаписывает существующие значения.

    Возвращает копию df.
    """
    out = df.copy()
    if col_trans not in out.columns:
        out[col_trans] = ""

    # 1) транскрибация
    if engine == "none":
        # ничего не делаем, просто копируем колонку
        out[col_trans_new] = out[col_trans].astype(str).fillna("")
    elif engine == "tone":
        model = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)

        transcriptions = []
        for idx, row in tqdm(out.iterrows(), total=len(out), desc="Транскрибация"):
            audio_path = str(row.get(col_audio, "")).strip()
            if not audio_path:
                transcriptions.append("")
                continue
            audio_path = _download_if_url(audio_path)
            audio = read_audio(audio_path)
            transcript = model.forward_offline(audio)
            transcriptions.append(transcript)

        out[col_trans_new] = transcriptions

    return out
