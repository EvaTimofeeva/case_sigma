"""
Этим файлом мы получаем описания к картинкам с помощью модели BLIP и переводим их на русский язык
файл не используется в запущенной программе.
Формируем датасет из 11 уникальных картинок и их описаний на русском(перевод с английского)
Источник ссылок: input_data/<ваш csv>, колонка "Картинка из вопроса"
Выход: output_data/image_captions.csv
"""

import os
import re
import sys
from typing import List, Optional
from io import BytesIO

import requests
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianTokenizer,
    MarianMTModel,
)

INPUT_DIR = "input_data"
OUTPUT_DIR = "output_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# если в папке несколько csv — берём первый попавшийся
def _pick_input_csv() -> str:
    for name in os.listdir(INPUT_DIR):
        if name.lower().endswith(".csv"):
            return os.path.join(INPUT_DIR, name)
    print(f"[error] В {INPUT_DIR} не найден .csv с данными.", file=sys.stderr)
    sys.exit(1)


INPUT_CSV = _pick_input_csv()
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "image_captions.csv")

IMAGE_COL_CANDIDATES = [
    "Картинка из вопроса",
    "Ссылка на картинку",
    "Картинка",
    "image_url",
]


def robust_read_csv(path: str) -> pd.DataFrame:
    attempts = [
        dict(encoding="utf-8-sig", sep=";"),
        dict(encoding="utf-8-sig", sep=","),
        dict(encoding="cp1251", sep=";"),
        dict(encoding="cp1251", sep=","),
        dict(encoding="utf-8", sep=None),
    ]
    last = None
    for a in attempts:
        try:
            return pd.read_csv(path, **a, engine="python")
        except Exception as e:
            last = e
    raise RuntimeError(f"Не удалось прочитать CSV: {last}")


def is_image_response(resp: requests.Response) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    return ct.startswith("image/")


def download_image(url: str, timeout: int = 12) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        if not is_image_response(r):
            pass
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img
    except Exception:
        return None


# BLIP + перевод на русский
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
print(f"[init] Загружаю BLIP: {BLIP_MODEL}")
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
blip_model.eval()

# переводчик EN->RU
MT_MODEL = "Helsinki-NLP/opus-mt-en-ru"
print(f"[init] Загружаю переводчик: {MT_MODEL}")
mt_tok = MarianTokenizer.from_pretrained(MT_MODEL)
mt_model = MarianMTModel.from_pretrained(MT_MODEL)
mt_model.eval()


@torch.no_grad()
def blip_caption(image: Image.Image, max_new_tokens: int = 30) -> str:
    inp = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inp, max_new_tokens=max_new_tokens)
    cap = blip_processor.decode(out[0], skip_special_tokens=True)
    return cap.strip()


CYR = re.compile(r"[А-Яа-яЁё]")


def translate_en_to_ru(texts: List[str], max_len: int = 256) -> List[str]:
    # батчевый перевод EN->RU. Если строка уже русская — возвращаем как есть
    result = []
    batch = []
    idxs = []
    for i, t in enumerate(texts):
        t = t or ""
        if CYR.search(t):
            result.append(t)
        else:
            result.append(None)
            batch.append(t)
            idxs.append(i)
    if batch:
        tokens = mt_tok(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,
        )
        with torch.no_grad():
            gen = mt_model.generate(**tokens, max_new_tokens=max_len // 2)
        outs = [mt_tok.decode(g, skip_special_tokens=True) for g in gen]
        for i, ru in zip(idxs, outs):
            result[i] = ru.strip()
    return result


print(f"[info] Входной файл: {INPUT_CSV}")
df = robust_read_csv(INPUT_CSV)

# ищем колонку с изображениями
image_col = None
for c in IMAGE_COL_CANDIDATES:
    if c in df.columns:
        image_col = c
        break
if not image_col:
    raise RuntimeError(
        f"Не нашёл колонку с ссылками на изображения. "
        f"Ожидал одно из: {IMAGE_COL_CANDIDATES}. Доступные: {list(df.columns)}"
    )

urls_raw = df[image_col].astype(str).str.strip()
# убираем пустые и очевидный мусор (не http/https)
urls_raw = urls_raw[urls_raw.str.startswith(("http://", "https://"))]

# оставляем уникальные в порядке появления
seen = set()
unique_urls = []
for u in urls_raw:
    if u and u not in seen:
        seen.add(u)
        unique_urls.append(u)
    if len(unique_urls) == 11:
        break

if len(unique_urls) < 11:
    print(
        f"[warn] нашёл только {len(unique_urls)} уникальных ссылок (< 11). Будут использованы все, что есть."
    )

print(f"[info] К обработке: {len(unique_urls)} уникальных изображений.")

rows = []
for url in tqdm(unique_urls, desc="Картинки"):
    img = download_image(url)
    if img is None:
        rows.append(
            dict(image_url=url, caption_en="", caption_ru="", status="Ошибка загрузки")
        )
        continue
    try:
        cap_en = blip_caption(img)
    except Exception as e:
        rows.append(
            dict(
                image_url=url,
                caption_en="",
                caption_ru="",
                status=f"Ошибка генерации: {e}",
            )
        )
        continue

    cap_ru = translate_en_to_ru([cap_en])[0]
    rows.append(dict(image_url=url, caption_en=cap_en, caption_ru=cap_ru, status="ok"))

out = pd.DataFrame(rows, columns=["image_url", "caption_en", "caption_ru", "status"])
out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n[done] Сохранено: {OUTPUT_CSV}")
print(out)
