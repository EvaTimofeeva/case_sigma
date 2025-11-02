"""
Входная точка.

Читает первый .csv/.xlsx из input_data,
Подмешиваем "Текст с картинки" из image_process/image_captions.csv,
Делаем (можем)(t-one) но сейчас используем данную нам транскрибацию,
Обучаем модель
Сохраняем predictions.csv и metrics.json в output_data
"""

import os
import json
import time
import argparse
import pandas as pd

from utils_app import (
    robust_read_table,  # читает csv/xlsx с русскими разделителями/кодировками
    get_image_captions,
)

import csv

from transcription import transcribe
from text_scoring import build_features, fit_or_load_model, predict_scores_01
from scoring import apply_scoring_with_comments, compute_metrics
from text_cleaning import clean_text

# Абсолютные пути относительно этого файла
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")
CAPTIONS_CSV = os.path.join(BASE_DIR, "image_process", "image_captions.csv")
NOISE_FILE = os.path.join(BASE_DIR, "noise_patterns.txt")

# Параметры
DO_TRANSCRIPTION = False  # делать ли транскрибацию (иначе использовать существующую)

# Колонки
COL_QNUM = "№ вопроса"
COL_QTEXT = "Текст вопроса"
COL_IMAGE = "Картинка из вопроса"
COL_TRANS = "Транскрибация ответа"
COL_AUDIO_TYPO = "Ссылка на оригинальный файл запис"
COL_AUDIO = "Ссылка на оригинальный файл записи"
COL_GOLD = "Оценка экзаменатора"
COL_CAPTION = "Текст с картинки"
COL_PREDICT = "Оценка экзаменатора (модель)"
COL_CLEANED_QUESTION = "Очищенный текст вопроса"
COL_CLEANED_TRANS = "Очищенная транскрибация ответа"
COL_IMAGE_TEXT = "Текст с картинки"
COL_SCORE_01 = "Исходная оценка модели"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def step_0_read_data(in_path: str = "input_data/Данные для кейса.csv") -> pd.DataFrame:
    df = robust_read_table(in_path)

    # Исправление опечатки в названии колонки (если есть)
    if COL_AUDIO_TYPO in df.columns and COL_AUDIO not in df.columns:
        df[COL_AUDIO] = df[COL_AUDIO_TYPO]
        df = df.drop(columns=[COL_AUDIO_TYPO])

    return df


def step_1_add_captions(df: pd.DataFrame) -> pd.DataFrame:
    captions_df = pd.read_csv(CAPTIONS_CSV)
    captions = get_image_captions(captions_df, COL_IMAGE)
    df[COL_CAPTION] = df[COL_IMAGE].map(captions)
    return df


def step_2_transcribe(df: pd.DataFrame) -> pd.DataFrame:
    df = transcribe(
        df,
        col_trans=COL_TRANS,
        col_audio=COL_AUDIO,
        engine="tone"
        if DO_TRANSCRIPTION
        else "none",  # 'none' — не транскрибировать; 'tone' — через T-one
        col_trans_new=COL_TRANS,
    )
    return df


def step_3_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_text(
        df,
        question_col=COL_QTEXT,
        transcript_col=COL_TRANS,
        cleaned_question_col=COL_CLEANED_QUESTION,
        cleaned_transcript_col=COL_CLEANED_TRANS,
        noise_patterns_path=NOISE_FILE,
    )
    return df


def step_4_build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = build_features(
        df,
        col_qtext=COL_CLEANED_QUESTION,
        col_answer=COL_CLEANED_TRANS,
        col_image_text=COL_IMAGE_TEXT,
    )
    return feats


def step_5_fit_or_load_model(df: pd.DataFrame, feats):
    model_path = os.path.join(OUTPUT_DIR, "model.pkl")
    model = fit_or_load_model(df, feats, col_gold=COL_GOLD, model_path=model_path)
    return model


def step_6_apply_scoring_with_comments(df: pd.DataFrame, model, feats) -> pd.DataFrame:
    df[COL_SCORE_01] = predict_scores_01(model, feats)
    df = apply_scoring_with_comments(
        df, col_qnum=COL_QNUM, col_predict=COL_PREDICT, col_score_01=COL_SCORE_01
    )
    return df


def step_7_save_artifacts(df: pd.DataFrame, job_id=None):
    metrics = compute_metrics(df, col_gold=COL_GOLD, col_predict=COL_PREDICT)
    if metrics and not job_id:
        with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if not metrics:
        df = df.drop(columns=[COL_GOLD])
        df = df.rename(columns={COL_PREDICT: COL_GOLD})

    if job_id:
        out_csv = os.path.join(OUTPUT_DIR, f"predictions_{job_id}.csv")
    else:
        out_csv = os.path.join(OUTPUT_DIR, "predictions.csv")

    df.to_csv(
        out_csv, sep=";", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )
    return out_csv, metrics


def main(input_path: str = "input_data/Данные для кейса.csv"):
    t0 = time.time()

    print("=" * 60)
    print("[main] Запуск пайплайна оценки экзаменационных ответов...")
    print("=" * 60)

    # 0) Берём файл из input_path
    df = step_0_read_data(input_path)

    # 1) Подмешиваем подписи к картинкам из image_process/image_captions.csv
    print("[step 1/7] Подмешиваю подписи к картинкам...")
    df = step_1_add_captions(df)

    print(f"[step 2/7] Транскрибация ({'tone' if DO_TRANSCRIPTION else 'none'})...")
    df = step_2_transcribe(df)

    print("[step 3/7] Очистка текстов...")
    df = step_3_clean_text(df)

    print("[step 4/7] Извлекаю фичи...")
    feats = step_4_build_features(df)

    # 4) Обучение/загрузка и предсказания
    print("[step 5/7] Обучение/загрузка модели и предсказание...")
    model = step_5_fit_or_load_model(df, feats)

    # 5) Баллы + комментарий
    print("[step 6/7] Выставляю баллы и комментарии...")
    df = step_6_apply_scoring_with_comments(df, model, feats)

    # 6) Сохранение
    print("[step 7/7] Сохраняю артефакты...")
    out_csv, metrics = step_7_save_artifacts(df)

    print(f"[main] Готово за {time.time() - t0:.1f}s")
    print(f"[main] predictions: {out_csv}")
    print(f"[main] metrics:     {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Пайплайн для оценки экзаменационных ответов"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="input_data/Данные для кейса.csv",
        help="Путь к входному CSV файлу",
    )
    parser.add_argument(
        "--do_transcription",
        action="store_true",
        help="Выполнять транскрибацию аудио с помощью T-one",
    )

    args = parser.parse_args()

    if args.do_transcription:
        DO_TRANSCRIPTION = True

    main(input_path=args.input_data)
