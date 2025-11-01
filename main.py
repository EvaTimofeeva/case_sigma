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
import pandas as pd

from utils_app import (
    pick_input_file,          # берёт первый файл из input_data
    robust_read_table,        # читает csv/xlsx с русскими разделителями/кодировками
    get_image_captions,
)

from transcription import transcribe
from text_scoring import build_features, fit_or_load_model, predict_scores_01
from scoring import apply_scoring_with_comments, compute_metrics
from text_cleaning import clean_text

# Абсолютные пути относительно этого файла
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR     = os.path.join(BASE_DIR, "input_data")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output_data")
CAPTIONS_CSV  = os.path.join(BASE_DIR, "image_process", "image_captions.csv")
NOISE_FILE    = os.path.join(BASE_DIR, "noise_patterns.txt")

# Параметры
DO_TRANSCRIPTION = False  # делать ли транскрибацию (иначе использовать существующую)

# Колонки
COL_QNUM  = "№ вопроса"
COL_QTEXT = "Текст вопроса"
COL_IMAGE = "Картинка из вопроса"
COL_TRANS = "Транскрибация ответа"
COL_AUDIO_TYPO = "Ссылка на оригинальный файл запис"
COL_AUDIO = "Ссылка на оригинальный файл записи"
COL_GOLD  = "Оценка экзаменатора"
COL_CAPTION = "Текст с картинки"
COL_PREDICT = "Оценка экзаменатора (модель)"
COL_CLEANED_QUESTION = "Очищенный текст вопроса"
COL_CLEANED_TRANS = "Очищенная транскрибация ответа"
COL_IMAGE_TEXT = "Текст с картинки"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    t0 = time.time()

    print("=" * 60)
    print("[main] Запуск пайплайна оценки экзаменационных ответов:")
    print("=" * 60)

    # 0) Берём файл из input_data (автоматически — первый .csv/.xlsx)
    in_path = pick_input_file(INPUT_DIR)
    print(f"[main] Входной файл: {in_path}")
    df = robust_read_table(in_path)
    print(f"[main] Датасет: {df.shape}")
    
    # Исправление опечатки в названии колонки (если есть)
    if COL_AUDIO_TYPO in df.columns and COL_AUDIO not in df.columns:
        df[COL_AUDIO] = df[COL_AUDIO_TYPO]
        df = df.drop(columns=[COL_AUDIO_TYPO])

    # 1) Подмешиваем подписи к картинкам из image_process/image_captions.csv
    print("[step 1/7] Подмешиваю подписи к картинкам:")
    captions_df = pd.read_csv(CAPTIONS_CSV)
    captions = get_image_captions(captions_df, COL_IMAGE)
    df[COL_CAPTION] = df[COL_IMAGE].map(captions)

    # 2) Транскрибация
    print(f"[step 2/7] Транскрибация ({'tone' if DO_TRANSCRIPTION else 'none'}):")
    df = transcribe(
        df,
        col_trans=COL_TRANS,
        col_audio=COL_AUDIO,
        engine="tone" if DO_TRANSCRIPTION else "none",  # 'none' — не транскрибировать; 'tone' — через T-one
        col_trans_new=COL_TRANS,
    )
    
    print("[step 3/7] Очистка текстов:")
    df = clean_text(
        df, 
        question_col=COL_QTEXT,
        transcript_col=COL_TRANS,
        cleaned_question_col=COL_CLEANED_QUESTION,
        cleaned_transcript_col=COL_CLEANED_TRANS,
        noise_patterns_path=NOISE_FILE,
    )

    # 3) Фичи
    print("[step 4/7] Извлекаю фичи:")
    feats = build_features(
        df,
        col_qtext=COL_CLEANED_QUESTION,
        col_answer=COL_CLEANED_TRANS,
        col_image_text=COL_IMAGE_TEXT,
    )

    # 4) Обучение/загрузка и предсказания
    print("[step 5/7] Обучение/загрузка модели и предсказание:")
    model_path = os.path.join(OUTPUT_DIR, "model.pkl")
    model = fit_or_load_model(df, feats, col_gold=COL_GOLD, model_path=model_path)
    df["Исходная оценка модели"] = predict_scores_01(model, feats)

    # 5) Баллы + комментарий
    print("[step 6/7] Выставляю баллы и комментарии:")
    df = apply_scoring_with_comments(df, col_qnum=COL_QNUM, col_predict=COL_PREDICT)

    # 6) Сохранение
    print("[step 7/7] Сохраняю артефакты:")
    out_csv = os.path.join(OUTPUT_DIR, "predictions.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    metrics = compute_metrics(df, col_gold=COL_GOLD)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[main] Готово за {time.time()-t0:.1f}s")
    print(f"[main] predictions: {out_csv}")
    print(f"[main] metrics:     {os.path.join(OUTPUT_DIR, 'metrics.json')}")

if __name__ == "__main__":
    main()
