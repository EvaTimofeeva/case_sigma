"""
Готовим простые текстовые признаки и обучаем базовую модель
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from typing import Optional

def _safe_str(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)

def _len_chars(s: str) -> int:
    return len(s)

def _len_words(s: str) -> int:
    return len([w for w in s.split() if w.strip()])

def _token_set(s: str) -> set:
    return set([w for w in s.lower().split() if w.strip()])

def _jaccard(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def build_features(
    df: pd.DataFrame,
    col_qtext: Optional[str] = "Текст вопроса",    
    col_answer: str = "Транскрибация ответа",
    col_image_text: Optional[str] = "Текст с картинки",
) -> pd.DataFrame:
    """
    Возвращает датафрейм с простыми признаками:
      - длины вопроса/ответа/текста с картинок
      - Jaccard(вопрос, ответ)
      - Jaccard(текст_картинки, ответ) — если доступен
    """
    q = df[col_qtext] if (col_qtext in df.columns) else ""
    a = df[col_answer] if (col_answer in df.columns) else ""
    im = df[col_image_text] if (col_image_text and col_image_text in df.columns) else ""

    q = q.astype(str).fillna("")
    a = a.astype(str).fillna("")
    im = im.astype(str).fillna("")

    feats = pd.DataFrame({
        "len_q_chars": q.apply(_len_chars),
        "len_a_chars": a.apply(_len_chars),
        "len_im_chars": im.apply(_len_chars) if len(im) else 0,
        "len_q_words": q.apply(_len_words),
        "len_a_words": a.apply(_len_words),
        "len_im_words": im.apply(_len_words) if len(im) else 0,
        "jac_q_a": [_jaccard(qi, ai) for qi, ai in zip(q, a)],
        "jac_im_a": [_jaccard(ii, ai) for ii, ai in zip(im, a)] if len(im) else 0,
        "qnum": df["№ вопроса"].fillna(0).astype(int),
    })

    # dummies по № вопроса
    feats = pd.concat([feats, pd.get_dummies(feats["qnum"], prefix="qnum", dtype=int)], axis=1)
    return feats

def fit_or_load_model(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    col_gold: str = "Оценка экзаменатора",
    model_path: str = "output_data/model.pkl",
    n_splits: int = 3,
):
    """
    Учит простой GBRT по нормированному таргету (y/max_points) и сохраняет модель.
    Если модель уже есть — грузит её.
    """
    if os.path.exists(model_path):
        return joblib.load(model_path)

    y = df[col_gold].fillna(0).astype(float)
    max_points = df["№ вопроса"].map({1:1,2:2,3:1,4:2}).fillna(1)
    y_norm = (y / max_points).clip(0, 1).values

    X = feats.values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(df), dtype=float)
    maes = []

    for tr, va in kf.split(X):
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)
        model.fit(X[tr], y_norm[tr])
        p = model.predict(X[va]).clip(0,1)
        oof[va] = p
        maes.append(mean_absolute_error(y_norm[va], p))

    print(f"[text scoring] CV MAE (norm 0..1): {np.mean(maes):.4f}")

    # финальная дообученная модель на всем датасете
    final_model = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)
    final_model.fit(X, y_norm)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"[text scoring] Модель сохранена: {model_path}")
    return final_model

def predict_scores_01(model, feats: pd.DataFrame) -> np.ndarray:
    """
    Возвращает нормированный скор (0..1).
    """
    p = model.predict(feats.values)
    return np.clip(p, 0, 1)
