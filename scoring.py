"""
Выставление итоговой оценки и комментария по правилам и расчёт метрик качества
"""

import numpy as np
import pandas as pd

_MAX_POINTS = {1: 1, 2: 2, 3: 1, 4: 2}


def _points_from_score01(score01: float, qnum: int) -> int:
    """Перевод нормированного скора (0..1) в баллы с учётом номера вопроса."""
    score01 = float(np.clip(score01, 0.0, 1.0))
    maxp = _MAX_POINTS.get(int(qnum), 1)
    return int(round(score01 * maxp))


def _make_comment(row: pd.Series) -> str:
    """
    Формирует человекочитаемый комментарий на основе простых критериев:
      1) единичные ошибки/несогласованности не штрафуем сильно,
      2) акцент не учитываем,
      3) выполнена ли коммуникативная задача (есть смысловой ответ),
      4) предложения преимущественно полные (грубая эвристика по длине/словам).
    Требует, чтобы в df уже были 'Транскрибация ответа', 'Текст вопроса' и/или 'Текст с картинки'.
    """
    qtext = str(row.get("Текст вопроса", "")).strip()
    imgtext = str(row.get("Текст с картинки", "")).strip()
    ans = str(row.get("Транскрибация ответа", "")).strip()

    # Набросок эвристик
    # коммуникативная задача: есть ли смысловые пересечения с вопросом или с текстом с картинки
    def _jacc(a, b):
        A = set([w.lower() for w in a.split() if w.strip()])
        B = set([w.lower() for w in b.split() if w.strip()])
        return (len(A & B) / len(A | B)) if (A and B) else 0.0

    jac_q = _jacc(qtext, ans)
    jac_im = _jacc(imgtext, ans)

    flags = []

    if jac_q > 0.1 or jac_im > 0.1:
        flags.append("коммуникативная задача выполнена (ответ по теме)")
    else:
        flags.append("слабая связанность с вопросом/картинкой")

    sentences = [s for s in ans.replace("!", ".").replace("?", ".").split(".")]
    sentences_len = [len(s.split()) for s in sentences]
    has_long_sentences = any([length for length in sentences_len if length >= 10])
    if has_long_sentences:
        flags.append("полные предложения присутствуют")
    else:
        flags.append("малоразвёрнутые реплики")

    return "; ".join(flags)


def apply_scoring_with_comments(
    df: pd.DataFrame,
    col_qnum: str,
    col_predict: str,
    col_score_01: str = "Исходная оценка модели",
) -> pd.DataFrame:
    """
    Проставляет итоговую колонку "Оценка экзаменатора (модель)" и "комментарий".
    """
    out = df.copy()

    if col_score_01 not in out.columns:
        # Если модель ещё не посчитала скор — создадим нули, чтобы не падать.
        out[col_score_01] = 0.0

    # Итоговый балл по каждому вопросу
    out[col_predict] = [
        _points_from_score01(s, q)
        for s, q in zip(out["Исходная оценка модели"], out[col_qnum])
    ]

    # Комментарий по критериям
    out["комментарий"] = out.apply(_make_comment, axis=1)

    return out


def compute_metrics(
    df: pd.DataFrame,
    col_gold: str = "Оценка экзаменатора",
    col_predict: str = "Оценка экзаменатора (модель)",
):
    """
    Считает MAE в баллах по всей выборке и по каждому номеру вопроса.
    """
    if col_gold not in df.columns or col_predict not in df.columns:
        return None

    mask = (~df[col_gold].isna()) & (~df[col_predict].isna())
    if not mask.any():
        return None

    res = {}

    y_true = df.loc[mask, col_gold].astype(float).values
    y_pred = df.loc[mask, col_predict].astype(float).values
    mae_overall = float(np.mean(np.abs(y_true - y_pred)))
    res["mae_overall"] = mae_overall

    # По каждому номеру вопроса
    by_q = {}
    for q in [1, 2, 3, 4]:
        m = mask & (df["№ вопроса"] == q)
        if m.any():
            yt = df.loc[m, col_gold].astype(float).values
            yp = df.loc[m, col_predict].astype(float).values
            by_q[str(q)] = float(np.mean(np.abs(yt - yp)))
    res["mae_by_question"] = by_q
    return res
