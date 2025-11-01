
"""Очистка транскрибации от мусора"""
from utils_app import load_noise_patterns, clean_text_noise



def clean_text(df,
               question_col,
               transcript_col,
               cleaned_question_col,
               cleaned_transcript_col,
               noise_patterns_path: str = "noise_patterns.txt",
    ):
    patterns = load_noise_patterns(noise_patterns_path)
    questions = df[question_col]
    transcripts = df[transcript_col]
    cleaned_questions = [clean_text_noise(q, patterns) for q in questions]  
    cleaned_transcripts = [clean_text_noise(t, patterns) for t in transcripts]

    df[cleaned_question_col] = cleaned_questions
    df[cleaned_transcript_col] = cleaned_transcripts
    return df