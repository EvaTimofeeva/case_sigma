"""
Flask web application for the scoring pipeline
"""
import os
import json
import time
import uuid
import threading

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import csv

from main import (
    step_0_read_data, step_1_add_captions, step_2_transcribe,
    step_3_clean_text, step_4_build_features, step_5_fit_or_load_model,
    step_6_apply_scoring_with_comments,
    COL_GOLD,
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'input_data'

jobs = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pipeline(job_id: str, input_path: str):
    """Run the pipeline in a background thread"""
    jobs[job_id] = {
        'status': 'running',
        'current_step': 0,
        'total_steps': 7,
        'message': 'Запускаю пайплайн обработки...',
        'steps': [
            {'name': 'Чтение данных', 'status': 'pending'},
            {'name': 'Добавление подписей к картинкам', 'status': 'pending'},
            {'name': 'Транскрибация (пропускаем)', 'status': 'pending'},
            {'name': 'Очистка текстов', 'status': 'pending'},
            {'name': 'Построение признаков', 'status': 'pending'},
            {'name': 'Загрузка модели', 'status': 'pending'},
            {'name': 'Применение оценки', 'status': 'pending'},
            {'name': 'Сохранение результатов', 'status': 'pending'},
        ],
        'output_file': None,
        'metrics': None,
        'error': None,
        'start_time': time.time()
    }
    
    try:
        # Step 0: Read data
        jobs[job_id]['current_step'] = 0
        jobs[job_id]['steps'][0]['status'] = 'running'
        jobs[job_id]['message'] = 'Читаю данные из файла...'
        df = step_0_read_data(input_path)
        jobs[job_id]['steps'][0]['status'] = 'completed'
        
        # Step 1: Add captions
        jobs[job_id]['current_step'] = 1
        jobs[job_id]['steps'][1]['status'] = 'running'
        jobs[job_id]['message'] = 'Добавляю подписи к картинкам...'
        df = step_1_add_captions(df)
        jobs[job_id]['steps'][1]['status'] = 'completed'
        
        # Step 2: Transcribe
        jobs[job_id]['current_step'] = 2
        jobs[job_id]['steps'][2]['status'] = 'running'
        jobs[job_id]['message'] = 'Выполняю транскрибацию...'
        df = step_2_transcribe(df)
        jobs[job_id]['steps'][2]['status'] = 'completed'
        
        # Step 3: Clean text
        jobs[job_id]['current_step'] = 3
        jobs[job_id]['steps'][3]['status'] = 'running'
        jobs[job_id]['message'] = 'Очищаю тексты...'
        df = step_3_clean_text(df)
        jobs[job_id]['steps'][3]['status'] = 'completed'
        
        # Step 4: Build features
        jobs[job_id]['current_step'] = 4
        jobs[job_id]['steps'][4]['status'] = 'running'
        jobs[job_id]['message'] = 'Извлекаю признаки...'
        feats = step_4_build_features(df)
        jobs[job_id]['steps'][4]['status'] = 'completed'
        
        # Step 5: Fit or load model
        jobs[job_id]['current_step'] = 5
        jobs[job_id]['steps'][5]['status'] = 'running'
        jobs[job_id]['message'] = 'Обучаю модель...'
        model = step_5_fit_or_load_model(df, feats)
        jobs[job_id]['steps'][5]['status'] = 'completed'
        
        # Step 6: Apply scoring
        jobs[job_id]['current_step'] = 6
        jobs[job_id]['steps'][6]['status'] = 'running'
        jobs[job_id]['message'] = 'Применяю оценку...'
        df = step_6_apply_scoring_with_comments(df, model, feats)
        jobs[job_id]['steps'][6]['status'] = 'completed'
        
        # Step 7: Save artifacts
        jobs[job_id]['current_step'] = 7
        jobs[job_id]['steps'][7]['status'] = 'running'
        jobs[job_id]['message'] = 'Сохраняю результаты...'
        
        # Create job-specific output file
        job_output_file = os.path.join(OUTPUT_DIR, f"predictions_{job_id}.csv")
        df.to_csv(job_output_file, index=False, encoding="utf-8-sig", sep=';', quoting=csv.QUOTE_ALL)
        
        metrics = None
        if COL_GOLD in df.columns:
            from scoring import compute_metrics
            try:
                metrics = compute_metrics(df, col_gold=COL_GOLD)
                metrics_file = os.path.join(OUTPUT_DIR, f"metrics_{job_id}.json")
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
            except Exception as e:
                pass
        
        jobs[job_id]['steps'][7]['status'] = 'completed'
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['message'] = 'Пайплайн завершён успешно!'
        jobs[job_id]['output_file'] = job_output_file
        jobs[job_id]['metrics'] = metrics
        jobs[job_id]['elapsed_time'] = time.time() - jobs[job_id]['start_time']
        
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['message'] = f'Ошибка: {str(e)}'
        import traceback
        traceback.print_exc()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start pipeline"""
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не выбран'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if file:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(INPUT_DIR, f"upload_{job_id}_{filename}")
        file.save(input_path)
        
        # Start pipeline in background thread
        thread = threading.Thread(target=run_pipeline, args=(job_id, input_path))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'Файл загружен, пайплайн запущен'
        })


@app.route('/calculate_metrics', methods=['POST'])
def calculate_metrics():
    """Run pipeline on default dataset to calculate metrics"""
    default_file = os.path.join(INPUT_DIR, "Данные для кейса.csv")
    if not os.path.exists(default_file):
        return jsonify({'error': 'Файл по умолчанию не найден'}), 404
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Start pipeline in background thread
    thread = threading.Thread(target=run_pipeline, args=(job_id, default_file))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': 'Запущен расчёт метрик на исходном датасете'
    })


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get current status of a job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'status': job['status'],
        'current_step': job['current_step'],
        'total_steps': job['total_steps'],
        'message': job['message'],
        'steps': job['steps'],
        'has_error': job['status'] == 'error',
        'error': job.get('error')
    }
    
    if job['status'] == 'completed':
        response['output_file'] = job['output_file']
        response['metrics'] = job.get('metrics')
        response['elapsed_time'] = job.get('elapsed_time', 0)
    
    return jsonify(response)


@app.route('/download/<job_id>')
def download_file(job_id):
    """Download the output file"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'completed' or not job['output_file']:
        return jsonify({'error': 'File not ready'}), 404
    
    output_file = job['output_file']
    if not os.path.exists(output_file):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        output_file,
        as_attachment=True,
        download_name=f'predictions_{job_id}.csv',
        mimetype='text/csv'
    )


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

