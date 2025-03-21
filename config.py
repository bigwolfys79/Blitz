# config.py

import os
import logging

# Определяем BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Настройки базы данных
DATABASE_PATH = 'results.db'

# Настройки моделей
SEQUENCE_LENGTH = 8
NUM_CLASSES = 4
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')  # Путь к директории models

# Настройки парсинга
# Относительный путь к chromedriver
CHROMEDRIVER_PATH = os.path.join(os.path.dirname(__file__), 'chromedriver-win64', 'chromedriver-win64', 'chromedriver.exe')
# CHROMEDRIVER_PATH = 'F:\\123\\125\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe'
BASE_URL = 'https://sportpari.by/ru/play/blits/game-results?filter_results_type=draw_date&draw_date={date}&page={page}'

# Настройка логирования
LOGGING_CONFIG = {
    'filename': 'script.log',
    'level': logging.INFO,
    'encoding': 'utf-8',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'filemode': 'w'
}

# Настройка логирования для сравнений
COMPARISON_LOGGING_CONFIG = {
    'filename': 'comparison.log',  # Отдельный файл для логов сравнений
    'level': logging.INFO,
    'encoding': 'utf-8',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'filemode': 'a'  # Режим добавления, чтобы не перезаписывать файл
}