# config.py

import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает логи TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем oneDNN
import logging
import tensorflow as tf
# Отключаем логи Keras и интерактивные логи
tf.get_logger().setLevel('ERROR')  # Отключает логи Keras
tf.keras.utils.disable_interactive_logging()  # Отключаем интерактивные логи

# Базовые настройки
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Настройки базы данных
DATABASE_PATH = 'results.db'

# Настройки моделей
SEQUENCE_LENGTH = 30  # колличество используемых комбинаций
NUM_CLASSES = 4 # Для поля 1-4
COMBINATION_LENGTH = 8 # 8 чисел в комбинации
NUMBERS_RANGE = 20 # Для чисел 1-20
BATCH_SIZE = 128
RETRAIN_HOURS = 0      # интервал переобучения в часах (от 0 до 24)
NEW_DATA_THRESHOLD = 1 # минимальное новых записей для обучения
MODEL_INPUT_SHAPE = (SEQUENCE_LENGTH, COMBINATION_LENGTH)  # Автоматически рассчитывается
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')  # Путь к директории models

# Настройки парсинга
CHROMEDRIVER_PATH = os.path.join(os.path.dirname(__file__), 'chromedriver-win64', 'chromedriver-win64', 'chromedriver.exe')
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

# Настройки TensorFlow
TF_CPP_MIN_LOG_LEVEL = '3'  # Отключает логи TensorFlow
TF_ENABLE_ONEDNN_OPTS = '0'  # Отключаем oneDNN

# Настройки TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает логи TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем oneDNN

