# config.py

import os
import logging
from pathlib import Path

PREDICTION_ADJUSTMENT = {
    # === Управление повторениями и базовой логикой ===
    'MAX_REPEATS': 2,        # Макс. повторений поля подряд перед принудительной сменой
    'NUMBERS_TO_ADJUST': 4,  # Сколько чисел менять при корректировке предсказания
    'USE_TRENDS': True,       # Использовать анализ горячих/холодных чисел (True/False)
    
    # === Пороги для активации корректировки ===
    'MIN_MATCH_TO_ADJUST': 3, # Минимум совпадений в предыдущих предсказаниях для активации корректировки
    
    # === Стратегии замены чисел ===
    'ADJUSTMENT_STRATEGY': 'hot', # Стратегия выбора чисел:
                                    # 'smart' - умный выбор, 'hot' - только горячие,
                                    # 'cold' - только холодные, 'random' - случайные
    
    # === Критерии для проблемных чисел ===
    'MIN_SUCCESS_RATE': 30,  # Минимальный процент успешных предсказаний для числа (0-100)
    'MAX_MISSES': 5,         # Максимальное допустимое количество промахов подряд
    
    # === Параметры для холодных чисел ===
    'COLD_RANK_THRESHOLD': 5,    # Числа в топ-N холодных считаются особо проблемными
    'COLD_RANK_MAX_MISSES': 2,   # Макс. промахов для холодных чисел из топ-N
    
    # === Статистическая значимость ===
    'MIN_ATTEMPTS': 10       # Минимальное количество попыток предсказания числа 
                             # для учёта его статистики
}

# ====================== #
#  Базовые настройки     #
# ====================== #
BASE_DIR = Path(__file__).parent.resolve()
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# ====================== #
#  Настройки TensorFlow  #
# ====================== #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает логи TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем oneDNN

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.keras.utils.disable_interactive_logging()

# ====================== #
#  Настройки базы данных #
# ====================== #
DATABASE_PATH = BASE_DIR / 'results.db'

# ====================== #
#  Настройки модели      #
# ====================== #
SEQUENCE_LENGTH = 50  # Количество используемых комбинаций
NUM_CLASSES = 4       # Для поля 1-4
COMBINATION_LENGTH = 8  # 8 чисел в комбинации
NUMBERS_RANGE = 20    # Для чисел 1-20
BATCH_SIZE = 128
RETRAIN_HOURS = 0     # Интервал переобучения в часах (0-24)
NEW_DATA_THRESHOLD = 5  # Минимальное новых записей для обучения
MODEL_INPUT_SHAPE = (SEQUENCE_LENGTH, COMBINATION_LENGTH)
MODEL_SAVE_PATH = BASE_DIR / 'models'

# ====================== #
#  Настройки отчетов     #
# ====================== #
REPORT_FREQUENCY = 1   # Частота генерации отчетов (в циклах)
REPORT_PERIOD_DAYS = 30  # Период для статистики (в днях)

# ====================== #
#  Настройки парсинга    #
# ====================== #
CHROMEDRIVER_PATH = BASE_DIR / 'chromedriver' / 'chromedriver.exe'
DRIVER_DIR = "drivers"  # Папка для хранения драйверов
AUTO_UPDATE_DRIVER = True  # Автоматическое обновление драйвера
DRIVER_VERSION = None  # Конкретная версия или None для автоматического определения
FORCE_DOWNLOAD = False  # Принудительное скачивание новой версии
CHROME_OPTIONS = {
    'headless': True,       # Работа в фоновом режиме
    'disable-gpu': True,    # Отключение GPU для headless
    'no-sandbox': True,     # Для Docker/CI
    'disable-dev-shm-usage': True  # Для ограниченных ресурсов
}
DAYS_TO_PARSE = 2  # Максимальное количество дней для парсинга
BASE_URL = 'https://sportpari.by/ru/play/blits/game-results?filter_results_type=draw_date&draw_date={date}&page={page}'

# ====================== #
#  Настройки логирования #
# ====================== #
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# LOGGING_CONFIG = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'formatters': {
#         'standard': {
#             'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             'datefmt': DATETIME_FORMAT
#         },
#     },
#     'handlers': {
#         'file': {
#             'class': 'logging.FileHandler',
#             'filename': LOGS_DIR / 'script.log',
#             'encoding': 'utf-8',
#             'formatter': 'standard',
#             'mode': 'a'
#         },
#         'console': {
#             'class': 'logging.StreamHandler',
#             'formatter': 'standard'
#         },
#     },
#     'root': {
#         'handlers': ['file', 'console'],
#         'level': 'INFO',
#     },
#     'loggers': {
#         'tensorflow': {
#             'handlers': ['file'],
#             'level': 'ERROR',
#             'propagate': False
#         },
#     }
# }
LOGGING_CONFIG = {
    'level': logging.DEBUG,
    'encoding': 'utf-8',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': [
        # Вывод в консоль
        logging.StreamHandler(),
        # Запись в файл
        logging.FileHandler(
            filename='logs/script.log',
            mode='a',
            encoding='utf-8'
        )
    ]
}

# Валидация значений
assert DAYS_TO_PARSE > 0, "DAYS_TO_PARSE должен быть положительным"
assert 0 <= RETRAIN_HOURS <= 24, "RETRAIN_HOURS должен быть между 0 и 24"