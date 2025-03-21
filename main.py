import os
import sys
import time
import datetime
import logging
import argparse
from database import create_database
from parsing import parse_data
from predict import predict_data
from config import LOGGING_CONFIG, COMPARISON_LOGGING_CONFIG

# Отключаем логи TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает логи TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Отключает логи Keras

# Отключаем интерактивные логи прогресса TensorFlow/Keras
tf.keras.utils.disable_interactive_logging()

# Настройка логгера для записи в файл
logging.basicConfig(
    filename='script.log',  # Имя файла для записи логов
    level=logging.INFO,     # Уровень логирования
    encoding='utf-8',       # Кодировка файла
    format='%(asctime)s - %(levelname)s - %(message)s',  # Формат записи
    filemode='a'            # Режим записи: 'a' - добавление, 'w' - перезапись
)

# # Перенаправление stdout и stderr в логгер
# class LoggerWriter:
#     def __init__(self, level):
#         self.level = level

#     def write(self, message):
#         if message != '\n':  # Игнорируем пустые строки
#             self.level(message)

#     def flush(self):
#         pass

# sys.stdout = LoggerWriter(logging.info)
# sys.stderr = LoggerWriter(logging.error)

# # Настройка логгера для сравнений
# comparison_logger = logging.getLogger('comparison_logger')
# comparison_logger.setLevel(COMPARISON_LOGGING_CONFIG['level'])

# # Добавляем обработчик только если его еще нет
# if not comparison_logger.handlers:
#     file_handler = logging.FileHandler(
#         filename=COMPARISON_LOGGING_CONFIG['filename'],
#         encoding=COMPARISON_LOGGING_CONFIG['encoding'],
#         mode=COMPARISON_LOGGING_CONFIG['filemode']
#     )
#     file_handler.setFormatter(logging.Formatter(COMPARISON_LOGGING_CONFIG['format']))
#     comparison_logger.addHandler(file_handler)

def wait_until_next_interval(interval_minutes=5, seconds_offset=30):
    """
    Ждет до следующего интервала и возвращает время следующего запуска.
    """
    now = datetime.datetime.now()
    current_minute = now.minute
    next_minute = ((current_minute // interval_minutes) + 1) * interval_minutes
    if next_minute >= 60:
        next_minute = 0
        delta_hours = 1
    else:
        delta_hours = 0
    next_time = now.replace(minute=next_minute, second=seconds_offset, microsecond=0)
    if delta_hours > 0:
        next_time += datetime.timedelta(hours=delta_hours)
    if next_time < now:
        next_time += datetime.timedelta(minutes=interval_minutes)
    wait_seconds = (next_time - now).total_seconds()
    logging.info(f"Следующий запуск в {next_time.strftime('%H:%M:%S')}")
    time.sleep(wait_seconds)
    return next_time

def main(interval_minutes=5, seconds_offset=30):
    logging.info("Скрипт запущен. Для остановки нажмите Ctrl+C.")
    try:
        # Создаем базу данных и таблицы, если они не существуют
        create_database()

        while True:
            try:
                # Записываем время следующего цикла
                next_time = wait_until_next_interval(interval_minutes, seconds_offset)
                logging.info(f"Следующий цикл начнется в {next_time.strftime('%H:%M:%S')}")

                # Запуск парсинга и предсказаний
                logging.info("Запуск парсинга...")
                parse_data()
                logging.info("Запуск предсказаний...")
                predict_data()
            except Exception as e:
                logging.error(f"Ошибка: {e}")
                time.sleep(60)  # Ждем 60 секунд перед повторной попыткой
    except KeyboardInterrupt:
        logging.info("Скрипт остановлен.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=5, help='Интервал между запусками (в минутах)')
    parser.add_argument('--offset', type=int, default=30, help='Смещение в секундах')
    args = parser.parse_args()
    main(interval_minutes=args.interval, seconds_offset=args.offset)