import os
import time
import signal
from config import LOGGING_CONFIG
import logging
from datetime import datetime, timedelta
from typing import Optional
from logging.handlers import RotatingFileHandler


from predict import LotteryPredictor
from parsing import parse_data
from LSTM_model import train_and_save_model
from train_models import ModelTrainChecker
from database import DatabaseManager, calculate_pages_to_parse

# Настройка логгера с записью только в файл
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Простой FileHandler с перезаписью
    file_handler = logging.FileHandler(
        filename='logs/script.log',
        mode='w',
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logging.getLogger('tensorflow').propagate = False

# Инициализируем логирование
setup_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join('models', 'best_lstm_model.keras')

class GracefulExit:
    stop = False

    @staticmethod
    def signal_handler(signal_received, frame):
        logger.info("Получен сигнал завершения (Ctrl+C). Завершаем выполнение...")
        GracefulExit.stop = True

signal.signal(signal.SIGINT, GracefulExit.signal_handler)
signal.signal(signal.SIGTERM, GracefulExit.signal_handler)

class ModelTrainer:
    def __init__(self):
        self.checker = ModelTrainChecker()
        os.makedirs('models', exist_ok=True)

    def ensure_model_exists(self) -> bool:
        """Проверяет и создает модель при необходимости."""
        if os.path.exists(MODEL_PATH):
            try:
                import tensorflow as tf
                tf.keras.models.load_model(MODEL_PATH)
                return True
            except Exception as e:
                logger.warning(f"Модель повреждена: {e}. Пересоздаем...")
                os.remove(MODEL_PATH)

        logger.info("Запуск первоначального обучения...")
        return self.run_training_cycle()

    def run_training_cycle(self) -> bool:
        """Выполняет цикл обучения модели."""
        try:
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                try:
                    should_train, reason = self.checker._check_should_train(cursor)
                    if not should_train:
                        logger.info(f"Обучение не требуется: {reason}")
                        return False

                    training_data = self.checker.get_training_model(cursor, incremental=False)
                    if not training_data:
                        logger.error("Не удалось получить данные для обучения")
                        return False

                    model_result = train_and_save_model(training_data)
                    if not model_result.get('success', False):
                        logger.error(f"Ошибка обучения: {model_result.get('message', 'Unknown error')}")
                        return False

                    self.checker.update_training_info(
                        data_count=model_result['data_count'],
                        accuracy=model_result.get('accuracy')
                    )
                    db.connection.commit()
                    return True
                finally:
                    cursor.close()
        except Exception as ex:
            logger.error(f"Ошибка обучения: {str(ex)}")
            return False

def wait_until_next_interval(interval_minutes: int = 5, seconds_offset: int = 30) -> Optional[datetime]:
    """Ожидает следующего интервала выполнения с проверкой флага остановки."""
    now = datetime.now()
    next_minute = ((now.minute // interval_minutes) + 1) * interval_minutes
    delta_hours = 0
    
    if next_minute >= 60:
        next_minute = 0
        delta_hours = 1
    
    next_time = now.replace(
        minute=next_minute, 
        second=seconds_offset, 
        microsecond=0
    ) + timedelta(hours=delta_hours)
    
    if next_time < now:
        next_time += timedelta(minutes=interval_minutes)
    
    wait_seconds = (next_time - now).total_seconds()
    logger.info(f"Следующий запуск в {next_time.strftime('%H:%M:%S')}")
    
    # Разбиваем ожидание на небольшие интервалы для проверки флага
    while wait_seconds > 0 and not GracefulExit.stop:
        time.sleep(min(1, wait_seconds))  # Проверяем каждую секунду
        wait_seconds = (next_time - datetime.now()).total_seconds()
    
    return next_time if not GracefulExit.stop else None


def main():
    """Точка входа в приложение."""
    logger.info("=" * 50)
    logger.info("Инициализация приложения")

    # Инициализация и проверка модели
    trainer = ModelTrainer()
    if not trainer.ensure_model_exists():
        logger.critical("Не удалось инициализировать модель! Завершение работы.")
        return

    predictor = LotteryPredictor()
    logger.info("Приложение готово к работе")
    logger.info("=" * 50)

    try:
        while not GracefulExit.stop:
            cycle_start = datetime.now()
            logger.info(f"\n{'-'*50}\nЦикл начат: {cycle_start.strftime('%H:%M:%S')}")

            # Проверяем флаг перед каждой операцией
            if GracefulExit.stop:
                break

            next_run = wait_until_next_interval()
            if GracefulExit.stop or next_run is None:
                break

            if GracefulExit.stop:
                break

            with DatabaseManager() as db:
                try:
                    if GracefulExit.stop:
                        break

                    # Парсинг данных
                    pages, _ = calculate_pages_to_parse()
                    parse_data(pages_to_parse=pages or 1)

                    if GracefulExit.stop:
                        break

                    # Обучение модели
                    trainer.run_training_cycle()

                    if GracefulExit.stop:
                        break

                    # Предсказание
                    if predictor.predict_and_save():
                        predictor.check_last_prediction()

                except Exception as e:
                    logger.error(f"Ошибка в цикле: {e}")
                    if GracefulExit.stop:
                        break

            logger.info(f"Цикл завершен за {(datetime.now() - cycle_start).total_seconds():.2f} сек")

    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
    finally:
        logger.info("Приложение завершено")

if __name__ == "__main__":
    main()