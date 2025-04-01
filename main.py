import os
import threading
import sys
import time
import signal
from config import REPORT_FREQUENCY, REPORT_PERIOD_DAYS
import logging
from datetime import datetime, timedelta
from typing import Optional
from predict import LotteryPredictor
from parsing import parse_data
from LSTM_model import train_and_save_model
from train_models import ModelTrainChecker
from database import DatabaseManager

class GracefulExit:
    stop = False
    _lock = threading.Lock()

    @classmethod
    def should_stop(cls):
        with cls._lock:
            return cls.stop

    @staticmethod
    def signal_handler(signal_received, frame):
        with GracefulExit._lock:
            if not GracefulExit.stop:
                logger.info("Получен сигнал завершения (Ctrl+C). Завершаем выполнение...")
                GracefulExit.stop = True
            else:
                logger.warning("Принудительное завершение! (повторное нажатие Ctrl+C)")
                sys.exit(1)

# Настройка логгера с записью только в файл
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
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

# Регистрируем обработчики сигналов
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
                    if GracefulExit.should_stop():
                        return False

                    should_train, reason = self.checker._check_should_train(cursor)
                    if not should_train:
                        logger.info(f"Обучение не требуется: {reason}")
                        return False

                    if GracefulExit.should_stop():
                        return False

                    training_data = self.checker.get_training_model(cursor, incremental=False)
                    if not training_data:
                        logger.error("Не удалось получить данные для обучения")
                        return False

                    if GracefulExit.should_stop():
                        return False

                    model_result = train_and_save_model(training_data, db.connection)
                    if not model_result.get('success', False):
                        logger.error(f"Ошибка обучения: {model_result.get('message', 'Unknown error')}")
                        return False

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
    
    while wait_seconds > 0 and not GracefulExit.should_stop():
        time.sleep(min(1, wait_seconds))
        wait_seconds = (next_time - datetime.now()).total_seconds()
    
    return next_time if not GracefulExit.should_stop() else None

def main():
    """Точка входа в приложение."""
    logger.info("=" * 50)
    logger.info("Инициализация приложения")

    trainer = ModelTrainer()
    if not trainer.ensure_model_exists():
        logger.critical("Не удалось инициализировать модель! Завершение работы.")
        return

    predictor = LotteryPredictor()
    logger.info("Приложение готово к работе")
    logger.info("=" * 50)

    cycle_counter = 0

    try:
        while not GracefulExit.should_stop():
            cycle_counter += 1
            cycle_start = datetime.now()
            logger.info(f"\n{'-'*83}\nЦикл начат: {cycle_start.strftime('%H:%M:%S')}")

            next_run = wait_until_next_interval()
            if GracefulExit.should_stop() or next_run is None:
                break

            try:
                with DatabaseManager() as db:
                    cursor = db.connection.cursor()
                    try:
                        if GracefulExit.should_stop():
                            break
                        parse_data()

                        if GracefulExit.should_stop():
                            break
                        trainer.run_training_cycle()

                        if GracefulExit.should_stop():
                            break
                        if predictor.predict_and_save():
                            predictor.check_last_prediction()
                    finally:
                        cursor.close()
            except Exception as e:
                logger.error(f"Ошибка в цикле: {e}")
                if GracefulExit.should_stop():
                    break

            if cycle_counter % REPORT_FREQUENCY == 0:
                try:
                    report = predictor.generate_performance_report(REPORT_PERIOD_DAYS)
                except Exception as e:
                    logger.error(f"Ошибка генерации отчета: {e}")

            logger.info(f"Цикл завершен за {(datetime.now() - cycle_start).total_seconds():.2f} сек")

    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
    finally:
        try:
            report = predictor.generate_performance_report(30)
            print(f"\n{' ФИНАЛЬНЫЙ ОТЧЕТ ':=^50}\n{report}\n{'='*50}")
            logger.info(f"Финальный отчет:\n{report}")
        except Exception as e:
            logger.error(f"Ошибка генерации финального отчета: {e}")
        
        logger.info("Приложение завершено")

if __name__ == "__main__":
    main()