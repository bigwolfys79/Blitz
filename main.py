import os
import threading
import sys
import time
import signal
from config import REPORT_FREQUENCY, REPORT_PERIOD_DAYS, SEQUENCE_LENGTH
import logging
from datetime import datetime, timedelta
from typing import Optional
from predict import LotteryPredictor
from parsing import parse_data
from LSTM_model import train_and_save_model
from train_models import ModelTrainChecker
from database import DatabaseManager
import tensorflow as tf

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

setup_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join('models', 'best_lstm_model.keras')

signal.signal(signal.SIGINT, GracefulExit.signal_handler)
signal.signal(signal.SIGTERM, GracefulExit.signal_handler)

class ModelTrainer:
    def __init__(self):
        self.checker = ModelTrainChecker()
        self.last_seq_length = SEQUENCE_LENGTH
        os.makedirs('models', exist_ok=True)

    def check_model_compatibility(self) -> bool:
        """Проверяет совместимость модели с текущим SEQUENCE_LENGTH"""
        if not os.path.exists(MODEL_PATH):
            return False
            
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            model_seq_length = model.input_shape[1]
            if model_seq_length != SEQUENCE_LENGTH:
                logger.warning(
                    f"Несоответствие SEQUENCE_LENGTH! "
                    f"Модель: {model_seq_length}, конфиг: {SEQUENCE_LENGTH}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Ошибка проверки модели: {e}")
            return False

    def ensure_model_exists(self) -> bool:
        """Проверяет и создает модель при необходимости."""
        if os.path.exists(MODEL_PATH) and self.check_model_compatibility():
            return True

        if os.path.exists(MODEL_PATH):
            logger.warning("Удаление несовместимой модели...")
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
                    if not should_train and self.check_model_compatibility():
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

                    logger.info(f"Начало обучения модели (SEQUENCE_LENGTH={SEQUENCE_LENGTH})")
                    model_result = train_and_save_model(training_data, db.connection)
                    if not model_result.get('success', False):
                        logger.error(f"Ошибка обучения: {model_result.get('message', 'Unknown error')}")
                        return False

                    self.last_seq_length = SEQUENCE_LENGTH
                    db.connection.commit()
                    logger.info("Модель успешно обучена")
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

            # Проверка изменения SEQUENCE_LENGTH
            if SEQUENCE_LENGTH != trainer.last_seq_length:
                logger.warning(
                    f"Обнаружено изменение SEQUENCE_LENGTH! "
                    f"Было: {trainer.last_seq_length}, стало: {SEQUENCE_LENGTH}. "
                    f"Требуется переобучение модели."
                )
                if not trainer.run_training_cycle():
                    logger.error("Не удалось переобучить модель!")
                    break

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
                    # После запуска предсказаний в течение нескольких дней
                    # stats = predictor.get_strategy_performance(60)

                    # print("{:<10} {:<10} {:<10} {:<10}".format(
                    #     "Стратегия", "Прогнозов", "Ср. совпад.", "Выигрыши %"
                    # ))
                    # for strategy, data in stats.items():
                    #     print("{:<10} {:<10} {:<10.2f} {:<10.2f}".format(
                    #         strategy, 
                    #         data['total'], 
                    #         data['avg_matches'], 
                    #         data['win_rate']
                    #     ))
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