import time
import signal

import numpy as np 
import aiosqlite
import os
import logging
from config import LOGGING_CONFIG

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)
from predict import LotteryPredictor
from datetime import datetime, timedelta
from contextlib import contextmanager
from parsing import parse_data
from LSTM_model import train_and_save_model
from train_models import ModelTrainChecker
from database import DatabaseManager, calculate_pages_to_parse
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Менеджер контекста для работы с БД
@contextmanager
def db_session():
    db = DatabaseManager()
    try:
        yield db
    finally:
        db.close()

class GracefulExit:
    stop = False

    @staticmethod
    def signal_handler(signal_received, frame):
        logger.info("Получен сигнал завершения (Ctrl+C). Завершаем выполнение...")
        GracefulExit.stop = True

signal.signal(signal.SIGINT, GracefulExit.signal_handler)
signal.signal(signal.SIGTERM, GracefulExit.signal_handler)

def wait_until_next_interval(interval_minutes: int = 5, seconds_offset: int = 30) -> datetime:
    """
    Ждет до следующего интервала и возвращает время следующего запуска.
    
    Args:
        interval_minutes: Интервал в минутах
        seconds_offset: Смещение в секундах внутри минуты
        
    Returns:
        datetime: Время следующего запуска
    """
    now = datetime.now()
    current_minute = now.minute
    
    # Вычисляем следующую минуту запуска
    next_minute = ((current_minute // interval_minutes) + 1) * interval_minutes
    delta_hours = 0
    
    if next_minute >= 60:
        next_minute = 0
        delta_hours = 1
    
    # Формируем время следующего запуска
    next_time = now.replace(
        minute=next_minute, 
        second=seconds_offset, 
        microsecond=0
    )
    
    if delta_hours > 0:
        next_time += timedelta(hours=delta_hours)
    
    # Корректировка если расчетное время уже прошло
    if next_time < now:
        next_time += timedelta(minutes=interval_minutes)
    
    wait_seconds = (next_time - now).total_seconds()
    logger.info(f"Следующий запуск в {next_time.strftime('%H:%M:%S')}")
    
    while wait_seconds > 0:
        if GracefulExit.stop:
            logger.info("Ожидание прервано. Завершаем выполнение...")
            return None
        time.sleep(0.1)  # Проверка каждые 100 мс
        wait_seconds -= 0.1

    
    return next_time

class ModelTrainer:
    def __init__(self):
        self.checker = ModelTrainChecker()  # Инициализация проверяющего
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)

    def run_training_cycle(self) -> bool:
        """Основной цикл обучения"""
        try:
            with DatabaseManager() as db:
                # Создаем курсор (без контекстного менеджера)
                cursor = db.connection.cursor()
                
                try:
                    # 1. Проверка необходимости обучения
                    should_train, reason = self.checker._check_should_train(cursor)
                    if not should_train:
                        logger.info(f"Обучение не требуется: {reason}")
                        return False
                        
                    logger.info(f"Начинаем обучение: {reason}")
                    
                    # 2. Получение данных для обучения (передаем курсор)
                    incremental = "инкрементальное" in reason.lower()
                    training_data = self.checker.get_training_model(cursor, incremental)
                    
                    if not training_data:
                        logger.error("Не удалось получить данные для обучения")
                        return False
                                            
                    # 3. Обучение модели (используем функцию train_and_save_model)
                    model_result = train_and_save_model(training_data)
                        
                    if not model_result.get('success', False):
                        logger.error(f"Ошибка обучения: {model_result.get('message', 'Unknown error')}")
                        return False
                        
                    # 6. Сохранение результатов
                    self.checker.update_training_info(
                        data_count=model_result['data_count'],
                        accuracy=model_result.get('accuracy')
                    )
                    db.connection.commit()
                    
                    logger.info("Обучение успешно завершено")
                    return True
                    
                finally:
                    if 'cursor' in locals():
                        cursor.close()
                        logger.debug("Курсор закрыт")
                    
        except Exception as ex:
            logger.error(f"Критическая ошибка соединения: {str(ex)}")
            return False

def main():
    """Основной цикл приложения с исправлениями"""
    trainer = ModelTrainChecker()
    predictor = LotteryPredictor()
    cicletrainer=ModelTrainer()
    
    try:
        while not GracefulExit.stop:
            cycle_start = datetime.now()
            logger.info(f"Начало цикла обработки в {cycle_start}")

            
            # 1. Ожидание следующего цикла (ВОТ ОН, ВОЗВРАЩЕННЫЙ БЛОК)
            next_run = wait_until_next_interval(
                interval_minutes=5, 
                seconds_offset=30
            )
            logger.debug(f"Следующий запуск в {next_run}")
            if GracefulExit.stop or next_run is None:
                break  # Завершаем выполнение, если поступил сигнал завершения


            # Используем единое соединение для всего цикла
            with DatabaseManager() as db:
                try:
                    # 2. Парсинг данных
                    logger.info("Парсинг новых данных...")
                    try:
                        parse_result = calculate_pages_to_parse()
                        pages = parse_result[0] if parse_result else 1
                        parse_data(pages_to_parse=pages)
                    except Exception as e:
                        logger.error(f"Ошибка парсинга: {e}")

                        continue

                    # 3. Обучение модели
                    logger.info("Обучение модели...")
                    if GracefulExit.stop:
                        break  # Проверка на сигнал завершения

                    try:
                        cicletrainer.run_training_cycle()
                    except Exception as e:
                        logger.error(f"Ошибка обучения: {e}")
                        continue

                    # 4. Генерация и проверка предсказаний
                    logger.info("Генерация предсказаний...")
                    if GracefulExit.stop:
                        break  # Проверка на сигнал завершения

                    try:
                        if not predictor.predict_and_save():
                            logger.error("Не удалось сохранить предсказание")
                            continue
                            
                        logger.info("Проверка предсказаний...")
                        check_result = predictor.check_last_prediction()
                        # logger.info(f"Результат проверки: {check_result}")
                        
                    except Exception as e:
                        logger.error(f"Ошибка предсказаний: {e}")
                        continue

                except Exception as e:
                    logger.critical(f"Критическая ошибка: {e}")

                
            # Контроль времени выполнения
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Цикл завершен за {cycle_time:.2f} сек")
            
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем.")

    except Exception as e:
        logger.critical(f"Фатальная ошибка: {e}")
    finally:
        logger.info("Освобождение ресурсов...")

if __name__ == "__main__":
    main()