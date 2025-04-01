import sqlite3
import logging
import numpy as np
import os
from config import LOGGING_CONFIG, DATETIME_FORMAT

from contextlib import contextmanager
from datetime import datetime, timedelta
datetime.now().strftime(DATETIME_FORMAT)
from typing import Optional, Iterator, List, Tuple, Union, Dict, Any 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Уровень ERROR и выше
tf.autograph.set_verbosity(0)  # Отключаем логи AutoGraph

# Отключаем прогресс-бары и информационные сообщения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = все, 3 = ничего
tf.keras.utils.disable_interactive_logging()  # Отключает прогресс-бары
logging.basicConfig(**LOGGING_CONFIG)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import contextmanager
# Менеджер контекста для работы с БД
@contextmanager
def db_session():
    db = DatabaseManager()
    try:
        yield db
    finally:
        db.close()
class DatabaseManager:
    _instance = None


    
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.db_path = 'results.db'
            self._connection = None
            self.timeout = 30  # seconds
            self._connection = sqlite3.connect(
                self.db_path, 
                timeout=self.timeout,
                check_same_thread=False
            )

            self._initialized = True
            print(f"!!! Подключение к БД: {self.db_path}")  # Куда именно подключаемся
        if not hasattr(self, '_already_initialized'):
            print("!!! ИНИЦИАЛИЗАЦИЯ БД !!!")
            self._already_initialized = True

    def _check_connection(self):
        """Проверяет состояние соединения"""
        if self._connection is None:
            raise RuntimeError("Соединение с БД не инициализировано")
        if self._connection.closed:  # <-- Новая проверка
            raise RuntimeError("Соединение с БД уже закрыто")
        return True        
    
    def __enter__(self):
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        logger.debug(f"Открыто соединение (ID: {id(self._connection)})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connection:
            logger.debug(f"Закрытие соединения (ID: {id(self._connection)})")
            self._connection.close()
            self._connection = None
            self.close()
    
    @property
    def connection(self):
        if self._connection is None:
            try:
                self._connection = sqlite3.connect(self.db_path)
                self._connection.row_factory = sqlite3.Row
                logger.info(f"Установлено соединение с БД: {self.db_path}")
            except sqlite3.Error as e:
                logger.error(f"Ошибка подключения к БД: {str(e)}")
                raise
        return self._connection
    
    def close(self):
        """Закрывает соединение с БД"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Соединение с БД закрыто")
    
    def execute(self, query: str, params=()) -> Optional[sqlite3.Cursor]:
        """Выполняет SQL-запрос"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            return cursor
        except sqlite3.Error as e:
            logger.error(f"Ошибка выполнения запроса: {str(e)}")
            self.connection.rollback()
            return None
    
    @contextmanager
    def db_session(self) -> Iterator[sqlite3.Connection]:
        """Альтернативный контекстный менеджер для сессий"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Ошибка в сессии БД: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_max_draw_number(self) -> int:
        """Возвращает максимальный номер тиража"""
        try:
            with self.db_session() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT MAX(draw_number) FROM results')
                result = cursor.fetchone()
                return int(result[0]) if result and result[0] else 0
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения номера тиража: {str(e)}")
            return 0

    def get_last_update_time(self) -> Optional[datetime]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT last_update FROM results ORDER BY draw_number DESC LIMIT 1")
        result = cursor.fetchone()
        if result and result['last_update']:
            # Преобразуем строку в datetime
            return datetime.strptime(result['last_update'], DATETIME_FORMAT)
        return None

    def update_last_update_time(self, draw_number: int = None) -> bool:
        """Обновляет время последнего обновления (локальное время без зоны)"""
        try:
            # Форматирование времени вручную
            now_local = datetime.now().strftime(DATETIME_FORMAT)
            logger.debug(f"Попытка обновления last_update: {now_local}, draw_number={draw_number}")

            with self.db_session() as conn:
                cursor = conn.cursor()
                
                if draw_number is not None:
                    # Проверка существования тиража
                    cursor.execute("SELECT 1 FROM results WHERE draw_number = ?", (draw_number,))
                    if not cursor.fetchone():
                        logger.error(f"Тираж {draw_number} не найден.")
                        return False
                    
                    cursor.execute('''
                        UPDATE results 
                        SET last_update = ?
                        WHERE draw_number = ?
                    ''', (now_local, draw_number))
                else:
                    # Проверка наличия данных в таблице
                    cursor.execute("SELECT MAX(draw_number) FROM results")
                    max_draw = cursor.fetchone()[0]
                    if not max_draw:
                        logger.error("Таблица results пуста. Обновление невозможно.")
                        return False
                    
                    cursor.execute('''
                        UPDATE results 
                        SET last_update = ?
                        WHERE draw_number = ?
                    ''', (now_local, max_draw))
                
                conn.commit()
                logger.debug("Обновление last_update успешно.")
                return True
        except Exception as e:
            logger.error(f"Ошибка обновления времени: {str(e)}", exc_info=True)
            return False

def safe_fromisoformat(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, DATETIME_FORMAT)
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Ошибка преобразования даты: {str(e)}")
        return None

def save_to_database_batch(data_batch: List[Tuple[Union[str, datetime], Union[str, int], Union[str, List[int]], int]]) -> None:
    """Сохраняет пачку данных в базу данных
    
    Args:
        data_batch: Список кортежей в формате (date, draw_number, combination, field)
            date: str или datetime - дата тиража
            draw_number: str или int - номер тиража
            combination: str (формат "1,2,3,4,5") или List[int] - выигрышная комбинация
            field: int - выигрышное поле
    """
    with DatabaseManager() as db:
        try:
            prepared_data = []
            now_local = datetime.now().strftime(DATETIME_FORMAT)  # Локальное время в правильном формате
            
            for data in data_batch:
                # Преобразование даты в строку
                if isinstance(data[0], datetime):
                    date_str = data[0].strftime(DATETIME_FORMAT)
                else:
                    # Если дата пришла как строка - проверяем формат
                    try:
                        datetime.strptime(data[0], DATETIME_FORMAT)
                        date_str = data[0]
                    except (ValueError, TypeError):
                        logging.error(f"Некорректный формат даты: {data[0]}")
                        continue
                
                # Преобразуем номер тиража в строку
                draw_number = str(data[1])
                
                # Обрабатываем комбинацию
                if isinstance(data[2], str):
                    try:
                        # Проверяем и преобразуем строку комбинации
                        combination_list = [int(x.strip()) for x in data[2].split(',')]
                        combination_str = ', '.join(map(str, combination_list))
                    except ValueError:
                        logging.error(f"Некорректный формат строки комбинации: {data[2]}")
                        continue
                elif isinstance(data[2], list):
                    combination_str = ', '.join(map(str, data[2]))
                else:
                    logging.error(f"Неизвестный формат комбинации: {type(data[2])}")
                    continue
                
                # Проверяем поле
                if not isinstance(data[3], int):
                    logging.error(f"Некорректный формат поля: {type(data[3])}")
                    continue
                
                # Добавляем с текущим локальным временем для created_at
                prepared_data.append((
                    date_str,          # draw_date из исходных данных
                    draw_number,       # номер тиража
                    combination_str,   # комбинация
                    data[3],           # поле
                    now_local          # created_at (локальное время)
                ))

            # Пакетная вставка
            if prepared_data:
                cursor = db.connection.cursor()
                cursor.executemany('''
                    INSERT OR IGNORE INTO results 
                    (draw_date, draw_number, combination, field, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', prepared_data)
                db.connection.commit()
                logging.info(f"Успешно сохранено {len(prepared_data)} записей")
            else:
                logging.warning("Нет валидных данных для сохранения")
            
        except sqlite3.Error as e:
            logging.error(f"Ошибка базы данных: {e}")
            raise
        except Exception as e:
            logging.error(f"Неожиданная ошибка: {e}", exc_info=True)
            raise  

def compare_combinations(
    predicted: List[int], 
    real: List[int]
) -> Tuple[int, List[int]]:
    """Сравнивает предсказанные и реальные комбинации"""
    predicted_set = set(predicted)
    real_set = set(real)
    matched_numbers = list(predicted_set & real_set)
    return len(matched_numbers), matched_numbers
                    