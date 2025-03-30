import sqlite3
import aiosqlite
import logging
import numpy as np
import os
from config import LOGGING_CONFIG, COMPARISON_LOGGING_CONFIG
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Iterator, List, Tuple, Union, Dict, Any 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Уровень ERROR и выше
tf.autograph.set_verbosity(0)  # Отключаем логи AutoGraph

# Отключаем прогресс-бары и информационные сообщения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = все, 3 = ничего
tf.keras.utils.disable_interactive_logging()  # Отключает прогресс-бары
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка логгера для сравнений
comparison_logger = logging.getLogger('comparison_logger')
comparison_logger.setLevel(COMPARISON_LOGGING_CONFIG['level'])

# Создаем обработчик для файла
file_handler = logging.FileHandler(
    filename=COMPARISON_LOGGING_CONFIG['filename'],
    encoding=COMPARISON_LOGGING_CONFIG['encoding'],
    mode=COMPARISON_LOGGING_CONFIG['filemode']
)
file_handler.setFormatter(logging.Formatter(COMPARISON_LOGGING_CONFIG['format']))

# Добавляем обработчик к логгеру
comparison_logger.addHandler(file_handler)

# Отключаем распространение логов в корневой логгер, чтобы избежать дублирования
comparison_logger.propagate = False
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
            self._initialized = True

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
        """Безопасно получает время последнего обновления"""
        try:
            with self.db_session() as conn:
                cursor = conn.cursor()
                
                # 1. Проверяем существование колонки last_update
                cursor.execute("PRAGMA table_info(results)")
                columns = [col['name'] for col in cursor.fetchall()]
                if 'last_update' not in columns:
                    logger.debug("Колонка last_update не найдена в таблице results")
                    return None
                    
                # 2. Получаем последнее время обновления
                cursor.execute('''
                    SELECT last_update FROM results 
                    WHERE last_update IS NOT NULL
                    ORDER BY draw_number DESC 
                    LIMIT 1
                ''')
                result = cursor.fetchone()
                
                if not result:
                    logger.debug("Нет данных о времени обновления")
                    return None
                    
                # 3. Безопасное преобразование
                last_update = result['last_update']
                if isinstance(last_update, str):
                    return datetime.fromisoformat(last_update)
                elif isinstance(last_update, datetime):
                    return last_update
                else:
                    logger.warning(f"Неизвестный формат времени: {type(last_update)}")
                    return None
                    
        except Exception as e:
            logger.error(f"Ошибка получения времени обновления: {str(e)}")
            return None

    def update_last_update_time(self, draw_number: int) -> bool:
        """Обновляет время последнего обновления"""
        try:
            with self.db_session() as conn:
                cursor = conn.cursor()
                
                # Проверяем/добавляем колонку если нужно
                cursor.execute("PRAGMA table_info(results)")
                columns = [col['name'] for col in cursor.fetchall()]
                if 'last_update' not in columns:
                    cursor.execute("ALTER TABLE results ADD COLUMN last_update TIMESTAMP")
                
                # Обновляем время
                cursor.execute('''
                    UPDATE results 
                    SET last_update = ?
                    WHERE draw_number = ?
                ''', (datetime.now().isoformat(), draw_number))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Ошибка обновления времени: {str(e)}")
            return False 

    def update_schema(self):
            """Проверяет целостность схемы (для будущих миграций)"""
            try:
                cursor = self.connection.cursor()
                cursor.execute("PRAGMA foreign_key_check")
                return True
            except sqlite3.Error:
                return False
                                  
    def create_clean_data(self, force_recreate: bool = False) -> bool:
        """
        Инициализирует базу данных
        :param force_recreate: Принудительно пересоздает БД, если True
        :return: True, если БД была создана, False если уже существовала
        """
        if not force_recreate and self.database_exists():
            logging.info("База данных уже существует")
            return False
            
        try:
            # Закрываем соединение перед пересозданием
            if self._connection:
                self._connection.close()
                self._connection = None
            
            # Удаляем старый файл БД при необходимости
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                logging.info(f"Удален старый файл БД: {self.db_path}")
            
            # Создаем новое соединение
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
            
            cursor = self.connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Создаем таблицы
            cursor.execute("""
                CREATE TABLE results (
                    draw_number INTEGER PRIMARY KEY,
                    combination TEXT NOT NULL,
                    field INTEGER NOT NULL,
                    draw_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    date TIMESTAMP,
                    last_update TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    draw_number INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    predicted_combination TEXT NOT NULL,
                    predicted_field INTEGER NOT NULL,
                    actual_combination TEXT,
                    actual_field INTEGER,
                    is_correct BOOLEAN,
                    matched_numbers TEXT,
                    match_count INTEGER,
                    checked_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(draw_number) REFERENCES results(draw_number),
                    UNIQUE(draw_number, model_name)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE model_metadata (
                    model_name TEXT PRIMARY KEY,
                    last_trained TIMESTAMP,
                    version TEXT,
                    parameters TEXT,
                    performance_metrics TEXT
                )
            """)
            
            self.connection.commit()
            logging.info("База данных успешно инициализирована")
            return True
            
        except Exception as e:
            logging.error(f"Ошибка инициализации БД: {e}")
            if self._connection:
                self._connection.rollback()
            raise             
    
def safe_fromisoformat(date_str: str) -> Optional[datetime]:
    """Безопасное преобразование строки в datetime"""
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError) as e:
        logger.warning(f"Ошибка преобразования даты: {str(e)}")
        return None
    
def calculate_pages_to_parse() -> tuple[int, str]:
    """Возвращает количество страниц для парсинга и причину"""
    try:
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM results")
            result = cursor.fetchone()
            total_records = result[0] if result else 0
            
            if total_records == 0:
                return 10, "Полный парсинг (БД пуста)"
                
            last_update = db.get_last_update_time()
            
            if not last_update:
                return 1, "Обычный парсинг (нет данных о последнем обновлении)"
            
            delta = datetime.now() - last_update
            
            if delta > timedelta(days=2):
                return 5, "Расширенный парсинг (последнее обновление >2 дней назад)"
            elif delta < timedelta(hours=1, minutes=35):
                return 1, "Минимальный парсинг (недавнее обновление)"
            
            pages = max(1, int(delta.total_seconds() // (3600 + 2100)))  # 1 час 35 минут в секундах
            return pages, f"Обычный парсинг (дельта времени: {delta})"
            
    except Exception as e:
        logging.error(f"Ошибка расчета страниц для парсинга: {e}")
        return 1, f"Ошибка: {str(e)}" 

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
            # Подготовка данных для пакетной вставки
            prepared_data = []
            for data in data_batch:
                # Преобразуем дату в строку, если это datetime
                date = data[0].isoformat() if isinstance(data[0], datetime) else data[0]
                
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
                
                prepared_data.append((date, draw_number, combination_str, data[3]))

            # Пакетная вставка
            if prepared_data:
                cursor = db.connection.cursor()  # Получаем курсор из соединения
                cursor.executemany('''
                    INSERT OR IGNORE INTO results 
                    (date, draw_number, combination, field)
                    VALUES (?, ?, ?, ?)
                ''', prepared_data)
                db.connection.commit()  # Явно коммитим изменения
                logging.info(f"Успешно сохранено {len(prepared_data)} записей")
            else:
                logging.warning("Нет валидных данных для сохранения")
            
        except sqlite3.Error as e:
            logging.error(f"Ошибка базы данных: {e}")
            raise
        except Exception as e:
            logging.error(f"Неожиданная ошибка: {e}", exc_info=True)
            raise  

def save_prediction_to_db(
    model_name: str,
    predicted_combination: List[int],
    predicted_field: int,
    actual_combination: Optional[List[int]] = None,
    actual_field: Optional[int] = None
) -> None:
    """Сохраняет предсказание с реальными результатами"""
    try:
        with DatabaseManager() as db:
            # Получаем номер тиража через метод класса
            draw_number = db.get_max_draw_number()
            
            cursor = db.connection.cursor()
            
            # Создаем таблицу если не существует (ИСПРАВЛЕНО: добавлена закрывающая скобка)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    draw_number INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    predicted_combination TEXT NOT NULL,
                    predicted_field INTEGER NOT NULL,
                    actual_combination TEXT,
                    actual_field INTEGER,
                    is_correct INTEGER,
                    matched_numbers TEXT,
                    match_count INTEGER,
                    checked_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(draw_number) REFERENCES results(draw_number)
                )
            """)
            
            # Вычисляем совпадения если есть фактические результаты
            is_correct = None
            matched_numbers = None
            match_count = None
            checked_at = None
            
            if actual_combination is not None and actual_field is not None:
                match_count, matched_numbers = compare_combinations(predicted_combination, actual_combination)
                is_correct = (match_count > 0) and (predicted_field == actual_field)
                checked_at = datetime.now().isoformat()
            
            # Преобразуем комбинации в строки (ИСПРАВЛЕНО: единообразный формат)
            pred_comb_str = ' '.join(map(str, predicted_combination))
            actual_comb_str = ' '.join(map(str, actual_combination)) if actual_combination else None
            matched_numbers_str = ' '.join(map(str, matched_numbers)) if matched_numbers else None
            
            cursor.execute('''
                INSERT INTO predictions (
                    draw_number, model_name,
                    predicted_combination, predicted_field,
                    actual_combination, actual_field,
                    is_correct, matched_numbers,
                    match_count, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                draw_number, model_name,
                pred_comb_str, predicted_field,
                actual_comb_str, actual_field,
                1 if is_correct else 0 if is_correct is not None else None,
                matched_numbers_str,
                match_count,
                checked_at
            ))
            db.connection.commit()
            
    except sqlite3.Error as e:
        logging.error(f"Ошибка сохранения предсказания в БД: {e}")
        raise
    except Exception as e:
        logging.error(f"Неожиданная ошибка при сохранении предсказания: {e}")
        raise

def load_data_from_db() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загружает данные для обучения модели"""
    try:
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute('SELECT combination, field FROM results ORDER BY draw_number')
            results = cursor.fetchall()
            
            if not results:
                return np.array([]), np.array([]), np.array([])
            
            X = []
            y_field = []
            y_comb = []
            
            for row in results:
                try:
                    # Преобразуем комбинацию в список чисел
                    comb = [int(x.strip()) for x in row['combination'].split(',')]
                    if len(comb) != 8:
                        continue
                    
                    # One-hot кодировка комбинации (8 чисел × 20 вариантов)
                    comb_encoded = np.zeros((8, 20))
                    for i, num in enumerate(comb):
                        if 1 <= num <= 20:
                            comb_encoded[i, num - 1] = 1  # Числа 1-20 → индексы 0-19
                    
                    X.append(comb)
                    y_field.append(row['field'] - 1)  # Поле 1-4 → 0-3
                    y_comb.append(comb_encoded)
                    
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Ошибка обработки строки {row}: {str(e)}")
                    continue
            
            return np.array(X), np.array(y_field), np.array(y_comb)
            
    except Exception as e:
        logging.error(f"Ошибка загрузки данных: {str(e)}", exc_info=True)
        return np.array([]), np.array([]), np.array([])

def compare_predictions_with_real_data() -> None:
    """Сравнивает предсказания с реальными результатами"""
    last_game = get_last_game()
    if not last_game:
        comparison_logger.info("Нет данных о последней игре")
        return

    draw_number, real_comb, real_field = last_game
    with DatabaseManager() as db:
        cursor = db.connection.cursor()
        cursor.execute('''
            SELECT id, model_name, predicted_combination, predicted_field
            FROM predictions
            WHERE draw_number = ? AND actual_combination IS NULL
        ''', (draw_number,))
        predictions = cursor.fetchall()

    if not predictions:
        comparison_logger.info(f"Нет новых предсказаний для тиража {draw_number}")
        return

    comparison_logger.info(f"\nСравнение для тиража №{draw_number}:")
    comparison_logger.info(f"Реальная комбинация: {real_comb}, Поле: {real_field}")

    for pred in predictions:
        pred_comb = clean_combination(pred['predicted_combination'])
        matched_count, matched_numbers = compare_combinations(pred_comb, real_comb)
        is_field_correct = int(pred['predicted_field']) == real_field

        comparison_logger.info(f"\nМодель {pred['model_name']}:")
        comparison_logger.info(f"  Предсказано: {pred_comb}, Поле: {pred['predicted_field']}")
        comparison_logger.info(f"  Совпадений: {matched_count} ({matched_numbers})")
        comparison_logger.info(f"  Поле: {'Совпало' if is_field_correct else 'Не совпало'}")

        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute('''
                UPDATE predictions
                SET actual_combination = ?, actual_field = ?, is_correct = ?
                WHERE id = ?
            ''', (
                ', '.join(map(str, real_comb)),
                real_field,
                matched_count > 0 and is_field_correct,
                pred['id']
            ))
            db.connection.commit()

def clean_combination(combination: Union[str, List[int]]) -> List[int]:
    """Очищает и преобразует комбинацию в список чисел"""
    if isinstance(combination, str):
        return [int(x.strip()) for x in combination.split(',') if x.strip().isdigit()]
    elif isinstance(combination, list):
        return [int(x) for x in combination]
    return []

def get_last_game() -> Optional[Tuple[int, List[int], int]]:
    """Возвращает данные последней игры"""
    with DatabaseManager() as db:
        cursor = db.connection.cursor()
        cursor.execute('''
            SELECT draw_number, combination, field 
            FROM results 
            ORDER BY draw_number DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        if result:
            return (
                int(result['draw_number']),
                [int(x) for x in result['combination'].split(',')],
                int(result['field'])
            )
        return None
    
def save_prediction(
    draw_number: int,
    predicted_combination: List[int],
    predicted_field: int,
    model_name: str
) -> None:
    """Сохраняет предсказание в базу данных"""
    try:
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute('''
                INSERT INTO predictions (draw_number, predicted_combination, predicted_field, model_name)
                VALUES (?, ?, ?, ?)
            ''', (
                draw_number,
                ', '.join(map(str, predicted_combination)),
                predicted_field,
                model_name
            ))
            db.connection.commit()
            logger.info(f"Предсказание сохранено: {model_name} для тиража {draw_number}")
    except sqlite3.Error as e:
        logger.error(f"Ошибка сохранения предсказания: {e}")

def set_default_timestamps():
    """Устанавливает значения по умолчанию для временных меток"""
    with DatabaseManager() as db:
        try:
            cursor = db.connection.cursor()
            
            # Для новых записей в predictions
            cursor.execute('''
                UPDATE predictions 
                SET created_at = CURRENT_TIMESTAMP 
                WHERE created_at IS NULL
            ''')
            
            # Для новых записей в results
            cursor.execute('''
                UPDATE results 
                SET draw_date = CURRENT_TIMESTAMP 
                WHERE draw_date IS NULL
            ''')
            
            db.connection.commit()
        except sqlite3.Error as e:
            logging.warning(f"Не удалось установить временные метки: {e}")

def compare_combinations(
    predicted: List[int], 
    real: List[int]
) -> Tuple[int, List[int]]:
    """Сравнивает предсказанные и реальные комбинации"""
    predicted_set = set(predicted)
    real_set = set(real)
    matched_numbers = list(predicted_set & real_set)
    return len(matched_numbers), matched_numbers

# def create_clean_data(self, force_recreate: bool = False) -> bool:
#         """
#         Инициализирует базу данных
#         :param force_recreate: Принудительно пересоздает БД, если True
#         :return: True, если БД была создана, False если уже существовала
#         """
#         if not force_recreate and self.database_exists():
#             logging.info("База данных уже существует")
#             return False
            
#         try:
#             # Закрываем соединение перед пересозданием
#             if self._connection:
#                 self._connection.close()
#                 self._connection = None
            
#             # Удаляем старый файл БД при необходимости
#             if os.path.exists(self.db_path):
#                 os.remove(self.db_path)
#                 logging.info(f"Удален старый файл БД: {self.db_path}")
            
#             # Создаем новое соединение
#             self._connection = sqlite3.connect(self.db_path)
#             self._connection.row_factory = sqlite3.Row
            
#             cursor = self.connection.cursor()
#             cursor.execute("PRAGMA foreign_keys = ON")
            
#             # Создаем таблицы
#             cursor.execute("""
#                 CREATE TABLE results (
#                     draw_number INTEGER PRIMARY KEY,
#                     combination TEXT NOT NULL,
#                     field INTEGER NOT NULL,
#                     draw_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                     date TIMESTAMP,
#                     last_update TIMESTAMP
#                 )
#             """)
            
#             cursor.execute("""
#                 CREATE TABLE predictions (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     draw_number INTEGER NOT NULL,
#                     model_name TEXT NOT NULL,
#                     predicted_combination TEXT NOT NULL,
#                     predicted_field INTEGER NOT NULL,
#                     actual_combination TEXT,
#                     actual_field INTEGER,
#                     is_correct BOOLEAN,
#                     matched_numbers TEXT,
#                     match_count INTEGER,
#                     checked_at TIMESTAMP,
#                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                     FOREIGN KEY(draw_number) REFERENCES results(draw_number),
#                     UNIQUE(draw_number, model_name)
#                 )
#             """)
            
#             cursor.execute("""
#                 CREATE TABLE model_metadata (
#                     model_name TEXT PRIMARY KEY,
#                     last_trained TIMESTAMP,
#                     version TEXT,
#                     parameters TEXT,
#                     performance_metrics TEXT
#                 )
#             """)
            
#             self.connection.commit()
#             logging.info("База данных успешно инициализирована")
#             return True
            
#         except Exception as e:
#             logging.error(f"Ошибка инициализации БД: {e}")
#             if self._connection:
#                 self._connection.rollback()
#             raise                         