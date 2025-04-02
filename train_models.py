import os
import sqlite3
import numpy as np
import logging
from config import DATETIME_FORMAT, NEW_DATA_THRESHOLD, RETRAIN_HOURS, LOGGING_CONFIG, SEQUENCE_LENGTH, NUMBERS_RANGE, COMBINATION_LENGTH

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)
from typing import Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from LSTM_model import train_and_save_model
datetime.now().strftime(DATETIME_FORMAT)
from database import DatabaseManager
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Уровень ERROR и выше
tf.autograph.set_verbosity(0)  # Отключаем логи AutoGraph

# Отключаем прогресс-бары и информационные сообщения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = все, 3 = ничего
tf.keras.utils.disable_interactive_logging()  # Отключает прогресс-бары
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
class ModelTrainChecker:
    def __init__(self):
        self.model_dir = 'models'
        self.error_model_dir = os.path.join(self.model_dir, 'error_models')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.error_model_dir, exist_ok=True)
        
        self.min_train_samples = 100
        self.new_data_threshold = NEW_DATA_THRESHOLD
        self.retrain_interval = timedelta(hours=RETRAIN_HOURS)
        self.min_error_samples = 50
        self._init_db_tables()



    def _init_db_tables(self):
        """Инициализирует все таблицы БД с проверкой существования и обработкой ошибок.
        Если БД существует - проверяет наличие всех столбцов в таблицах.
        Если БД новая - создает все таблицы и индексы."""
        try:
            db = DatabaseManager()
            with db.db_session() as conn:
                cursor = conn.cursor()
                
                # Проверяем, существует ли уже БД (проверяем наличие хотя бы одной из наших таблиц)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
                db_exists = cursor.fetchone() is not None
                
                if not db_exists:
                    # Создаем новую БД
                    self._create_new_database(cursor)
                    logger.info("Создана новая БД со всеми таблицами и индексами")
                else:
                    # Проверяем существование всех столбцов в существующей БД
                    self._validate_existing_database(cursor)
                    logger.info("БД уже существует, проверка структуры выполнена")
                    
        except sqlite3.Error as e:
            logger.error(f"Ошибка SQL при инициализации БД: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            raise
        except Exception as e:
            logger.error(f"Критическая ошибка инициализации БД: {str(e)}", exc_info=True)
            if 'conn' in locals():
                conn.rollback()
            raise

    def _create_new_database(self, cursor):
        """Создает все таблицы и индексы в новой БД"""
        # Список таблиц и их определений
        tables = {
            'results': """
                CREATE TABLE results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    combination TEXT NOT NULL,
                    field INTEGER NOT NULL,
                    draw_number INTEGER NOT NULL,
                    draw_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_update TIMESTAMP,
                    UNIQUE(draw_number)
                )""", 
            
            'model_training_history': """
            CREATE TABLE model_training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                train_time TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                data_count INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                accuracy FLOAT,
                loss FLOAT,
                training_duration INTEGER,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                last_data_time TEXT,
                new_data_count INTEGER DEFAULT 0
            )""",
            
            'error_model_history': """
                CREATE TABLE error_model_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    train_time TEXT NOT NULL,
                    error_samples INTEGER NOT NULL,
                    field_accuracy FLOAT,
                    comb_accuracy FLOAT,
                    model_type TEXT,
                    base_model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""",
            
            'model_metadata': """
            CREATE TABLE model_metadata (
                model_name TEXT PRIMARY KEY,
                last_trained TEXT DEFAULT (datetime('now', 'localtime')),
                version TEXT,
                parameters TEXT,
                performance_metrics TEXT,
                is_active BOOLEAN DEFAULT 1,
                description TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                last_data_time TEXT,
                total_data_count INTEGER DEFAULT 0,
                new_data_count INTEGER DEFAULT 0
            )""",
            
            'predictions': """
                CREATE TABLE predictions (
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
                    result_code TEXT,
                    winning_tier TEXT,
                    checked_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(draw_number) REFERENCES results(draw_number) ON DELETE CASCADE
                )"""
        }

        # Создаем таблицы
        for table_name, create_sql in tables.items():
            cursor.execute(create_sql)
            logger.debug(f"Таблица {table_name} создана")

        # Создаем индексы
        indexes = [
            "CREATE INDEX idx_results_draw_number ON results(draw_number)",
            "CREATE INDEX idx_results_created ON results(created_at)",
            "CREATE INDEX idx_predictions_draw ON predictions(draw_number)"
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

        # Триггер для автоматического обновления метки времени
        cursor.execute("""
            CREATE TRIGGER update_model_metadata_timestamp
            AFTER UPDATE ON model_metadata
            FOR EACH ROW
            BEGIN
                UPDATE model_metadata 
                SET updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime') 
                WHERE model_name = NEW.model_name;
            END
        """)

    def _validate_existing_database(self, cursor):
        """Проверяет наличие всех столбцов в существующей БД"""
        # Словарь с ожидаемыми таблицами и их столбцами
        expected_tables = {
            'results': ['id', 'combination', 'field', 'draw_number', 'draw_date', 'created_at', 'last_update'],
            'model_training_history': ['id', 'train_time', 'data_count', 'model_version', 'accuracy', 'loss', 
                                    'training_duration', 'created_at', 'last_data_time', 'new_data_count'],
            'error_model_history': ['id', 'train_time', 'error_samples', 'field_accuracy', 'comb_accuracy', 
                                'model_type', 'base_model_version', 'created_at'],
            'model_metadata': ['model_name', 'last_trained', 'version', 'parameters', 'performance_metrics', 
                            'is_active', 'description', 'created_at', 'updated_at', 'last_data_time', 
                            'total_data_count', 'new_data_count'],
            'predictions': ['id', 'draw_number', 'model_name', 'predicted_combination', 'predicted_field', 
                        'actual_combination', 'actual_field', 'is_correct', 'matched_numbers', 'match_count', 
                        'result_code', 'winning_tier', 'checked_at', 'created_at']
        }

        for table_name, expected_columns in expected_tables.items():
            # Проверяем существование таблицы
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                logger.error(f"Таблица {table_name} отсутствует в существующей БД")
                continue

            # Получаем список существующих столбцов
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = [column[1] for column in cursor.fetchall()]

            # Проверяем наличие всех ожидаемых столбцов
            missing_columns = set(expected_columns) - set(existing_columns)
            if missing_columns:
                logger.warning(f"В таблице {table_name} отсутствуют столбцы: {', '.join(missing_columns)}")
            else:
                logger.debug(f"Таблица {table_name} проверена, все столбцы на месте")
   

    def _check_should_train(self, cursor) -> Tuple[bool, str]:
        """Проверяет необходимость обучения модели с учетом:
        1. Совместимости текущей модели с SEQUENCE_LENGTH
        2. Наличия новых данных
        3. Интервала переобучения
        4. Минимального количества данных"""
        
        # 1. Проверка совместимости модели
        model_path = os.path.join('models', 'best_lstm_model.keras')
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                current_seq_length = model.input_shape[1]
                if current_seq_length != SEQUENCE_LENGTH:
                    return True, (f"Требуется переобучение: модель обучена на {current_seq_length} "
                                f"последовательностях, текущий SEQUENCE_LENGTH = {SEQUENCE_LENGTH}")
            except Exception as e:
                logger.warning(f"Ошибка проверки модели: {str(e)}. Будет создана новая модель")
        # 1. Проверяем наличие предыдущего обучения
        cursor.execute("""
            SELECT train_time, data_count, last_data_time 
            FROM model_training_history 
            ORDER BY train_time DESC 
            LIMIT 1
        """)
        last_train = cursor.fetchone()
        
        # Если модель никогда не обучалась или файла модели нет - обучаем
        if not last_train or not os.path.exists('models/lstm_model.keras'):
            return True, "Первое обучение модели"
        
        last_time = datetime.strptime(last_train['train_time'], '%Y-%m-%d %H:%M:%S')
        current_time = datetime.now()  # Сохраняем как datetime объект
        time_since_last = current_time - last_time
        
        # 2. Получаем количество новых данных с момента последнего обучения
        new_data = 0
        last_data_time = last_train['last_data_time'] or last_train['train_time']
        
        try:
            cursor.execute("PRAGMA table_info(results)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Используем более точный метод подсчета
            if 'created_at' in columns:
                cursor.execute("""
                    SELECT COUNT(*) FROM results 
                    WHERE created_at > ?
                """, (last_data_time,))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM results 
                    WHERE draw_date > ?
                """, (last_data_time,))
                
            new_data = cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Ошибка подсчета новых данных: {str(e)}")
            return False, f"Ошибка подсчета данных: {str(e)}"
        
        # 3. Проверяем общее количество данных
        cursor.execute("SELECT COUNT(*) FROM results")
        total_data_now = cursor.fetchone()[0]
        
        # 4. Формируем информационное сообщение
        info_message = (
            f"Новых данных: {new_data}/{self.new_data_threshold} | "
            f"Всего данных: {total_data_now}/{self.min_train_samples} | "
            f"Последнее обучение: {last_time.strftime('%d.%m %H:%M')} | "
            f"Прошло времени: {time_since_last.days}д {time_since_last.seconds//3600}ч"
           )
        
        # 5. Проверяем минимальное количество данных
        if total_data_now < self.min_train_samples:
            return False, f"{info_message} | Недостаточно общих данных"
        logger.info(f"Новых результатов: {new_data}, Всего: {total_data_now}")
        # 6. Проверяем новые данные (главное условие)
        if new_data < self.new_data_threshold:
            return False, f"{info_message} | Недостаточно новых данных"
        
        # 7. Проверяем временной интервал (если RETRAIN_HOURS > 0)
        if self.retrain_interval.total_seconds() > 0:
            if time_since_last < self.retrain_interval:
                remaining = self.retrain_interval - time_since_last
                return False, (
                    f"{info_message} | Ожидание: {remaining.seconds//3600}ч "
                    f"{(remaining.seconds%3600)//60}м"
                )
        
        # 8. Если все условия выполнены
        return True, f"{info_message} | Запуск обучения"


    
    def get_training_model(self, cursor, incremental: bool) -> Optional[dict]:
        """Подготавливает данные для обучения"""
        try:
            # Убрали LIMIT для получения всех данных
            cursor.execute("""
                SELECT combination, field, draw_number 
                FROM results 
                ORDER BY draw_number DESC
            """)
            
            rows = cursor.fetchall()
            if len(rows) < SEQUENCE_LENGTH + 1:
                logger.error(f"Недостаточно данных. Нужно минимум {SEQUENCE_LENGTH+1} записей")
                return None
                
            # Подготовка данных
            combinations = []
            fields = []
            draw_numbers = []
            
            for row in rows:
                try:
                    comb = list(map(int, row['combination'].split(',')))
                    if len(comb) != COMBINATION_LENGTH:
                        logger.warning(f"Некорректная комбинация: {row['combination']}")
                        continue
                    combinations.append(comb)
                    fields.append(row['field'] - 1)  # Поля 0-3
                    draw_numbers.append(row['draw_number'])
                except ValueError as e:
                    logger.warning(f"Ошибка преобразования комбинации: {row['combination']}. Ошибка: {e}")
                    continue
            
            logger.info(f"Получено {len(combinations)} валидных комбинаций. Последний тираж: {draw_numbers[0]}")
            
            # Формирование последовательностей
            X = []
            y_field = []
            y_comb = []
            
            for i in range(len(combinations) - SEQUENCE_LENGTH):
                seq = combinations[i:i + SEQUENCE_LENGTH]
                target_comb = combinations[i + SEQUENCE_LENGTH]
                target_field = fields[i + SEQUENCE_LENGTH]
                
                # One-hot кодирование комбинации
                comb_encoded = np.zeros((COMBINATION_LENGTH, NUMBERS_RANGE))
                for pos, num in enumerate(target_comb):
                    if 1 <= num <= NUMBERS_RANGE:
                        comb_encoded[pos, num - 1] = 1
                    else:
                        logger.warning(f"Некорректный номер: {num} в комбинации {target_comb}")
                
                X.append(seq)
                y_field.append(target_field)
                y_comb.append(comb_encoded)
            
            logger.info(f"Сформировано {len(X)} обучающих последовательностей")
            
            return {
                'X_train': np.array(X, dtype=np.int32),
                'y_field': np.array(y_field, dtype=np.int32),
                'y_comb': np.array(y_comb, dtype=np.float32),
                'last_draw_number': draw_numbers[0] if draw_numbers else None
            }
            
        except Exception as e:
            logger.error(f"Ошибка подготовки данных: {str(e)}", exc_info=True)
            return None
    
    def update_training_info(self, accuracy: float = None, last_draw_number: int = None) -> bool:
        """Обновляет информацию о тренировке модели в таблице model_training_history"""
        logger.info("Запуск update_training_info...")
        
        try:
            with DatabaseManager() as db:
                cursor = db.connection.cursor()

                # Получаем общее количество данных
                cursor.execute("SELECT COUNT(*) FROM results")
                total_data = cursor.fetchone()[0]
                
                # Получаем последний тираж, если не передан
                if last_draw_number is None:
                    cursor.execute("SELECT MAX(draw_number) FROM results")
                    last_draw_number = cursor.fetchone()[0]

                # Определяем столбец для времени
                cursor.execute("PRAGMA table_info(results)")
                columns = [col[1] for col in cursor.fetchall()]
                time_column = 'created_at' if 'created_at' in columns else 'draw_date'

                # Получаем время последних данных
                cursor.execute(f"""
                    SELECT {time_column} FROM results 
                    WHERE draw_number = ? 
                    LIMIT 1
                """, (last_draw_number,))
                last_data_time = cursor.fetchone()[0] or datetime.now().strftime(DATETIME_FORMAT)

                # Считаем новые данные
                new_data_count = 0
                cursor.execute("SELECT MAX(train_time) FROM model_training_history")
                last_train_result = cursor.fetchone()
                
                if last_train_result and last_train_result[0]:
                    try:
                        cursor.execute(f"""
                            SELECT COUNT(*) FROM results 
                            WHERE {time_column} > ?
                        """, (last_train_result[0],))
                        new_data_count = cursor.fetchone()[0]
                    except sqlite3.Error:
                        new_data_count = 0

                # Вставляем в model_training_history
                current_time = datetime.now().strftime(DATETIME_FORMAT)
                cursor.execute("""
                    INSERT INTO model_training_history 
                    (train_time, data_count, new_data_count, model_version, 
                    accuracy, last_data_time, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    current_time, 
                    total_data,
                    new_data_count,
                    self._get_next_version(),
                    accuracy or 0.0, 
                    last_data_time, 
                    current_time
                ))
                
                db.connection.commit()
                return True

        except Exception as e:
            logger.error(f"Неожиданная ошибка: {str(e)}", exc_info=True)
            if 'db' in locals():
                db.connection.rollback()
            return False

    def _get_last_train_info(self) -> Optional[Dict[str, Any]]:
        """Получает информацию о последнем обучении"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute("""
                SELECT * FROM model_training_history 
                ORDER BY train_time DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None

    def _get_new_data_count(self, since_time: datetime) -> int:
        """Считает количество новых данных с указанного времени"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            try:
                # Пробуем использовать created_at, если доступен
                cursor.execute("PRAGMA table_info(results)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'created_at' in columns:
                    cursor.execute("""
                        SELECT COUNT(*) FROM results 
                        WHERE created_at > ?
                    """, (since_time.isoformat(),))
                else:
                    # Fallback на draw_date, если created_at отсутствует
                    cursor.execute("""
                        SELECT COUNT(*) FROM results 
                        WHERE draw_date > ?
                    """, (since_time.isoformat(),))
                    
                return cursor.fetchone()[0]
            except Exception as e:
                logger.error(f"Ошибка подсчета новых данных: {str(e)}")
                return 0

    def _get_next_version(self) -> str:
        """Генерирует следующую версию модели"""
        last_version = self._get_last_version()
        if not last_version:
            return "1.0"
        major, minor = last_version.split('.')
        return f"{major}.{int(minor)+1}"

    def _get_last_version(self) -> Optional[str]:
        """Получает последнюю версию модели"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute("""
                SELECT version FROM model_metadata 
                WHERE model_name = 'lstm_model' 
                ORDER BY last_trained DESC LIMIT 1
            """)
            result = cursor.fetchone()
            return result[0] if result else None
        
if __name__ == "__main__":
    checker = ModelTrainChecker()
    with DatabaseManager() as db:
        cursor = db.connection.cursor()
        should_train, message = checker._check_should_train(cursor)
        print(f"Should train: {should_train}, Message: {message}")
        
        training_data = checker.get_training_model(cursor, incremental=False)
        if not training_data:
            logger.error("Не удалось получить данные для обучения")
            exit(1)
        
        # Вся работа с БД теперь внутри train_and_save_model
        model_result = train_and_save_model(training_data, db.connection)
        
        if not model_result.get('success', False):
            logger.error(f"Ошибка обучения: {model_result.get('message', 'Unknown error')}")
            exit(1)
        
        logger.info("Обучение завершено успешно")
                    

