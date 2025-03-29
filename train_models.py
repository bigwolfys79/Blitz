import os
import sqlite3
import numpy as np
import json
import logging
from config import LOGGING_CONFIG, SEQUENCE_LENGTH, NUMBERS_RANGE, COMBINATION_LENGTH

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from database import DatabaseManager
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
        self.new_data_threshold = 1
        self.retrain_interval = timedelta(days=1)
        self.min_error_samples = 50
        self._init_db_tables()



    def _init_db_tables(self):
        """Инициализирует таблицы БД для хранения истории обучения"""
        try:
            db = DatabaseManager()
            with db.db_session() as conn:
                cursor = conn.cursor()
                
                # 1. Таблица истории обучения основной модели
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_training_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        train_time TEXT NOT NULL,
                        data_count INTEGER NOT NULL,
                        model_version TEXT NOT NULL,
                        accuracy FLOAT,
                        loss FLOAT,
                        training_duration INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 2. Таблица истории обучения корректирующих моделей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS error_model_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        train_time TEXT NOT NULL,
                        error_samples INTEGER NOT NULL,
                        field_accuracy FLOAT,
                        comb_accuracy FLOAT,
                        model_type TEXT,
                        base_model_version TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 3. Таблица метаданных модели
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_metadata (
                        model_name TEXT PRIMARY KEY,
                        last_trained TIMESTAMP,
                        version TEXT,
                        parameters TEXT,
                        performance_metrics TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 4. Триггер для обновления метки времени
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_model_metadata_timestamp
                    AFTER UPDATE ON model_metadata
                    FOR EACH ROW
                    BEGIN
                        UPDATE model_metadata SET updated_at = CURRENT_TIMESTAMP 
                        WHERE model_name = NEW.model_name;
                    END
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Ошибка инициализации таблиц БД: {str(e)}", exc_info=True)
            raise

    def _check_table_structure(self, cursor):
        """Проверяет структуру таблицы model_training_history"""
        cursor.execute("PRAGMA table_info(model_training_history)")
        columns = {column[1]: column[2] for column in cursor.fetchall()}
        
        if 'accuracy' not in columns:
            cursor.execute("ALTER TABLE model_training_history ADD COLUMN accuracy FLOAT")
            logger.info("Добавлена колонка accuracy в model_training_history")    

    def _ensure_columns_exist(self, cursor, table_name, columns):
        """Вспомогательный метод для проверки существования столбцов"""
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [column[1] for column in cursor.fetchall()]
        
        for column in columns:
            if column not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} TEXT")
                    logger.info(f"Добавлен столбец {column} в таблицу {table_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        logger.warning(f"Не удалось добавить столбец {column}: {str(e)}")    

    def _ensure_columns_exist(self, cursor, table_name, columns):
        """Проверяет существование столбцов и добавляет отсутствующие"""
        for column in columns:
            try:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} TEXT")
                logger.info(f"Добавлен столбец {column} в таблицу {table_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    logger.warning(f"Не удалось добавить столбец {column}: {str(e)}")

    def _check_should_train(self, cursor) -> Tuple[bool, str]:
        """Проверяет необходимость обучения используя существующий курсор"""
        cursor.execute("SELECT * FROM model_training_history ORDER BY train_time DESC LIMIT 1")
        last_train = cursor.fetchone()
        
        if not last_train or not os.path.exists('models/lstm_model.keras'):
            return True, "Первое обучение"
        
        last_time = datetime.fromisoformat(last_train['train_time'])
        if (datetime.now() - last_time) > self.retrain_interval:
            return True, "Плановое переобучение"
        
        # Используем сохранённое количество данных из журнала
        cursor.execute("SELECT COUNT(*) FROM results")
        total_data_now = cursor.fetchone()[0]

        # Последнее количество данных, сохранённое при обучении
        last_data_count = last_train['data_count']

        new_data = total_data_now - last_data_count

        return (new_data >= self.new_data_threshold,
                f"Новых данных: {new_data} (порог: {self.new_data_threshold})")


    def get_training_model(self, cursor, incremental: bool) -> Optional[dict]:
        """Подготавливает данные для обучения"""
        try:
            limit = 1000  # Достаточно для 30 последовательностей
            cursor.execute("""
                SELECT combination, field FROM results 
                ORDER BY draw_number DESC LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            if len(rows) < SEQUENCE_LENGTH + 1:  # Нужно хотя бы SEQUENCE_LENGTH+1 комбинаций
                logger.error(f"Недостаточно данных. Нужно минимум {SEQUENCE_LENGTH+1} записей")
                return None
                
            # Подготовка данных
            combinations = []
            fields = []
            
            for row in rows:
                try:
                    comb = list(map(int, row['combination'].split(',')))
                    if len(comb) != COMBINATION_LENGTH:
                        continue
                    combinations.append(comb)
                    fields.append(row['field'] - 1)  # Поля 0-3
                except ValueError:
                    continue
            
            # Формирование последовательностей
            X = []
            y_field = []
            y_comb = []
            
            for i in range(len(combinations) - SEQUENCE_LENGTH):
                # Входная последовательность
                seq = combinations[i:i + SEQUENCE_LENGTH]
                
                # Целевые значения
                target_comb = combinations[i + SEQUENCE_LENGTH]
                target_field = fields[i + SEQUENCE_LENGTH]
                
                # One-hot кодирование комбинации
                comb_encoded = np.zeros((COMBINATION_LENGTH, NUMBERS_RANGE))
                for pos, num in enumerate(target_comb):
                    comb_encoded[pos, num - 1] = 1
                
                X.append(seq)
                y_field.append(target_field)
                y_comb.append(comb_encoded)
            
            return {
                'X_train': np.array(X, dtype=np.int32),
                'y_field': np.array(y_field, dtype=np.int32),
                'y_comb': np.array(y_comb, dtype=np.float32)
            }
            
        except Exception as e:
            logger.error(f"Ошибка подготовки данных: {str(e)}")
            return None

    def update_training_info(self, data_count: int, accuracy: float = None) -> bool:
        """Обновляет информацию о тренировке в БД с проверкой структуры"""
        try:
            version = self._get_next_version()
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                
                # 1. Проверяем и обновляем структуру таблицы
                self._check_table_structure(cursor)
                
                # 2. Вставляем данные
                cursor.execute("""
                    INSERT INTO model_training_history 
                    (train_time, data_count, model_version, accuracy, created_at)
                    VALUES (datetime('now'), ?, ?, ?, datetime('now'))
                """, (data_count, version, accuracy))
                
                # 3. Обновляем метаданные
                cursor.execute("""
                    INSERT OR REPLACE INTO model_metadata 
                    (model_name, last_trained, version, performance_metrics, updated_at)
                    VALUES (?, datetime('now'), ?, ?, datetime('now'))
                """, ("lstm_model", version, 
                    json.dumps({'accuracy': accuracy}) if accuracy else None))
                
                db.connection.commit()
                return True
                
        except Exception as e:
            logger.error(f"Ошибка обновления информации: {str(e)}", exc_info=True)
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

    def _get_new_data_count(self, since_time: str) -> int:
        """Считает количество новых данных с указанного времени"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM results 
                WHERE draw_date > ?
            """, (since_time,))
            return cursor.fetchone()[0]

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