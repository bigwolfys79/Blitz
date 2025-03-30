import os
import sqlite3
import numpy as np
import json
import logging
from config import NEW_DATA_THRESHOLD, RETRAIN_HOURS, LOGGING_CONFIG, SEQUENCE_LENGTH, NUMBERS_RANGE, COMBINATION_LENGTH

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta, timezone
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
        self.new_data_threshold = NEW_DATA_THRESHOLD
        self.retrain_interval = timedelta(hours=RETRAIN_HOURS)
        self.min_error_samples = 50
        self._init_db_tables()



    def _init_db_tables(self):
        """Инициализирует все таблицы БД с проверкой существования и обработкой ошибок"""
        try:
            db = DatabaseManager()
            with db.db_session() as conn:
                cursor = conn.cursor()

                # Список таблиц и их определений
                tables = {
                    'results': """
                        CREATE TABLE IF NOT EXISTS results (
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
                        CREATE TABLE IF NOT EXISTS model_training_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            train_time TEXT NOT NULL,
                            data_count INTEGER NOT NULL,
                            model_version TEXT NOT NULL,
                            accuracy FLOAT,
                            loss FLOAT,
                            training_duration INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_data_time TEXT
                        )""",
                    
                    'error_model_history': """
                        CREATE TABLE IF NOT EXISTS error_model_history (
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
                        CREATE TABLE IF NOT EXISTS model_metadata (
                            model_name TEXT PRIMARY KEY,
                            last_trained TIMESTAMP,
                            version TEXT,
                            parameters TEXT,
                            performance_metrics TEXT,
                            is_active BOOLEAN DEFAULT 1,
                            description TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_data_time TEXT
                        )""",
                    
                    'predictions': """
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
                            FOREIGN KEY(draw_number) REFERENCES results(draw_number) ON DELETE CASCADE
                        )"""
                }

                # Создаем таблицы
                for table_name, create_sql in tables.items():
                    try:
                        cursor.execute(create_sql)
                        logger.debug(f"Таблица {table_name} создана/проверена")
                    except sqlite3.Error as e:
                        logger.error(f"Ошибка создания таблицы {table_name}: {str(e)}")
                        raise

                # Создаем индексы
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_results_draw_number ON results(draw_number)",
                    "CREATE INDEX IF NOT EXISTS idx_results_created ON results(created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_predictions_draw ON predictions(draw_number)"
                ]

                for index_sql in indexes:
                    cursor.execute(index_sql)

                # Триггер для автоматического обновления метки времени
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_model_metadata_timestamp
                    AFTER UPDATE ON model_metadata
                    FOR EACH ROW
                    BEGIN
                        UPDATE model_metadata 
                        SET updated_at = CURRENT_TIMESTAMP 
                        WHERE model_name = NEW.model_name;
                    END
                """)

                conn.commit()
                logger.info("Все таблицы и индексы успешно инициализированы")

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

    def _check_should_train(self, cursor) -> Tuple[bool, str]:
        """Проверяет необходимость обучения модели с улучшенной логикой"""
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
        
        last_time = datetime.strptime(last_train['train_time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=None)
        current_time = datetime.now()
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
        """Обновляет информацию о тренировке с сохранением точного времени"""
        try:
            version = self._get_next_version()
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                
                # 1. Проверяем наличие колонки created_at
                cursor.execute("PRAGMA table_info(results)")
                columns = [col[1] for col in cursor.fetchall()]
                time_column = 'created_at' if 'created_at' in columns else 'draw_date'
                
                # 2. Получаем время последних данных (с защитой от None)
                cursor.execute(f"""
                    SELECT MAX({time_column}) FROM results 
                    WHERE {time_column} IS NOT NULL
                """)
                last_data_result = cursor.fetchone()
                last_data_time = (
                    last_data_result[0] 
                    if last_data_result and last_data_result[0] 
                    else datetime.now().isoformat()
                )
                
                # 3. Подготовка данных для вставки
                metrics = json.dumps({'accuracy': accuracy}) if accuracy is not None else None
                local_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # 4. Вставка в историю обучения
                cursor.execute("""
                    INSERT INTO model_training_history 
                    (train_time, data_count, model_version, accuracy, 
                    last_data_time, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (local_time, data_count, version, metrics, last_data_time, local_time))
                
                # 5. Обновление метаданных модели (ИСПРАВЛЕНО: добавлены недостающие параметры)
                cursor.execute("""
                    INSERT OR REPLACE INTO model_metadata 
                    (model_name, version, performance_metrics, 
                    last_data_time, updated_at, last_trained)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ("lstm_model", version, metrics, last_data_time, local_time, local_time))
                
                db.connection.commit()
                logger.info(
                    f"Обновлена информация о обучении. Версия: {version}, "
                    f"Данных: {data_count}, Точность: {accuracy}"
                )
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Ошибка БД при обновлении информации: {str(e)}")
            if 'db' in locals():
                db.connection.rollback()
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {str(e)}", exc_info=True)
            if 'db' in locals():
                db.connection.rollback()
            return False
                
        except sqlite3.Error as e:
            logger.error(f"Ошибка БД при обновлении информации: {str(e)}")
            if 'db' in locals():
                db.connection.rollback()
            return False
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