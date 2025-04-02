
import os
import sqlite3
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Уровень ERROR и выше
tf.autograph.set_verbosity(0)  # Отключаем логи AutoGraph

# Отключаем прогресс-бары и информационные сообщения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = все, 3 = ничего
tf.keras.utils.disable_interactive_logging()  # Отключает прогресс-бары
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import Model # type: ignore
from tensorflow.keras.layers import Conv1D, Input, Reshape, LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from typing import Dict, List, Tuple, Optional, Any
import logging
from config import DATETIME_FORMAT, LOGGING_CONFIG, BATCH_SIZE, MODEL_INPUT_SHAPE, NUM_CLASSES, NUMBERS_RANGE, COMBINATION_LENGTH

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int] = MODEL_INPUT_SHAPE, num_classes: int = NUM_CLASSES):
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._load_or_rebuild_model()

    def _load_or_rebuild_model(self) -> Model:
        """Загружает модель или пересоздает если она несовместима с текущими параметрами"""
        model_path = os.path.join(self.model_dir, 'best_lstm_model.keras')
        
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                # Проверяем совместимость модели
                if model.input_shape[1] == self.input_shape[0]:
                    logger.info("Модель загружена и совместима с текущими параметрами")
                    return model
                logger.warning(f"Модель несовместима (ожидалось {self.input_shape}, получено {model.input_shape})")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {str(e)}")
        
        logger.info("Создание новой модели с текущими параметрами")
        return self._build_model()    

    def __enter__(self):
        """Действия при входе в блок `with`"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Действия при выходе из блока `with`"""
        if exc_type:
            logger.error(f"Произошла ошибка: {exc_value}")
        else:
            logger.info("Контекст завершён без ошибок.")

    def _build_model(self) -> Model:
        """Строит модель LSTM с правильной архитектурой"""
        input_layer = Input(shape=MODEL_INPUT_SHAPE)
        
        # Основная ветвь обработки
        x = Conv1D(64, kernel_size=8, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        
        x = LSTM(512, return_sequences=True, kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.4)(x)
        x = LSTM(256, return_sequences=False, kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        
        # Ветвь для предсказания поля (1-4)
        field_branch = Dense(128, activation='relu')(x)
        field_branch = Dense(64, activation='relu')(field_branch)
        output_field = Dense(NUM_CLASSES, activation='softmax', name='field_output')(field_branch)
        
        # Ветвь для предсказания комбинации (8 чисел)
        comb_branch = Dense(512, activation='relu')(x)
        comb_output = Dense(8 * NUMBERS_RANGE, activation='softmax')(comb_branch)
        output_comb = Reshape((8, NUMBERS_RANGE), name='comb_output')(comb_output)

        model = Model(inputs=input_layer, outputs=[output_field, output_comb])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss={
                'field_output': 'sparse_categorical_crossentropy',
                'comb_output': 'categorical_crossentropy'
            },
            metrics={
                'field_output': 'accuracy',
                'comb_output': 'accuracy'
            }
        )
        return model

    def train(self, X_train: np.ndarray, y_field: np.ndarray, 
         y_comb: np.ndarray, epochs: int = 150, 
         batch_size: int = BATCH_SIZE) -> Optional[Dict[str, list]]:
        """Обучение модели с явным указанием режима early stopping"""
        try:
            # Проверка и нормализация данных
            if len(X_train) == 0:
                logger.error("Нет данных для обучения")
                return None

            X_normalized = (X_train - 1) / 19.0

            # Callbacks с явным указанием режима
            callbacks = [
                EarlyStopping(
                    monitor='val_comb_output_accuracy',
                    patience=20,
                    restore_best_weights=True,
                    mode='max',
                    verbose=0
                ),
                ModelCheckpoint(
                    os.path.join(self.model_dir, 'best_lstm_model.keras'),
                    monitor='val_comb_output_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    mode='min',
                    verbose=0
                ),
        ]
            
            # Обучение модели
            history = self.model.fit(
                X_normalized,
                {
                    'field_output': y_field,
                    'comb_output': y_comb
                },
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            self.save_model()
            return history.history
                
        except ValueError as e:
            logger.error(f"Ошибка в данных: {str(e)}", exc_info=True)
            return None
            
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"Недостаточно памяти GPU: {str(e)}", exc_info=True)
            return None
            
        except Exception as e:
            logger.error(f"Критическая ошибка обучения: {str(e)}", exc_info=True)
            return None

    def save_model(self, filename: str = 'lstm_model.keras') -> bool:
        """Сохраняет модель"""
        try:
            path = os.path.join(self.model_dir, filename)
            self.model.save(path)
            logger.info(f"Модель сохранена: {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения: {str(e)}")
            return False

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Оценивает качество модели"""
        if not self._validate_data(X_test, y_test):
            return {}
            
        X_ready = self._prepare_input(X_test)
        y_test = np.array(y_test, dtype=np.int32)
        
        try:
            loss, accuracy = self.model.evaluate(X_ready, y_test)
            return {
                'loss': loss,
                'accuracy': accuracy
            }
        except Exception as e:
            logger.error(f"Ошибка оценки: {str(e)}")
            return {}

    def load_model(self, filename: str = 'lstm_model.keras') -> bool:
        """Загружает модель из файла"""
        try:
            path = os.path.join(self.model_dir, filename)
            if not os.path.exists(path):
                logger.error(f"Файл модели не найден: {path}")
                return False
                
            self.model = load_model(path)
            logger.info(f"Модель загружена: {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки: {str(e)}")
            return False

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Подготавливает входные данные"""
        return ((X - 1) / 19).reshape((X.shape[0], 8, 1))

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Проверяет корректность данных"""
        if len(X) == 0 or len(y) == 0:
            logger.error("Пустые данные")
            return False
        return True


def train_and_save_model(data: dict, db_conn=None) -> dict:
    """Обучает модель LSTM и сохраняет её, обновляет метаданные в БД"""
    try:
        model = LSTMModel()
        
        # Проверяем наличие всех необходимых данных
        required_keys = ['X_train', 'y_field', 'y_comb']
        if not all(key in data for key in required_keys):
            return {
                'success': False,
                'message': f'Отсутствуют необходимые данные. Требуются: {required_keys}'
            }
            
        X_train = data['X_train']
        y_field = data['y_field']
        y_comb = data['y_comb']
        last_draw_number = data.get('last_draw_number')
        
        if len(X_train) < 100:
            return {
                'success': False, 
                'message': 'Недостаточно данных для обучения (минимум 100 записей)'
            }
        
        with LSTMModel() as model:
            history = model.train(X_train, y_field, y_comb)
        
        if not history:
            return {
                'success': False, 
                'message': 'Ошибка во время обучения модели'
            }
        
        # Получаем метрики
        field_accuracy = history.get('val_field_accuracy', [0])[-1]
        comb_accuracy = history.get('val_comb_accuracy', [0])[-1]
        loss = history.get('val_loss', [0])[-1]
        
        # Обновляем данные в БД если передан connection
        if db_conn:
            try:
                cursor = db_conn.cursor()
                current_time = datetime.now().strftime(DATETIME_FORMAT)
                
                # 1. Получаем количество данных и время последних данных
                cursor.execute("SELECT COUNT(*) FROM results")
                total_data = cursor.fetchone()[0]
                
                # Определяем столбец для времени
                cursor.execute("PRAGMA table_info(results)")
                columns = [col[1] for col in cursor.fetchall()]
                time_column = 'created_at' if 'created_at' in columns else 'draw_date'
                
                # Получаем время последних данных
                if last_draw_number:
                    cursor.execute(f"""
                        SELECT {time_column} FROM results 
                        WHERE draw_number = ? 
                        LIMIT 1
                    """, (last_draw_number,))
                    last_data_time = cursor.fetchone()[0] or current_time
                else:
                    last_data_time = current_time
                
                # 2. Считаем новые данные с момента последнего обучения
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
                
                # 3. Получаем следующую версию модели
                cursor.execute("""
                    SELECT version FROM model_metadata 
                    WHERE model_name = 'lstm_model' 
                    ORDER BY last_trained DESC LIMIT 1
                """)
                last_version = cursor.fetchone()
                version = "1.0" if not last_version else f"{float(last_version[0]) + 0.1:.1f}"
                
                # 4. Вставляем в model_training_history
                cursor.execute("""
                    INSERT INTO model_training_history 
                    (train_time, data_count, new_data_count, model_version, 
                    accuracy, last_data_time, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    current_time, 
                    total_data,
                    new_data_count,
                    version,
                    comb_accuracy, 
                    last_data_time, 
                    current_time
                ))
                
                # 5. Обновляем model_metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO model_metadata 
                    (model_name, version, performance_metrics, 
                    last_data_time, updated_at, last_trained,
                    total_data_count, new_data_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    "lstm_model",
                    version,
                    json.dumps({
                        'field_accuracy': field_accuracy,
                        'comb_accuracy': comb_accuracy,
                        'loss': loss
                    }),
                    last_data_time,
                    current_time,
                    current_time,
                    total_data,
                    new_data_count
                ))
                
                db_conn.commit()
                logger.info("Данные обучения успешно сохранены в БД")
            except Exception as e:
                logger.error(f"Ошибка при обновлении БД: {str(e)}")
                if db_conn:
                    db_conn.rollback()
        
        return {
            'success': True,
            'data_count': len(X_train),
            'field_accuracy': field_accuracy,
            'comb_accuracy': comb_accuracy,
            'loss': loss,
            'message': 'Модель успешно обучена'
        }
        
    except Exception as e:
        logger.error(f"Критическая ошибка обучения: {str(e)}", exc_info=True)
        return {
            'success': False, 
            'message': f'Ошибка обучения: {str(e)}'
        }