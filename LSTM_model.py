
import os
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Уровень ERROR и выше
tf.autograph.set_verbosity(0)  # Отключаем логи AutoGraph

# Отключаем прогресс-бары и информационные сообщения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = все, 3 = ничего
tf.keras.utils.disable_interactive_logging()  # Отключает прогресс-бары
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from typing import Dict, List, Tuple, Optional, Any
import logging
import logging
from config import LOGGING_CONFIG

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)
import joblib
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)
# from contextlib import contextmanager
# # Менеджер контекста для работы с БД
# @contextmanager

class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int] = (8, 1), num_classes: int = 4):
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def __enter__(self):
        """Действия при входе в блок `with`"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Действия при выходе из блока `with`"""
        if exc_type:
            logger.error(f"Произошла ошибка: {exc_value}")
        else:
            logger.info("Контекст завершён без ошибок.")



    def _build_model(self) -> Sequential:
        """Строит архитектуру LSTM модели"""
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(256, return_sequences=True,
                kernel_regularizer=l2(0.01),
                recurrent_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            
            LSTM(128, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train: np.ndarray, y_field: np.ndarray, 
             epochs: int = 150, batch_size: int = 64) -> Optional[Dict[str, list]]:
        """Обучает модель"""
            # Проверка типов данных
        if X_train.dtype != np.int32 or y_field.dtype != np.int32:
            logger.error("Некорректный тип данных. Ожидаются np.int32")
            return None
        
        if not self._validate_data(X_train, y_field):
            return None
            
        X_reshaped = self._prepare_input(X_train)
        y_field = np.array(y_field, dtype=np.int32)
        
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_lstm_model.keras'),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            ReduceLROnPlateau(factor=0.5, patience=10)
        ]
        
        try:
            history = self.model.fit(
                X_reshaped, y_field,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            self.save_model()
            return history.history
        except Exception as e:
            logger.error(f"Ошибка обучения: {str(e)}")
            return None

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

    def save_model(self, filename: str = 'lstm_model.keras') -> bool:
        """Сохраняет модель в файл"""
        try:
            path = os.path.join(self.model_dir, filename)
            self.model.save(path)
            logger.info(f"Модель сохранена: {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения: {str(e)}")
            return False

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


def train_and_save_model(data: dict) -> dict:
    """
    Обучает модель LSTM и сохраняет её
    """
    try:
        # 1. Создаём экземпляр модели
        model = LSTMModel()
        
        # 2. Получаем данные для обучения
        X_train = data['X_train']
        y_field = data['y_field']
        
        # 3. Проверяем, что данных достаточно
        if len(X_train) < 100:
            return {
                'success': False, 
                'message': 'Недостаточно данных для обучения (требуется минимум 100 записей)'
            }
        
        # 4. Обучаем модель
        with LSTMModel() as model:
            history = model.train(X_train, y_field)
        
        if not history:
            return {
                'success': False, 
                'message': 'Ошибка во время обучения модели'
            }
        
        # 5. Возвращаем результат
        return {
            'success': True,
            'data_count': len(X_train),
            'accuracy': history.get('val_accuracy', [0])[-1],  # Последнее значение точности
            'metrics': {
                'loss': history.get('loss', [0])[-1],
                'val_loss': history.get('val_loss', [0])[-1]
            },
            'message': 'Модель успешно обучена'
        }
        
    except Exception as e:
        logger.error(f"Критическая ошибка обучения: {str(e)}", exc_info=True)
        return {
            'success': False, 
            'message': f'Ошибка обучения: {str(e)}'
        }