
import os
import sys
import time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Уровень ERROR и выше
tf.autograph.set_verbosity(0)  # Отключаем логи AutoGraph

# Отключаем прогресс-бары и информационные сообщения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = все, 3 = ничего
tf.keras.utils.disable_interactive_logging()  # Отключает прогресс-бары
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras import Model # type: ignore
from tensorflow.keras.layers import Conv1D, Input, Reshape, LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from typing import Dict, List, Tuple, Optional, Any
import logging
from config import LOGGING_CONFIG, BATCH_SIZE, SEQUENCE_LENGTH, MODEL_INPUT_SHAPE, NUM_CLASSES, NUMBERS_RANGE, COMBINATION_LENGTH

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)
import joblib
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)
# from contextlib import contextmanager
# # Менеджер контекста для работы с БД
# @contextmanager

# class ProgressBar:
#     def __init__(self, total, length=50):
#         self.total = total
#         self.length = length
#         self.start_time = time.time()
#         self.stopped_early = False
    
#     def update(self, epoch, stopped_early=False):
#         self.stopped_early = stopped_early
        
#         # Если обучение остановлено, показываем 100%
#         if stopped_early:
#             progress = 1.0
#             epoch = self.total - 1  # Показываем последнюю эпоху как total
#         else:
#             progress = (epoch + 1) / self.total
        
#         filled = int(self.length * progress)
#         bar = '█' * filled + '-' * (self.length - filled)
#         elapsed = time.time() - self.start_time
        
#         # Расчет оставшегося времени
#         if stopped_early:
#             eta = 0
#         elif epoch > 0:
#             eta = (elapsed / (epoch + 1)) * (self.total - (epoch + 1))
#         else:
#             eta = elapsed * (self.total - 1)
        
#         status = " (early stop)" if stopped_early else ""
#         sys.stdout.write(
#             f"\rОбучение: |{bar}| {int(100 * progress)}% "
#             f"[{epoch + 1}/{self.total}] "
#             f"Осталось: {eta:.1f} сек{status}"
#         )
#         sys.stdout.flush()
    
#     def close(self):
#         if self.stopped_early:
#             # Показываем полный прогресс при досрочном завершении
#             bar = '█' * self.length
#             sys.stdout.write(
#                 f"\rОбучение: |{bar}| 100% "
#                 f"[{self.total}/{self.total}] "
#                 f"Осталось: 0.0 сек (обучение завершено досрочно)\n"
#             )
#         else:
#             sys.stdout.write(" (обучение завершено)\n")
#         sys.stdout.flush()

class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int] = MODEL_INPUT_SHAPE, num_classes: int = NUM_CLASSES):
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
                
            # progress = ProgressBar(total=epochs)  # Инициализируем прогресс-бар    
            X_normalized = (X_train - 1) / 19.0
            
            # # Добавляем флаг досрочной остановки
            # early_stop_flag = False

            # def check_early_stop(epoch, logs):
            #     nonlocal early_stop_flag
            #     if self.model.stop_training:
            #         early_stop_flag = True
            #         progress.update(epoch, stopped_early=True)

            # callbacks = [
            #     EarlyStopping(monitor='val_comb_output_accuracy', patience=20),
            #     LambdaCallback(on_epoch_end=lambda epoch, logs: progress.update(epoch)),
            #     LambdaCallback(on_train_end=lambda logs: progress.close()),
            #     LambdaCallback(on_epoch_end=check_early_stop)
            # ]

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
            # if early_stop_flag:
            #     progress.update(epochs-1, stopped_early=True)  # Принудительно показываем 100%
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


def train_and_save_model(data: dict) -> dict:
    """Обучает модель LSTM и сохраняет её"""
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
        
        return {
            'success': True,
            'data_count': len(X_train),
            'field_accuracy': history.get('val_field_accuracy', [0])[-1],
            'comb_accuracy': history.get('val_comb_accuracy', [0])[-1],
            'loss': history.get('val_loss', [0])[-1],
            'message': 'Модель успешно обучена'
        }
        
    except Exception as e:
        logger.error(f"Критическая ошибка обучения: {str(e)}", exc_info=True)
        return {
            'success': False, 
            'message': f'Ошибка обучения: {str(e)}'
        }