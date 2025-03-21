# model_utils.py

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает логи TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем oneDNN
import tensorflow as tf
import json
import logging
import joblib
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    MaxPooling1D, BatchNormalization, Bidirectional, Concatenate, Input,
    Dense, LSTM, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# Отключаем логи TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает логи TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Отключает логи Keras

# Отключаем интерактивные логи прогресса TensorFlow/Keras
tf.keras.utils.disable_interactive_logging()

# Настройка логгера
logging.basicConfig(level=logging.INFO)

def build_lstm_model(input_length, num_classes):
    inputs = Input(shape=(input_length,))
    
    # Увеличенный Embedding слой
    x = Embedding(input_dim=100, output_dim=64)(inputs)
    
    # Первый Bidirectional LSTM слой
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    
    # Второй Bidirectional LSTM слой
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    
    # Механизм Multi-Head Attention
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Concatenate()([x, attention])
    x = Dropout(0.5)(x)
    
    # Полносвязные слои с BatchNormalization
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Выходные слои
    field_output = Dense(num_classes, activation='softmax', name='field_output')(x)
    combination_output = Dense(input_length, activation='linear', name='combination_output')(x)
    
    model = Model(inputs=inputs, outputs=[field_output, combination_output])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={'field_output': 'sparse_categorical_crossentropy', 'combination_output': 'mse'},
        metrics={'field_output': 'accuracy', 'combination_output': 'mae'}
    )
    return model

def build_cnn_model(input_length, num_classes):
    inputs = Input(shape=(input_length,))
    
    # Увеличенный Embedding слой
    x = Embedding(input_dim=50, output_dim=32)(inputs)
    
    # Первый Conv1D слой
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Второй Conv1D слой
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Третий Conv1D слой
    x = Conv1D(32, kernel_size=7, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.3)(x)
    
    # Полносвязные слои
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    
    # Выходные слои
    field_output = Dense(num_classes, activation='softmax', name='field_output')(x)
    combination_output = Dense(input_length, activation='linear', name='combination_output')(x)
    
    model = Model(inputs=inputs, outputs=[field_output, combination_output])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={'field_output': 'sparse_categorical_crossentropy', 'combination_output': 'mse'},
        metrics={'field_output': 'accuracy', 'combination_output': 'mae'}
    )
    return model

def build_hybrid_model(input_length, num_classes):
    inputs = Input(shape=(input_length,))
    
    # Увеличенный Embedding слой
    x = Embedding(input_dim=50, output_dim=32)(inputs)
    
    # Первый Conv1D слой
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Второй Conv1D слой
    x = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # Bidirectional LSTM слой
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    
    # Полносвязные слои
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    
    # Выходные слои
    field_output = Dense(num_classes, activation='softmax', name='field_output')(x)
    combination_output = Dense(input_length, activation='linear', name='combination_output')(x)
    
    model = Model(inputs=inputs, outputs=[field_output, combination_output])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={'field_output': 'sparse_categorical_crossentropy', 'combination_output': 'mse'},
        metrics={'field_output': 'accuracy', 'combination_output': 'mae'}
    )
    return model

def train_random_forest(X, y_field, y_combination):
    rf_field = RandomForestClassifier(
        n_estimators=200,  # Увеличение количества деревьев
        max_depth=20,      # Увеличение глубины деревьев
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    rf_field.fit(X, y_field)
    
    rf_combination = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    rf_combination.fit(X, y_combination)
    
    return rf_field, rf_combination

def train_xgboost(X, y_field, y_combination):
    xgb_field = XGBClassifier(
        n_estimators=200,  # Увеличение количества деревьев
        max_depth=10,      # Увеличение глубины деревьев
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_field.fit(X, y_field)
    
    xgb_combination = XGBRegressor(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_combination.fit(X, y_combination)
    
    return xgb_field, xgb_combination

def load_model(model_path, model_type, input_length=None, num_classes=None):
    """
    Загружает модель в зависимости от её типа.
    
    Параметры:
        model_path: Путь к модели.
        model_type: Тип модели ('nn_lstm', 'nn_cnn', 'random_forest', 'xgboost').
        input_length: Длина входной последовательности.
        num_classes: Количество классов.
    
    Возвращает:
        model: Загруженная модель.
    """
    if not os.path.exists(model_path):
        logging.info(f"Модель {model_path} не найдена. Будет создана новая.")
        return None
    logging.info(f"Загружена модель {model_path} ({model_type})")
    
    if model_type.startswith('nn'):
        return tf.keras.models.load_model(model_path)
    elif model_type in ('random_forest', 'xgboost'):
        return joblib.load(model_path)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
def save_model(model, model_path, model_type, input_length=None, num_classes=None):
    """
    Сохраняет модель в зависимости от её типа.
    
    Параметры:
        model: Модель для сохранения.
        model_path: Путь для сохранения модели.
        model_type: Тип модели ('nn_lstm', 'nn_cnn', 'random_forest', 'xgboost').
        input_length: Длина входной последовательности.
        num_classes: Количество классов.
    """
    if model_type.startswith('nn'):
        model.save(model_path)
    elif model_type in ('random_forest', 'xgboost'):
        joblib.dump(model, model_path)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
def compare_combinations(predicted, real):
    """
    Сравнивает две комбинации и возвращает количество совпавших чисел и их список.
    """
    predicted_set = set(predicted)
    real_set = set(real)
    matched_numbers = predicted_set.intersection(real_set)  # Находим совпадающие числа
    return len(matched_numbers), sorted(matched_numbers)  # Возвращаем количество и список