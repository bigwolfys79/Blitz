# prediction_utils.py

import os
import random
import numpy as np
import tensorflow as tf

import logging
from config import LOGGING_CONFIG  # Импортируем LOGGING_CONFIG

# Отключаем логи TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает логи TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Отключает логи Keras

# Отключаем интерактивные логи прогресса TensorFlow/Keras
tf.keras.utils.disable_interactive_logging()

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)

def ensure_unique_combination(predicted_combination, min_num=1, max_num=20):
    unique_combination = list(set(predicted_combination))
    while len(unique_combination) < len(predicted_combination):
        new_number = random.randint(min_num, max_num)
        if new_number not in unique_combination:
            unique_combination.append(new_number)
    return sorted(unique_combination)


def predict_combination_and_field(model, new_combination):
    """
    Предсказывает комбинацию и поле с использованием модели.
    
    Параметры:
        model: Модель для предсказания (или кортеж из field_model и combination_model).
        new_combination: Новая комбинация.
    
    Возвращает:
        predicted_field: Предсказанное поле.
        predicted_combination: Предсказанная комбинация.
    """
    if isinstance(model, tuple):
        # Если передали кортеж (field_model, combination_model)
        field_model, combination_model = model
        y_field_pred = field_model.predict(np.array([new_combination]))
        y_combination_pred = combination_model.predict(np.array([new_combination]))
    else:
        # Если передали одну модель (например, LSTM, CNN)
        y_field_pred, y_combination_pred = model.predict(np.array([new_combination]))
    
    predicted_field = np.argmax(y_field_pred, axis=1)[0] + 1
    predicted_combination = [int(x) + 1 for x in y_combination_pred[0]]
    return predicted_field, predicted_combination

def predict_with_random_forest(field_model, combination_model, new_combination):
    """
    Предсказывает комбинацию и поле с использованием Random Forest.
    
    Параметры:
        field_model: Модель для предсказания поля.
        combination_model: Модель для предсказания комбинации.
        new_combination: Новая комбинация.
    
    Возвращает:
        predicted_field: Предсказанное поле.
        predicted_combination: Предсказанная комбинация.
    """
    new_combination = np.array([new_combination])
    predicted_field = field_model.predict(new_combination)[0] + 1
    predicted_combination = [int(x) + 1 for x in combination_model.predict(new_combination)[0]]
    return predicted_field, predicted_combination

def predict_with_xgboost(field_model, combination_model, new_combination):
    """
    Предсказывает комбинацию и поле с использованием XGBoost.
    
    Параметры:
        field_model: Модель для предсказания поля.
        combination_model: Модель для предсказания комбинации.
        new_combination: Новая комбинация.
    
    Возвращает:
        predicted_field: Предсказанное поле.
        predicted_combination: Предсказанная комбинация.
    """
    new_combination = np.array([new_combination])
    predicted_field = field_model.predict(new_combination)[0] + 1
    predicted_combination = [int(x) + 1 for x in combination_model.predict(new_combination)[0]]
    return predicted_field, predicted_combination

def print_all_predictions(models, new_combination):
    """
    Выводит предсказания всех моделей.
    
    Параметры:
        models: Словарь с моделями.
        new_combination: Новая комбинация.
    """
    print("\nПредсказания всех моделей:")
    for model_name, model in models.items():
        if model_name.startswith('nn'):
            predicted_field, predicted_combination = predict_combination_and_field(model, new_combination)
        else:
            predicted_field, predicted_combination = predict_with_random_forest(model[0], model[1], new_combination)
        print(f"{model_name}:")
        print(f"  Поле: {predicted_field}")
        print(f"  Комбинация: {predicted_combination}")