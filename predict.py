# predict.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает логи TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '3'  # Отключаем oneDNN
import tensorflow as tf

# tf.debugging.set_log_device_placement(True)
# tf.get_logger().setLevel('DEBUG')  # Включаем DEBUG-логи
# tf.get_logger().setLevel('ERROR')  # Отключает логи уровня INFO и WARNING
tf.get_logger().setLevel('INFO')
import random
import json

import numpy as np

from sklearn.model_selection import train_test_split
from database import load_data, save_prediction, get_last_game, get_max_draw_number, compare_predictions_with_real_data

from model_evaluation import evaluate_models, print_model_comparison
from prediction_utils import predict_combination_and_field, predict_with_random_forest, print_all_predictions

from model_utils import build_lstm_model, build_cnn_model, build_hybrid_model, train_random_forest, train_xgboost, load_model, save_model
from data_utils import prepare_data_multitask

import logging
from config import LOGGING_CONFIG # Импортируем LOGGING_CONFIG

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)

# Пути к моделям (относительные)
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')  # Папка models рядом с main.py

MODEL_PATHS = {
    'nn_lstm': os.path.join(MODELS_DIR, 'nn_lstm_model.keras'),
    'nn_cnn': os.path.join(MODELS_DIR, 'nn_cnn_model.keras'),
    'nn_hybrid': os.path.join(MODELS_DIR, 'nn_hybrid_model.keras'),
    'random_forest': os.path.join(MODELS_DIR, 'random_forest_model.keras'),
    'xgboost': os.path.join(MODELS_DIR, 'xgboost_model.keras')
}

def predict_data():
    # Загрузка данных
    data = load_data()
    X, y_field, y_combination = prepare_data_multitask(data)
    
    if len(X) > 0 and len(y_field) > 0 and len(y_combination) > 0:
        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_field_train, y_field_test, y_combination_train, y_combination_test = train_test_split(
            X, y_field, y_combination, test_size=0.2, random_state=42
        )
        
        # Определение параметров
        input_length = X.shape[1]  # Длина входной последовательности (8 чисел)
        num_classes = len(set(y_field))  # Количество классов для поля
        
        # Загрузка или создание моделей
        models = {}
        for model_name, model_path in MODEL_PATHS.items():
            model = load_model(model_path, model_name)
            if model is None:
                if model_name == 'nn_lstm':
                    model = build_lstm_model(input_length, num_classes)
                elif model_name == 'nn_cnn':
                    model = build_cnn_model(input_length, num_classes)
                elif model_name == 'nn_hybrid':
                    model = build_hybrid_model(input_length, num_classes)
                elif model_name == 'random_forest':
                    model = train_random_forest(X_train, y_field_train, y_combination_train)
                elif model_name == 'xgboost':
                    model = train_xgboost(X_train, y_field_train, y_combination_train)
                logging.info(f"Создана новая модель: {model_name}")
            else:
                logging.info(f"Загружена существующая модель: {model_name}")
            models[model_name] = model
        
        # Дообучение нейронных сетей
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        for model_name, model in models.items():
            if model_name.startswith('nn'):
                logging.info(f"Дообучение модели: {model_name}")
                history = model.fit(
                    X_train, 
                    {'field_output': y_field_train, 'combination_output': y_combination_train},
                    validation_data=(X_test, {'field_output': y_field_test, 'combination_output': y_combination_test}),
                    epochs=100, 
                    batch_size=32, 
                    callbacks=[early_stopping], 
                    verbose=0
                )
                logging.info(f"Дообучение модели {model_name} завершено.")
        
        # Сохранение всех моделей
        for model_name, model in models.items():
            save_model(model, MODEL_PATHS[model_name], model_name)
            logging.info(f"Модель {model_name} сохранена.")
        
        # Оценка моделей
        evaluation_results = evaluate_models(X_test, y_field_test, y_combination_test, models)
        print_model_comparison(evaluation_results)
        
        # Предсказание новой комбинации
        new_combination = random.sample(range(1, 21), 8)
        print_all_predictions(models, new_combination)
        
        # Выбор лучшей модели
        best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['field_accuracy'])
        print(f"\nЛучшая модель: {best_model_name}")
        
        # Предсказание лучшей модели
        if best_model_name.startswith('nn'):
            model = models[best_model_name]
            predicted_field, predicted_combination = predict_combination_and_field(model, new_combination)
        else:
            model = models[best_model_name]
            predicted_field, predicted_combination = predict_with_random_forest(model[0], model[1], new_combination)
        
        print(f"\nПредсказание лучшей модели ({best_model_name}):")
        print(f"  Поле: {predicted_field}")
        print(f"  Комбинация: {predicted_combination}")
        
        # Сохранение предсказания
        max_draw_number = get_max_draw_number()
        next_draw_number = max_draw_number + 1
        save_prediction(str(next_draw_number), predicted_combination, predicted_field)
        
        # Сравнение с реальными данными
        last_game = get_last_game()
        if last_game:
            compare_predictions_with_real_data()
        else:
            print("Нет данных о прошедших играх.")
if __name__ == "__main__":
    predict_data()