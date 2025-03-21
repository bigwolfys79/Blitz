# model_evaluation.py

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error

def evaluate_models(X_test, y_field_test, y_combination_test, models):
    """
    Оценивает модели на тестовых данных.
    
    Параметры:
        X_test: Тестовые данные.
        y_field_test: Целевые значения для поля.
        y_combination_test: Целевые значения для комбинации.
        models: Словарь с моделями.
    
    Возвращает:
        evaluation_results: Словарь с результатами оценки.
    """
    evaluation_results = {}
    for model_name, model in models.items():
        if model_name.startswith('nn'):
            # Обработка нейронных сетей
            y_field_pred, y_combination_pred = model.predict(X_test)
            field_accuracy = accuracy_score(y_field_test, np.argmax(y_field_pred, axis=1))
            combination_mae = mean_absolute_error(y_combination_test, y_combination_pred)
        else:
            # Обработка Random Forest и XGBoost
            y_field_pred = model[0].predict(X_test)
            y_combination_pred = model[1].predict(X_test)
            field_accuracy = accuracy_score(y_field_test, y_field_pred)
            combination_mae = mean_absolute_error(y_combination_test, y_combination_pred)
        
        evaluation_results[model_name] = {
            'field_accuracy': field_accuracy,
            'combination_mae': combination_mae
        }
    return evaluation_results

def print_model_comparison(evaluation_results):
    """
    Выводит результаты сравнения моделей.
    
    Параметры:
        evaluation_results: Словарь с результатами оценки.
    """
    print("\nСравнение моделей:")
    for model_name, results in evaluation_results.items():
        print(f"{model_name}:")
        print(f"  Точность предсказания поля: {results['field_accuracy']:.4f}")
        print(f"  MAE предсказания комбинации: {results['combination_mae']:.4f}")