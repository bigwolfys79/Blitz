# data_utils.py

import numpy as np
import logging

def prepare_data_multitask(data, sequence_length=8):
    """
    Подготавливает данные для многозадачного обучения.
    
    Параметры:
        data: Список кортежей (комбинация, поле).
        sequence_length: Длина комбинации (по умолчанию 8).
    
    Возвращает:
        X: Входные данные (комбинации).
        y_field: Целевые значения для поля.
        y_combination: Целевые значения для комбинации.
    """
    X, y_field, y_combination = [], [], []
    for combination, field in data:
        try:
            nums = list(map(int, combination.split(', ')))
            if len(nums) == sequence_length:
                X.append(nums)
                y_field.append(field - 1)  # Поле начинается с 1, приводим к индексу 0
                y_combination.append([x - 1 for x in nums])  # Комбинация: 1-20 -> 0-19
        except (ValueError, AttributeError):
            logging.warning(f"Некорректные данные: {combination}")
    return np.array(X), np.array(y_field), np.array(y_combination)