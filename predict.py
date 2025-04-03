import os
import logging
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import random
import config
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Уровень ERROR и выше
tf.autograph.set_verbosity(0)  # Отключаем логи AutoGraph
from datetime import datetime
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# Отключаем прогресс-бары и информационные сообщения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = все, 3 = ничего
tf.keras.utils.disable_interactive_logging()  # Отключает прогресс-бары
from typing import Dict, List, Tuple, Optional, Any
import sqlite3
from config import PREDICTION_ADJUSTMENT, DATETIME_FORMAT,MODEL_SAVE_PATH, SEQUENCE_LENGTH, NUM_CLASSES, SEQUENCE_LENGTH
from database import DatabaseManager
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# logging = logging.getlogging(__name__)
from contextlib import contextmanager
# Менеджер контекста для работы с БД
@contextmanager
def db_session():
    db = DatabaseManager()
    try:
        yield db
    finally:
        db.close()
class PredictionError(Exception):
    """Пользовательское исключение для ошибок предсказания"""
    pass

class LotteryPredictor:
    def __init__(self, skip_initial_check: bool = False):
        self.model_path = os.path.join(MODEL_SAVE_PATH, 'best_lstm_model.keras')
        self.sequence_length = SEQUENCE_LENGTH
        self.num_classes = NUM_CLASSES
        self.numbers_range = (1, 20)
        self.combination_length = 8
        self.model_name = "LSTM_v1.0"
        self.model = self._load_model()
        self.skip_initial_check = skip_initial_check
        self.adjustment_config = PREDICTION_ADJUSTMENT  # Загружаем конфиг

    def analyze_prediction_effectiveness(self, last_n=20) -> dict:
        """Анализирует эффективность последних N предсказаний"""
        analysis = {
            'total': 0,
            'correct': 0,
            'avg_match': 0,
            'field_accuracy': 0,
            'common_misses': defaultdict(int),
            'field_counts': defaultdict(int),
            'field_accuracy_by_num': defaultdict(float),
            'number_frequency': defaultdict(int),
            'success_rate_by_number': defaultdict(float),
            'numbers': defaultdict(dict)
        }
        
        try:
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                cursor.execute('''
                    SELECT predicted_combination, actual_combination, 
                        predicted_field, actual_field, match_count
                    FROM predictions 
                    WHERE actual_combination IS NOT NULL
                    ORDER BY draw_number DESC 
                    LIMIT ?
                ''', (last_n,))

                # Собираем сырые данные
                field_data = defaultdict(lambda: {'correct': 0, 'total': 0})
                number_stats = defaultdict(lambda: {'attempts': 0, 'hits': 0, 'miss_count': 0})
                
                for row in cursor.fetchall():
                    analysis['total'] += 1
                    match_count = row['match_count']
                    
                    # Общая статистика
                    analysis['avg_match'] += match_count
                    if match_count >= 5:
                        analysis['correct'] += 1
                    
                    # Статистика по полям
                    pred_field = row['predicted_field']
                    field_data[pred_field]['total'] += 1
                    if pred_field == row['actual_field']:
                        analysis['field_accuracy'] += 1
                        field_data[pred_field]['correct'] += 1
                    
                    # Анализ чисел
                    if row['actual_combination']:
                        actual_nums = set(map(int, row['actual_combination'].split(',')))
                        predicted_nums = set(map(int, row['predicted_combination'].split(',')))
                        
                        # Обновляем статистику для каждого числа
                        for num in predicted_nums:
                            number_stats[num]['attempts'] += 1
                            if num in actual_nums:
                                number_stats[num]['hits'] += 1
                            else:
                                number_stats[num]['miss_count'] += 1
                        
                        # Частота использования чисел
                        for num in predicted_nums:
                            analysis['number_frequency'][num] += 1
                
                # Заполняем 'numbers' в analysis
                for num in number_stats:
                    attempts = number_stats[num]['attempts']
                    hits = number_stats[num]['hits']
                    miss_count = number_stats[num]['miss_count']
                    success_rate = (hits / attempts * 100) if attempts > 0 else 0
                    
                    analysis['numbers'][num] = {
                        'attempts': attempts,
                        'success_rate': round(success_rate, 2),
                        'miss_count': miss_count
                    }
                
                # Расчет производных метрик
                if analysis['total'] > 0:
                    analysis['avg_match'] = round(analysis['avg_match'] / analysis['total'], 2)
                    analysis['correct'] = round(analysis['correct'] / analysis['total'] * 100, 2)
                    analysis['field_accuracy'] = round(analysis['field_accuracy'] / analysis['total'] * 100, 2)
                    
                    # Точность по полям
                    for field in field_data:
                        analysis['field_accuracy_by_num'][field] = round(
                            field_data[field]['correct'] / field_data[field]['total'] * 100, 2
                        ) if field_data[field]['total'] > 0 else 0

        except Exception as e:
            logging.error(f"Ошибка анализа эффективности: {e}")
        
        return analysis

    def get_previous_prediction_stats(self, last_n=10) -> dict:
        """Возвращает статистику по последним N предсказаниям"""
        stats = {
            'total': 0,
            'same_combinations': 0,
            'same_field': 0,
            'last_combinations': [],
            'last_fields': []
        }
        
        try:
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                cursor.execute('''
                    SELECT predicted_combination, predicted_field 
                    FROM predictions 
                    ORDER BY draw_number DESC 
                    LIMIT ?
                ''', (last_n,))
                
                for row in cursor.fetchall():
                    stats['total'] += 1
                    stats['last_combinations'].append(row['predicted_combination'])
                    stats['last_fields'].append(row['predicted_field'])
                
                # Анализ повторяющихся комбинаций
                if stats['last_combinations']:
                    stats['same_combinations'] = stats['last_combinations'].count(stats['last_combinations'][0])
                    stats['same_field'] = stats['last_fields'].count(stats['last_fields'][0])
                    
        except Exception as e:
            logging.error(f"Ошибка получения статистики предсказаний: {e}")
        
        return stats    

    def get_actual_result(self, draw_number: int) -> Optional[Tuple[str, int]]:
        """Получает реальные результаты для указанного тиража из таблицы results"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute('''
                SELECT combination, field FROM results
                WHERE draw_number = ?
            ''', (draw_number,))
            result = cursor.fetchone()
            return (result['combination'], result['field']) if result else None
    
    def analyze_number_trends(self, last_n_draws=50) -> Tuple[List[int], List[int]]:
        """Анализирует частоту выпадения чисел и возвращает отсортированные по порядку списки"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute('''
                SELECT combination FROM results 
                ORDER BY draw_number DESC 
                LIMIT ?
            ''', (last_n_draws,))
            
            # Считаем частоту чисел
            freq = defaultdict(int)
            for row in cursor.fetchall():
                nums = list(map(int, row['combination'].split(',')))
                for num in nums:
                    freq[num] += 1
            
            # Сортируем по частоте (по убыванию)
            sorted_by_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            
            # Получаем топ-5 горячих и холодных чисел
            hot_nums = [num for num, cnt in sorted_by_freq[:5]]
            cold_nums = [num for num, cnt in sorted_by_freq[-5:]]
            
            # Сортируем числа по порядку (от меньшего к большему)
            hot_sorted = sorted(hot_nums)
            cold_sorted = sorted(cold_nums)
            
            return hot_sorted, cold_sorted

    def get_prediction_accuracy(self) -> str:
        """Возвращает статистику точности предсказаний в формате 'верные/все (процент)'"""
        try:
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                
                # Получаем количество верных предсказаний
                cursor.execute('SELECT COUNT(*) FROM predictions WHERE is_correct = 1')
                correct = cursor.fetchone()[0]
                
                # Получаем общее количество проверенных предсказаний
                cursor.execute('SELECT COUNT(*) FROM predictions WHERE is_correct IS NOT NULL')
                total = cursor.fetchone()[0]
                
                if total > 0:
                    percentage = (correct / total) * 100
                    return f"{correct}/{total} ({percentage:.2f}%)"
                return "Нет данных для расчета"
                
        except Exception as e:
            logging.error(f"Ошибка расчета точности: {e}")
            return "Ошибка расчета" 

    def get_historical_accuracy(self, days=7):
        """Возвращает точность за последние N дней"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM predictions
                WHERE date(timestamp) >= date('now', ?)
            ''', (f'-{days} days',))
            total, correct = cursor.fetchone()
            return f"{correct}/{total} ({(correct/total)*100:.2f}%)" if total > 0 else "Нет данных"              
   
    def check_last_prediction(self) -> dict:
        """Сравнивает последнее предсказание с фактическими результатами"""
        result = {
            'draw_number': 0,
            'predicted': [],
            'actual': [],
            'matched': [],
            'match_count': 0,
            'field_match': False,
            'result_code': '0-0',
            'is_winning': False,
            'winning_tier': 'Проигрыш'
        }
        
        try:
            last_draw = self._get_last_draw_number()
            if not last_draw:
                return result
                
            result['draw_number'] = last_draw

            # Получаем предсказание
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                cursor.execute('''
                    SELECT predicted_combination, predicted_field 
                    FROM predictions 
                    WHERE draw_number = ?
                ''', (last_draw,))
                prediction = cursor.fetchone()
                
                if not prediction:
                    return result
                    
                result['predicted'] = list(map(int, prediction['predicted_combination'].split(',')))
                result['predicted_field'] = prediction['predicted_field']

            # Получаем фактические результаты
            actual_data = self.get_actual_result(last_draw)
            if not actual_data:
                return result

            result['actual'] = list(map(int, actual_data[0].split(',')))
            result['actual_field'] = actual_data[1]

            # Анализ совпадений
            matched = set(result['predicted']) & set(result['actual'])
            result['matched'] = sorted(matched)
            result['match_count'] = len(matched)
            result['field_match'] = (result['predicted_field'] == result['actual_field'])

            # Определение результата
            winning_combinations = {
                (8, True): ('8-1', True, "Джекпот (8 чисел + поле)"),
                (8, False): ('8-0', True, "Главный приз (8 чисел)"),
                (7, True): ('7-1', True, "Суперприз (7 чисел + поле)"),
                (7, False): ('7-0', True, "Суперприз (7 чисел)"),
                (6, True): ('6-1', True, "Крупный выигрыш (6+поле)"),
                (6, False): ('6-0', True, "Крупный выигрыш (6 чисел)"),
                (5, True): ('5-1', True, "Большой выигрыш (5+поле)"),
                (5, False): ('5-0', True, "Большой выигрыш (5 чисел)"),
                (4, True): ('4-1', True, "Выигрыш (4+поле)"),
                (0, False): ('0-0', True, "Минимальный выигрыш (ничего не совпало)")
            }
            
            losing_combinations = {
                (4, False): ('4-0', False, "Проигрыш"),
                (3, True): ('3-1', False, "Проигрыш"),
                (3, False): ('3-0', False, "Проигрыш"),
                (2, True): ('2-1', False, "Проигрыш"),
                (2, False): ('2-0', False, "Проигрыш"),
                (1, True): ('1-1', False, "Проигрыш"),
                (1, False): ('1-0', False, "Проигрыш"),
                (0, True): ('0-1', False, "Проигрыш")
            }

            key = (result['match_count'], result['field_match'])
            if key in winning_combinations:
                result.update(zip(['result_code', 'is_winning', 'winning_tier'], winning_combinations[key]))
            elif key in losing_combinations:
                result.update(zip(['result_code', 'is_winning', 'winning_tier'], losing_combinations[key]))

            # Сохранение результатов
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                cursor.execute('''
                    UPDATE predictions SET
                        actual_combination = ?,
                        actual_field = ?,
                        is_correct = ?,
                        matched_numbers = ?,
                        match_count = ?,
                        result_code = ?,
                        winning_tier = ?,
                        checked_at = CURRENT_TIMESTAMP
                    WHERE draw_number = ?
                ''', (
                    actual_data[0],
                    actual_data[1],
                    int(result['is_winning']),
                    ','.join(map(str, result['matched'])),
                    result['match_count'],
                    result['result_code'],
                    result['winning_tier'],
                    last_draw
                ))
                db.connection.commit()

            # Анализ трендов
            hot, cold = self.analyze_number_trends()
            
            logging.info(f"""
            Анализ тиража #{last_draw}:
            ├── Предсказание: {result['predicted']} (поле: {result['predicted_field']})
            ├── Фактически: {result['actual']} (поле: {result['actual_field']})
            ├── Совпадений: {result['match_count']} чисел ({result['matched']})
            ├── Результат: {result['result_code']} ({result['winning_tier']})
            ├── Статус: {'✅ Выигрыш' if result['is_winning'] else '❌ Проигрыш'}
            ├── 🔥 Горячие: {', '.join(map(str, hot))}
            └── ❄️ Холодные: {', '.join(map(str, cold))}
            """)

        except Exception as e:
            logging.error(f"Ошибка при проверке предсказания: {e}", exc_info=True)
        
        return result
    
    def get_performance_statistics(self, days: int = None) -> dict:
        """Возвращает статистику эффективности предсказаний за указанный период"""
        if days is None:
            from config import REPORT_PERIOD_DAYS
            days = REPORT_PERIOD_DAYS

        stats = {
            'total_predictions': 0,
            'winning_predictions': 0,
            'winning_rate': 0.0,
            'winning_tiers': {
                "8-1 (полное совпадение)": 0,
                "8-0 (все числа, не поле)": 0,
                "7-1 (7 чисел + поле)": 0,
                "7-0 (7 чисел)": 0,
                "6-1 (6 чисел + поле)": 0,
                "6-0 (6 чисел)": 0,
                "5-1 (5 чисел + поле)": 0,
                "5-0 (5 чисел)": 0,
                "4-1 (4 числа + поле)": 0,
                "0-0 (0 чисел)": 0,
                "проигрыш": 0
            },
            'average_match_count': 0.0,
            'field_accuracy': 0.0
        }
        
        try:
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                
                # 1. Получаем общую статистику
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN result_code IN ('8-1', '8-0', '7-1', '7-0', '6-1', '6-0', '5-1', '5-0', '4-1', '0-0') THEN 1 ELSE 0 END) as wins,
                        AVG(match_count) as avg_matches,
                        AVG(CASE WHEN predicted_field = actual_field THEN 1 ELSE 0 END) as field_acc
                    FROM predictions 
                    WHERE actual_combination IS NOT NULL
                    AND date(checked_at) >= date('now', '-{days} days')
                ''')
                
                row = cursor.fetchone()
                if row:
                    stats['total_predictions'] = row['total'] or 0
                    stats['winning_predictions'] = row['wins'] or 0
                    stats['average_match_count'] = round(float(row['avg_matches'] or 0), 1)
                    stats['field_accuracy'] = round(float(row['field_acc'] or 0) * 100, 1)
                    
                    if stats['total_predictions'] > 0:
                        stats['winning_rate'] = round(
                            (stats['winning_predictions'] / stats['total_predictions']) * 100, 2
                        )
                
                # 2. Получаем детальное распределение результатов
                cursor.execute(f'''
                    SELECT 
                        result_code,
                        COUNT(*) as count
                    FROM predictions
                    WHERE result_code IS NOT NULL
                    AND actual_combination IS NOT NULL
                    AND date(checked_at) >= date('now', '-{days} days')
                    GROUP BY result_code
                ''')
                
                # Словарь для преобразования кодов в читаемые названия
                code_to_tier = {
                    '8-1': "8-1 (полное совпадение)",
                    '8-0': "8-0 (все числа, не поле)",
                    '7-1': "7-1 (7 чисел + поле)",
                    '7-0': "7-0 (7 чисел)",
                    '6-1': "6-1 (6 чисел + поле)",
                    '6-0': "6-0 (6 чисел)",
                    '5-1': "5-1 (5 чисел + поле)",
                    '5-0': "5-0 (5 чисел)",
                    '4-1': "4-1 (4 числа + поле)",
                    '0-0': "0-0 (0 чисел)"
                }
                
                for row in cursor.fetchall():
                    result_code = row['result_code']
                    count = row['count'] or 0
                    
                    if result_code in code_to_tier:
                        stats['winning_tiers'][code_to_tier[result_code]] = count
                    else:
                        stats['winning_tiers']["проигрыш"] += count
        
        except Exception as e:
            logging.error(f"Ошибка получения статистики: {e}", exc_info=True)
        
        return stats
      
    def generate_performance_report(self, days: int = None) -> str:
        """Генерирует текстовый отчет об эффективности предсказаний"""
        if days is None:
            from config import REPORT_PERIOD_DAYS
            days = REPORT_PERIOD_DAYS

        stats = self.get_performance_statistics(days)
        
        # Определяем точные ширины колонок
        left_width = 24  # Ширина левой колонки (названия)
        left_width1 = 15
        right_width = 10  # Ширина правой колонки (значения)
        box_width = left_width + right_width + 3  # +3 для границ и пробелов
        
        # Формируем верхнюю часть отчета
        report_lines = [
            f"\n{'📊 ОТЧЕТ ЭФФЕКТИВНОСТИ 📊':^{box_width}}",
            f"╔{'═' * (box_width-2)}╗",
            f"║ {'Период:':<{left_width1}} {f'последние {days} дней':<{right_width}}",
            f"║ {'Всего проверено:':<{left_width}} {stats['total_predictions']:<{right_width}}",
            f"║ {'Выигрышных:':<{left_width}} {f"{stats['winning_predictions']} ({stats['winning_rate']:.2f}%)":<{right_width}}",
            f"║ {'Среднее совпадений:':<{left_width}} {f"{stats['average_match_count']:.1f}/8":<{right_width}}",
            f"║ {'Точность поля:':<{left_width}} {f"{stats['field_accuracy']:.1f}%":<{right_width}}",
            f"╠{'═' * (box_width-2)}╣",
            f"║ {'Распределение результатов:':<{box_width-3}} "
        ]
        
        # Добавляем распределение результатов
        result_distribution = [
            "8-1 (полное совпадение)",
            "8-0 (все числа, не поле)",
            "7-1 (7 чисел + поле)",
            "7-0 (7 чисел)",
            "6-1 (6 чисел + поле)",
            "6-0 (6 чисел)",
            "5-1 (5 чисел + поле)",
            "5-0 (5 чисел)",
            "4-1 (4 числа + поле)",
            "0-0 (0 чисел)",
            "проигрыш"
        ]
        
        for result in result_distribution:
            count = stats['winning_tiers'].get(result, 0)
            report_lines.append(f"║ {result:<{left_width}} {count:^{right_width}}")

        # Закрываем отчет
        report_lines.append(f"└{'─' * (box_width-2)}┘")
        
        # Собираем все строки в один отчет
        report = "\n".join(report_lines)
        
        logging.info(report)
        return report

    def _load_model(self):
        """Загружает предварительно обученную модель Keras"""
        try:
            # Проверяем существование файла модели
            if not os.path.exists(self.model_path):
                error_msg = (
                    f"Файл модели не найден по пути: {self.model_path}\n"
                    "Возможные решения:\n"
                    "1. Сначала обучите модель, выполнив main.py\n"
                    "2. Проверьте правильность пути в config.py (MODEL_SAVE_PATH)\n"
                    "3. Убедитесь, что файл не был удален"
                )
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Пытаемся загрузить модель
            logging.info(f"Загрузка модели из {self.model_path}")
            model = tf.keras.models.load_model(self.model_path)
            logging.info("Модель успешно загружена")
            return model

        except tf.errors.OpError as e:
            error_msg = f"Ошибка загрузки файла модели (возможно поврежден файл): {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = f"Непредвиденная ошибка при загрузке модели: {str(e)}"
            logging.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def prediction_exists(self, draw_number: int) -> bool:
        """Проверяет, существует ли предсказание для указанного тиража"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute('''
                SELECT 1 FROM predictions 
                WHERE draw_number = ?
                LIMIT 1
            ''', (draw_number,))
            return cursor.fetchone() is not None

    def save_or_update_prediction(self, draw_number: int, combination: List[int], field: int) -> bool:
        """Сохраняет или обновляет предсказание"""
        try:
            # Логируем начало операции
            logging.info("Начало сохранения/обновления предсказания.")
            logging.debug(f"Параметры: draw_number={draw_number}, combination={combination}, field={field}")
            
            # Валидация входных данных
            if not isinstance(draw_number, int) or draw_number <= 0:
                logging.error(f"Некорректный номер тиража: {draw_number}")
                return False
            
            if not combination or len(combination) != 8:
                logging.error(f"Комбинация должна содержать 8 чисел. Получено: {len(combination)}")
                return False
            
            if not isinstance(field, int):
                try:
                    field = int(field)
                    logging.debug(f"Преобразовано в тип int: {field}")
                except ValueError:
                    logging.error(f"Ошибка преобразования. Невозможно привести {field} к типу int.")
                    return False
            
            if field < 1 or field > 4:
                logging.error(f"Номер поля должен быть от 1 до 4. Получено: {field}")
                return False
            
            logging.info(f"Проверка параметров завершена успешно.")

            # Преобразуем комбинацию в строку
            comb_str = ', '.join(map(str, sorted(combination)))
            logging.debug(f"Комбинация преобразована в строку: {comb_str}")
            
            # Работа с БД
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                
                # Проверяем существование записи
                cursor.execute(
                    'SELECT 1 FROM predictions WHERE draw_number = ?',
                    (draw_number,)
                )
                exists = cursor.fetchone() is not None
                logging.debug(f"Существует ли запись с draw_number={draw_number}: {exists}")
                
                if exists:
                    # Обновляем существующее предсказание
                    cursor.execute('''
                        UPDATE predictions 
                        SET predicted_combination = ?,
                            predicted_field = ?,
                            model_name = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE draw_number = ?
                    ''', (comb_str, field, self.model_name, draw_number))
                    logging.info(f"Обновлено предсказание для тиража {draw_number}")
                else:
                    # Создаем новое предсказание
                    current_time = datetime.now().strftime(DATETIME_FORMAT)
                    cursor.execute('''
                        INSERT INTO predictions 
                        (draw_number, predicted_combination, predicted_field, model_name, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (draw_number, comb_str, field, self.model_name, current_time))
                    logging.info(f"Создано предсказание для тиража {draw_number}")
                
                db.connection.commit()
                logging.info(f"Операция для тиража {draw_number} успешно завершена.")
                return True

        except sqlite3.Error as e:
            logging.error(f"Ошибка БД при сохранении предсказания: {e}")
            if 'db' in locals():
                db.connection.rollback()
            return False

        except Exception as e:
            logging.error(f"Неожиданная ошибка: {e}")
            return False

    def get_last_combinations(self, limit: int) -> List[List[int]]:
        """Получает последние комбинации из БД"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            cursor.execute(
                "SELECT combination FROM results ORDER BY draw_number DESC LIMIT ?",
                (limit,)
            )
            return [[int(num) for num in row['combination'].split(',')] for row in cursor.fetchall()]

    def _get_last_draw_number(self) -> int:
        """Возвращает номер последнего тиража"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()  # Получаем курсор
            cursor.execute("SELECT MAX(draw_number) FROM results")
            result = cursor.fetchone()  # Теперь fetchone() будет работать
            return result[0] if result else 0

    def _adjust_prediction(self, predicted_comb: List[int], predicted_field: int) -> Tuple[List[int], int]:
        """Улучшенная корректировка с интеграцией вашего конфига"""
        cfg = config.PREDICTION_ADJUSTMENT
        predicted_comb = [int(num) for num in predicted_comb]
        
        # Инициализация трендов только если включено
        hot_nums, cold_nums = [], []
        if cfg['USE_TRENDS']:
            hot_nums, cold_nums = self.analyze_number_trends(last_n_draws=30)
        
        effectiveness = self.analyze_prediction_effectiveness(last_n=30)
        
        # Определение проблемных чисел
        bad_numbers = set()
        for num in range(1, 21):
            stats = effectiveness['numbers'].get(num, {})
            success_rate = stats.get('success_rate', 0)
            miss_count = stats.get('miss_count', 0)
            attempts = stats.get('attempts', 0)
            
            # Основные критерии
            if (attempts >= cfg['MIN_ATTEMPTS'] and success_rate < cfg['MIN_SUCCESS_RATE']) or \
            (miss_count > cfg['MAX_MISSES']):
                bad_numbers.add(num)
            
            # Критерий для холодных чисел
            if cfg['USE_TRENDS'] and num in cold_nums:
                cold_rank = cold_nums.index(num)
                if cold_rank < cfg['COLD_RANK_THRESHOLD'] and miss_count > cfg['COLD_RANK_MAX_MISSES']:
                    bad_numbers.add(num)
        
        # Стратегия выбора кандидатов
        candidates = []
        if cfg['ADJUSTMENT_STRATEGY'] == 'smart':
            candidates = [
                num for num in range(1, 21)
                if num not in bad_numbers and 
                effectiveness['numbers'].get(num, {}).get('success_rate', 0) > 40
            ]
        elif cfg['ADJUSTMENT_STRATEGY'] == 'hot':
            candidates = hot_nums[:10]
        elif cfg['ADJUSTMENT_STRATEGY'] == 'cold':
            candidates = cold_nums[-10:]
        else:  # random
            candidates = [n for n in range(1, 21) if n not in bad_numbers]
        
        # Корректировка чисел
        adjusted_comb = predicted_comb.copy()
        changes_made = 0
        
        for i in range(len(adjusted_comb)):
            if changes_made >= cfg['NUMBERS_TO_ADJUST']:
                break
                
            if adjusted_comb[i] in bad_numbers or cfg['ADJUSTMENT_STRATEGY'] != 'random':
                available = [n for n in candidates if n not in adjusted_comb]
                if available:
                    adjusted_comb[i] = random.choice(available)
                    changes_made += 1
        
        # Ротация поля
        last_fields = self.get_previous_prediction_stats(5)['last_fields']
        if last_fields.count(predicted_field) >= cfg['MAX_REPEATS']:
            adjusted_field = random.choice([f for f in range(1,5) if f != predicted_field])
        else:
            adjusted_field = predicted_field
        
        # Логирование
        logger.info(f"Adjusted {changes_made} numbers using {cfg['ADJUSTMENT_STRATEGY']} strategy")
        
        # 6. Логирование с подробной статистикой
        changes = [f"{old}→{new}" for old, new in zip(predicted_comb, adjusted_comb) if old != new]
        logger.info(
            f"\n{' КОРРЕКТИРОВКА ПРЕДСКАЗАНИЯ ':=^50}\n"
            f"Изменения: {', '.join(changes)}\n"
            f"Новая комбинация: {sorted(adjusted_comb)}\n"
            f"Поле: {predicted_field}→{adjusted_field}\n"
            f"Горячие числа: {hot_nums[:5]}\n"
            f"Холодные числа: {cold_nums[-3:]}\n"
            f"Проблемные числа: {sorted(bad_numbers)}\n"
            f"Точность: {effectiveness['correct']}%\n"
            f"Среднее совпадений: {effectiveness['avg_match']}\n"
            f"{'='*50}"
        )
        return sorted(adjusted_comb), adjusted_field

    def predict_next(self) -> Tuple[int, List[int], int]:
        """Предсказывает следующий тираж с автоматической корректировкой"""
        try:
            # Стандартное предсказание
            next_draw_num, predicted_comb, predicted_field = self._standard_prediction()
            
            # Проверяем, нужно ли корректировать
            pred_stats = self.get_previous_prediction_stats()

            # Логируем статистику
            logger.info(
                f"Статистика предсказаний: "
                f"Повторений комбинаций: {pred_stats['same_combinations']}, "
                f"Повторений поля: {pred_stats['same_field']}"
            )
            
            if (pred_stats['same_combinations'] >= self.adjustment_config['MAX_REPEATS'] or
                pred_stats['same_field'] >= self.adjustment_config['MAX_REPEATS']):
                
                logging.warning(
                    f"Обнаружено повторение предсказаний: "
                    f"{pred_stats['same_combinations']} одинаковых комбинаций, "
                    f"{pred_stats['same_field']} одинаковых полей. "
                    f"Применяю корректировку."
                )
                
                predicted_comb, predicted_field = self._adjust_prediction(predicted_comb, predicted_field)
            else:
                logger.info("Условия для корректировки не выполнены.")
            return next_draw_num, predicted_comb, predicted_field
            
        except Exception as e:
            logging.error(f"Ошибка : {str(e)}", exc_info=True)
            raise PredictionError(f"Ошибка предсказания: {str(e)}")

    def _standard_prediction(self) -> Tuple[int, List[int], int]:
        """Стандартное предсказание без модификаций"""
        # 1. Получаем ожидаемую длину последовательности из модели
        expected_seq_length = self.model.input_shape[1]
        
        # 2. Получаем данные с учетом SEQUENCE_LENGTH из конфига
        last_combinations = self.get_last_combinations(max(self.sequence_length, expected_seq_length))
        
        # 3. Адаптируем данные под требования модели
        if len(last_combinations) > expected_seq_length:
            last_combinations = last_combinations[-expected_seq_length:]
            logging.info(f"Используются последние {expected_seq_length} комбинаций из {len(last_combinations)} доступных")
        
        # 4. Проверка минимального количества данных
        if len(last_combinations) < expected_seq_length:
            raise ValueError(
                f"Модель требует {expected_seq_length} комбинаций. "
                f"Доступно: {len(last_combinations)}. "
                f"SEQUENCE_LENGTH в config.py: {self.sequence_length}"
            )

        # 5. Подготовка входных данных
        X = np.array([last_combinations], dtype=np.float32)
        X = (X - 1) / 19  # Нормализация [1-20] -> [0-1]

        # 6. Предсказание модели
        field_probs, comb_probs = self.model.predict(X, verbose=0)

        # 7. Обработка предсказанного поля (1-4)
        predicted_field = np.argmax(field_probs[0]) + 1

        # 8. Обработка предсказанной комбинации (8 уникальных чисел)
        predicted_comb = []
        used_numbers = set()
        
        for i in range(8):  # Для каждого из 8 чисел
            probs = comb_probs[0][i]  # Вероятности для i-го числа
            sorted_num_indices = np.argsort(probs)[::-1]  # Сортировка по убыванию
            
            for num_idx in sorted_num_indices:
                num = num_idx + 1  # Преобразуем индекс 0-19 -> число 1-20
                if num not in used_numbers:
                    predicted_comb.append(num)
                    used_numbers.add(num)
                    break

        # 9. Получаем номер следующего тиража
        last_draw_num = self._get_last_draw_number()
        next_draw_num = last_draw_num + 1 if last_draw_num else 1

        return next_draw_num, sorted(predicted_comb), predicted_field
 
    def predict_and_save(self) -> bool:
        """Выполняет предсказание и сохраняет результат
        
        Returns:
            bool: True если предсказание успешно сохранено, False в случае ошибки
        """
        try:
            # Получаем предсказание
            next_draw, comb, field = self.predict_next()

            # Дополнительная проверка поля
            if field < 1 or field > 4:
                logging.warning(f"Некорректное поле {field}. Установлено значение по умолчанию 1")
            # field = 1
            
            # Форматированный вывод
            comb_str = '  '.join(f"{num:2d}" for num in sorted(comb))
            border = "─" * 30
            logging.info(f"\n{'🎱 ПРЕДСКАЗАНИЕ 🎱':^30}")
            logging.info(border)
            logging.info(f"Тираж:      #{next_draw}")
            logging.info(f"Комбинация: {comb_str}")
            logging.info(f"Поле:       {field}")
            logging.info(f"Модель:     {self.model_name}")
            logging.info(border + "\n")
            
            # Сохраняем предсказание в транзакции
            success = self.save_or_update_prediction(next_draw, comb, field)
            
            if not success:
                logging.warning("Не удалось сохранить предсказание в БД")
                return False
                
            logging.info(f"Успешно сохранено предсказание для тиража #{next_draw}")
            return True
            
        except Exception as e:
            logging.error(f"Критическая ошибка при сохранении предсказания: {str(e)}", exc_info=True)
            return False

    def update_actual_results(self, draw_id: int, actual_combination: str, actual_field: int) -> None:
        """Обновляет реальные результаты розыгрыша и проверяет предсказание"""
        try:
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                
                # Обновляем запись и сразу проверяем совпадение
                cursor.execute('''
                    UPDATE predictions 
                    SET actual_combination = ?,
                        actual_field = ?,
                        is_correct = (predicted_combination = ? AND predicted_field = ?),
                        checked_at = CURRENT_TIMESTAMP
                    WHERE draw_number = ?
                ''', (
                    actual_combination,
                    actual_field,
                    actual_combination,
                    actual_field,
                    draw_id
                ))
                
                db.connection.commit()
                
                # Проверяем, была ли обновлена запись
                if cursor.rowcount == 0:
                    logging.warning(f"Прогноз для розыгрыша {draw_id} не найден!")
                else:
                    # Получаем результат сравнения для логирования
                    cursor.execute('''
                        SELECT is_correct FROM predictions
                        WHERE draw_number = ?
                    ''', (draw_id,))
                    result = cursor.fetchone()
                    is_correct = result['is_correct'] if result else False
                    logging.info(f"Результаты розыгрыша {draw_id} обновлены. Совпадение: {'Да' if is_correct else 'Нет'}")

        except Exception as e:
            logging.error(f"Ошибка при обновлении результатов: {e}", exc_info=True)
            raise

def main():
    try:
        predictor = LotteryPredictor()
        predictor.predict_and_save()
        predictor.check_last_prediction()  # Проверяем предпоследний тираж
    except Exception as e:
        logging.error(f"Ошибка в основном потоке: {e}", exc_info=True)

if __name__ == "__main__":
    main()