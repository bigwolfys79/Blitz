import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import json
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Уровень ERROR и выше
tf.autograph.set_verbosity(0)  # Отключаем логи AutoGraph

# Отключаем прогресс-бары и информационные сообщения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = все, 3 = ничего
tf.keras.utils.disable_interactive_logging()  # Отключает прогресс-бары
from typing import List, Tuple, Optional
import sqlite3
import logging
from config import MODEL_SAVE_PATH, SEQUENCE_LENGTH, NUM_CLASSES, SEQUENCE_LENGTH
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
    def __init__(self):
        self.model_path = os.path.join(MODEL_SAVE_PATH, 'best_lstm_model.keras')
        self.sequence_length = SEQUENCE_LENGTH
        self.num_classes = NUM_CLASSES
        self.numbers_range = (1, 20)
        self.combination_length = 8
        self.model_name = "LSTM_v1.0"
        self.model = self._load_model()

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
        
    def _save_training_result(self, result: dict):
        """Сохраняет результаты обучения одним запросом"""
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            
            # Вставка в историю обучения
            cursor.execute("""
                INSERT INTO model_training_history 
                (train_time, data_count, model_version, accuracy, loss, training_duration)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                result['data_count'],
                result['version'],
                result.get('accuracy'),
                result.get('loss'),
                result.get('duration')
            ))
            
            # Обновление метаданных
            cursor.execute("""
                INSERT OR REPLACE INTO model_metadata 
                (model_name, last_trained, version, performance_metrics)
                VALUES (?, ?, ?, ?)
            """, (
                "lstm_model",
                result['version'],
                json.dumps({
                    'accuracy': result.get('accuracy'),
                    'loss': result.get('loss')
                })
            ))
            
            db.connection.commit()    

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

    def check_last_prediction(self) -> None:
        """Сравнивает последнее предсказание с фактическими результатами"""
        try:
            # 1. Обновляем схему БД
            with DatabaseManager() as db:
                if not db.update_schema():
                    logging.error("Не удалось обновить схему БД")
                    return
                
                # 2. Получаем последний тираж
                last_draw = self._get_last_draw_number()
                if not last_draw:
                    logging.warning("Нет данных о тиражах")
                    return

                # 3. Получаем предсказание
                cursor = db.connection.cursor()
                cursor.execute('''
                    SELECT predicted_combination, predicted_field 
                    FROM predictions 
                    WHERE draw_number = ?
                ''', (last_draw,))
                prediction = cursor.fetchone()
                
                if not prediction:
                    logging.info(f"Нет предсказания для тиража {last_draw}")
                    return
                    
                pred_comb, pred_field = prediction['predicted_combination'], prediction['predicted_field']
                pred_numbers = sorted(map(int, pred_comb.split(',')))

            # 4. Получаем фактические результаты
            actual_data = self.get_actual_result(last_draw)
            if not actual_data:
                logging.warning(f"Нет результатов для тиража {last_draw}")
                return

            actual_comb, actual_field = actual_data
            actual_numbers = sorted(map(int, actual_comb.split(',')))

            # 5. Анализ совпадений
            matched_set = set(pred_numbers) & set(actual_numbers)
            matched_numbers = sorted(matched_set)
            num_matched = len(matched_numbers)
            total_numbers = len(pred_numbers)
            match_percentage = (num_matched / total_numbers * 100) if total_numbers > 0 else 0
            field_match = (int(pred_field) == int(actual_field))
            is_correct = int(num_matched == total_numbers and field_match)

            # 6. Сохраняем результаты
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                cursor.execute('''
                    UPDATE predictions SET
                        actual_combination = ?,
                        actual_field = ?,
                        is_correct = ?,
                        matched_numbers = ?,
                        match_count = ?,
                        checked_at = CURRENT_TIMESTAMP
                    WHERE draw_number = ?
                ''', (
                    actual_comb,
                    actual_field,
                    is_correct,
                    ','.join(map(str, matched_numbers)),
                    num_matched,
                    last_draw
                ))
                db.connection.commit()

            # 7. Получаем статистику точности
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                
                # Общая точность
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(is_correct) as correct
                    FROM predictions
                    WHERE actual_combination IS NOT NULL
                ''')
                stats = cursor.fetchone()
                total = stats['total'] if stats else 0
                correct = stats['correct'] if stats else 0
                accuracy = f"{correct}/{total} ({(correct/total)*100:.1f}%)" if total > 0 else "Нет данных"
                
                # Недельная точность
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(is_correct) as correct
                    FROM predictions
                    WHERE actual_combination IS NOT NULL
                    AND date(created_at) >= date('now', '-7 days')
                ''')
                weekly_stats = cursor.fetchone()
                weekly_total = weekly_stats['total'] if weekly_stats else 0
                weekly_correct = weekly_stats['correct'] if weekly_stats else 0
                weekly_accuracy = f"{weekly_correct}/{weekly_total} ({(weekly_correct/weekly_total)*100:.1f}%)" if weekly_total > 0 else "Нет данных"

            # 8. Анализ трендов
            hot_numbers, cold_numbers = self.analyze_number_trends()

            # 9. Форматированный вывод
            logging.info(f"""
            🔍 Анализ тиража #{last_draw}
            ├── Предсказание: [{' '.join(map(str, pred_numbers))}] (поле: {pred_field})
            ├── Результат:    [{' '.join(map(str, actual_numbers))}] (поле: {actual_field})
            ├── Совпадения:   [{' '.join(map(str, matched_numbers))}] ({num_matched}/{total_numbers} = {match_percentage:.1f}%)
            ├── Поле:        {'✅ Совпало' if field_match else '❌ Не совпало'}
            ├── Результат:   {'🎯 Полное совпадение' if is_correct else '🔻 Частичное совпадение'}
            ├── Точность:    {accuracy}
            ├── 📊 Недельная: {weekly_accuracy}
            ├── 🔥 Горячие:  {', '.join(map(str, hot_numbers))}
            └── ❄️ Холодные: {', '.join(map(str, cold_numbers))}
            """)

        except sqlite3.Error as e:
            logging.error(f"Ошибка базы данных при проверке предсказания: {e}")
        except ValueError as e:
            logging.error(f"Ошибка формата данных: {e}")
        except Exception as e:
            logging.error(f"Критическая ошибка при проверке: {e}", exc_info=True)


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
                    cursor.execute('''
                        INSERT INTO predictions 
                        (draw_number, predicted_combination, predicted_field, model_name, created_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (draw_number, comb_str, field, self.model_name))
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

    def predict_next(self) -> Tuple[int, List[int], int]:
        """Предсказывает следующий тираж с учетом SEQUENCE_LENGTH"""
        try:
            # 1. Получаем последние SEQUENCE_LENGTH комбинаций
            last_combinations = self.get_last_combinations(self.sequence_length)
            if len(last_combinations) < self.sequence_length:
                raise ValueError(f"Требуется {self.sequence_length} комбинаций, получено {len(last_combinations)}")

            # 2. Подготавливаем входные данные (форма: [1, SEQUENCE_LENGTH, COMBINATION_LENGTH])
            X = np.array([last_combinations], dtype=np.float32)  # Форма: (1, 30, 8)
            X = (X - 1) / 19  # Нормализация [1-20] -> [0-1]

            # 3. Предсказание модели
            field_probs, comb_probs = self.model.predict(X, verbose=0)

            # 4. Обработка предсказанного поля (1-4)
            predicted_field = np.argmax(field_probs[0]) + 1

            # 5. Обработка предсказанной комбинации (8 уникальных чисел)
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

            # 6. Получаем номер следующего тиража
            last_draw_num = self._get_last_draw_number()
            next_draw_num = last_draw_num + 1 if last_draw_num else 1

            return next_draw_num, sorted(predicted_comb), predicted_field

        except Exception as e:
            logging.error(f"Ошибка предсказания: {str(e)}", exc_info=True)
            raise PredictionError(f"Ошибка предсказания: {str(e)}")


    def _predict_field(self, sequences: List[List[int]]) -> int:
        """
        Предсказывает номер поля (гарантированно 1-4)
        
        Returns:
            int: Номер поля (всегда 1, 2, 3 или 4)
        """
        try:
            # Подготовка данных
            sequences = sequences[:self.sequence_length]
            X = np.array([(num - 1)/19 for seq in sequences for num in seq])
            X = X.reshape(1, self.sequence_length, self.combination_length)
            X = X[:, :, 0].reshape(1, self.sequence_length, 1)
            
            # Предсказание
            field_probs = self.model.predict(X, verbose=0)[0]
            field = int(np.argmax(field_probs)) + 1  # Явное преобразование в int
            
            # Гарантируем корректный диапазон
            return max(1, min(4, field))
            
        except Exception as e:
            logging.error(f"Ошибка предсказания поля: {e}")
            return 1  # Значение по умолчанию

    def load_data_from_db() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Загружает данные из базы данных для обучения моделей"""
        try:
            with DatabaseManager() as db:
                cursor = db.connection.cursor()
                cursor.execute('SELECT combination, field FROM results ORDER BY draw_number')
                results = cursor.fetchall()
                
                if not results:
                    return np.array([]), np.array([]), np.array([])
                
                X = []
                y_field = []
                y_comb = []
                
                for row in results:
                    try:
                        comb = [int(x.strip()) for x in row['combination'].split(',')]
                        if len(comb) != 8:
                            continue
                        
                        # One-hot кодировка комбинации (8 чисел × 20 вариантов)
                        comb_encoded = np.zeros((8, 20))
                        for i, num in enumerate(comb):
                            if 1 <= num <= 20:
                                comb_encoded[i, num - 1] = 1
                        
                        X.append(comb)
                        y_field.append(row['field'] - 1)  # Поле 1-4 → 0-3
                        y_comb.append(comb_encoded)
                        
                    except (ValueError, AttributeError) as e:
                        logging.warning(f"Ошибка обработки строки {row}: {str(e)}")
                        continue
                
                return np.array(X), np.array(y_field), np.array(y_comb)
                
        except Exception as e:
            logging.error(f"Ошибка загрузки данных из БД: {str(e)}", exc_info=True)
            return np.array([]), np.array([]), np.array([])


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