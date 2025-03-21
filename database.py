import sqlite3
import logging

from datetime import datetime
from config import LOGGING_CONFIG, COMPARISON_LOGGING_CONFIG


# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)

def safe_fromisoformat(date_input):
    """
    Безопасно преобразует строку или объект datetime в объект datetime.
    :param date_input: Строка в формате ISO или объект datetime.
    :return: Объект datetime или None, если преобразование не удалось.
    """
    if isinstance(date_input, datetime):
        return date_input  # Если это уже datetime, возвращаем как есть
    elif isinstance(date_input, str):
        try:
            return datetime.fromisoformat(date_input)
        except ValueError as e:
            logging.error(f"Ошибка при преобразовании даты: {e}")
            return None
    else:
        logging.error(f"Ошибка: ожидалась строка или объект datetime, получен {type(date_input)}.")
        return None

def save_prediction(draw_number, predicted_combination, predicted_field):
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'
        ''')
        table_exists = cursor.fetchone()
        
        if not table_exists:
            cursor.execute('''
                CREATE TABLE predictions (
                    draw_number INTEGER PRIMARY KEY,
                    predicted_combination TEXT,
                    predicted_field INTEGER
                )
            ''')
            logging.info("Таблица predictions создана.")
        
        cursor.execute('SELECT MAX(draw_number) FROM predictions')
        max_draw_number = cursor.fetchone()[0]
        
        if max_draw_number is None:
            max_draw_number = 0
        else:
            max_draw_number = int(max_draw_number)
        
        if int(draw_number) <= max_draw_number:
            draw_number = max_draw_number + 1
            logging.info(f"Номер тиража увеличен до {draw_number}, чтобы избежать конфликта.")
        
        if isinstance(predicted_field, bytes):
            predicted_field = int.from_bytes(predicted_field, byteorder='little')
        
        logging.info(f"Попытка сохранить предсказание для тиража №{draw_number}...")
        cursor.execute('''
            INSERT INTO predictions (draw_number, predicted_combination, predicted_field)
            VALUES (?, ?, ?)
        ''', (draw_number, ', '.join(map(str, predicted_combination)), int(predicted_field)))
        conn.commit()
        logging.info(f"Предсказание для тиража №{draw_number} успешно сохранено.")
    except sqlite3.Error as e:
        logging.error(f"Ошибка при сохранении предсказания: {e}")
    finally:
        conn.close()

def get_last_game():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    cursor.execute('SELECT draw_number, combination, field FROM results ORDER BY draw_number DESC LIMIT 1')
    last_game = cursor.fetchone()
    conn.close()
    if last_game:
        # logging.info(f"Последний тираж: {last_game}")
        return last_game
    logging.info("Нет данных о прошедших играх.")
    return None

def get_max_draw_number():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT MAX(draw_number) FROM (
            SELECT draw_number FROM results
            UNION ALL
            SELECT draw_number FROM predictions
        )
    ''')
    max_draw_number = cursor.fetchone()[0]
    conn.close()
    return int(max_draw_number) if max_draw_number else 1000000

def get_prediction_for_draw(draw_number):
    """
    Возвращает предсказание для указанного тиража.
    """
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    cursor.execute("SELECT predicted_combination, predicted_field FROM predictions WHERE draw_number = ?", (draw_number,))
    result = cursor.fetchone()
    conn.close()
    if result:
        combination = list(map(int, result[0].split(', ')))  # Преобразуем строку в список чисел
        field = result[1]
        return combination, field
    return None

def clean_combination(combination):
    """
    Очищает комбинацию от лишних запятых и пробелов, возвращает список чисел.
    """
    if isinstance(combination, str):  # Если комбинация передана как строка
        # Удаляем лишние запятые и пробелы, разделяем по запятым и фильтруем пустые элементы
        cleaned_numbers = [num.strip() for num in combination.split(",") if num.strip()]
        # Преобразуем в числа
        return [int(num) for num in cleaned_numbers if num.isdigit()]
    elif isinstance(combination, list):  # Если комбинация уже список
        return combination
    else:
        return []

def compare_combinations(predicted, real):
    # """
    # Сравнивает предсказания с реальными данными и записывает результаты в файл.
    # """
    # # Создаем отдельный логгер для сравнений
    # comparison_logger = logging.getLogger('comparison_logger')
    # comparison_logger.setLevel(COMPARISON_LOGGING_CONFIG['level'])
    # """
    # Сравнивает две комбинации и возвращает количество совпавших чисел и список совпавших чисел.
    # """
    predicted_set = set(predicted)  # Преобразуем предсказанную комбинацию в множество
    real_set = set(real)  # Преобразуем реальную комбинацию в множество
    matched_numbers = list(predicted_set.intersection(real_set))  # Находим пересечение
    matched_count = len(matched_numbers)  # Считаем количество совпавших чисел
    return matched_count, matched_numbers

def compare_predictions_with_real_data():
    """
    Сравнивает предсказания с реальными данными и записывает результаты в файл.
    """
    # Создаем отдельный логгер для сравнений
    comparison_logger = logging.getLogger('comparison_logger')
    comparison_logger.setLevel(COMPARISON_LOGGING_CONFIG['level'])

    # Создаем обработчик для записи в файл
    file_handler = logging.FileHandler(
        filename=COMPARISON_LOGGING_CONFIG['filename'],
        encoding=COMPARISON_LOGGING_CONFIG['encoding'],
        mode=COMPARISON_LOGGING_CONFIG['filemode']
    )
    file_handler.setFormatter(logging.Formatter(COMPARISON_LOGGING_CONFIG['format']))

    # Добавляем обработчик к логгеру
    comparison_logger.addHandler(file_handler)

    last_game = get_last_game()  # Получаем последний тираж из базы данных
    if last_game:
        draw_number, real_combination, real_field = last_game
        predicted_data = get_prediction_for_draw(draw_number)  # Получаем предсказание для этого тиража
        if predicted_data:
            predicted_combination, predicted_field = predicted_data

            # Очищаем реальную комбинацию
            real_combination_cleaned = clean_combination(real_combination)

            # Логируем предсказанные и реальные данные
            comparison_logger.info(f"Сравнение для тиража №{draw_number}:")
            comparison_logger.info(f"Предсказанная комбинация: {', '.join(map(str, predicted_combination))}, Поле: {predicted_field}")
            comparison_logger.info(f"Реальная комбинация: {', '.join(map(str, real_combination_cleaned))}, Поле: {real_field}")

            # Сравнение комбинаций
            matched_count, matched_numbers = compare_combinations(predicted_combination, real_combination_cleaned)
            if matched_count > 0:
                comparison_logger.info(f"Комбинация: Совпало {matched_count} чисел ({', '.join(map(str, matched_numbers))})")
            else:
                comparison_logger.info(f"Комбинация: Не совпала (совпало {matched_count} чисел: {', '.join(map(str, matched_numbers))})")

            # Сравнение полей
            if predicted_field == real_field:
                comparison_logger.info("Поле: Совпало")
            else:
                comparison_logger.info("Поле: Не совпало")
        else:
            comparison_logger.info(f"Для тиража №{draw_number} нет предсказания.")
    else:
        comparison_logger.info("Нет данных о прошедших играх.")

# Создание базы данных
def create_database():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            draw_number TEXT NOT NULL,
            combination TEXT NOT NULL,
            field INTEGER NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS last_update (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_update_time TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            draw_number TEXT NOT NULL,
            predicted_combination TEXT NOT NULL,
            predicted_field INTEGER NOT NULL,
            actual_combination TEXT,
            actual_field INTEGER,
            is_correct BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

# Остальные функции из database.py
def load_data():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    cursor.execute('SELECT combination, field FROM results')
    data = cursor.fetchall()
    conn.close()
    return data  

def save_to_database_batch(data):
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    filtered_data = []
    existing_count = 0  # Счётчик уже существующих записей

    for entry in data:
        date, draw_number, combination, field = entry
        cursor.execute('SELECT 1 FROM results WHERE draw_number = ? AND combination = ? AND field = ?', 
                       (draw_number, combination, field))
        if cursor.fetchone():
            existing_count += 1
        else:
            filtered_data.append(entry)

    if filtered_data:
        cursor.executemany('INSERT INTO results (date, draw_number, combination, field) VALUES (?, ?, ?, ?)', filtered_data)
        print(f'Сохранено {len(filtered_data)} новых записей в базу данных.')
    else:
        print("Новых данных для добавления нет.")

    if existing_count > 0:
        print(f'{existing_count} записей уже существовали в базе данных и не были добавлены.')

    conn.commit()  # Добавлен отступ
    conn.close()   # Добавлен отступ

def update_last_update_time():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    current_time = datetime.now().isoformat()
    cursor.execute('INSERT INTO last_update (last_update_time) VALUES (?)', (current_time,))
    conn.commit()
    conn.close()

def get_last_update_time():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    cursor.execute('SELECT last_update_time FROM last_update ORDER BY id DESC LIMIT 1')
    result = cursor.fetchone()
    conn.close()
    return datetime.fromisoformat(result[0]) if result else None