import os
import sqlite3
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
import logging
from config import CHROMEDRIVER_PATH, BASE_URL, LOGGING_CONFIG, TF_CPP_MIN_LOG_LEVEL, TF_ENABLE_ONEDNN_OPTS
from database import DatabaseManager, save_to_database_batch, safe_fromisoformat
from database import DatabaseManager
# Устанавливаем переменные окружения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = TF_ENABLE_ONEDNN_OPTS

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)
from contextlib import contextmanager
# Менеджер контекста для работы с БД
@contextmanager
def db_session():
    db = DatabaseManager()
    try:
        yield db
    finally:
        db.close()
def setup_driver():
    service = Service(CHROMEDRIVER_PATH)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--log-level=3')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    return webdriver.Chrome(service=service, options=options)

def get_total_pages(driver, url):
    """
    Определяет количество страниц для заданного URL.
    """
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'li.pagination-item a.pagination-link span.pagination-link__text'))
        )
        page_elements = driver.find_elements(By.CSS_SELECTOR, 'li.pagination-item a.pagination-link span.pagination-link__text')
        pages = [int(element.text.strip()) for element in page_elements if element.text.strip().isdigit()]
        return max(pages) if pages else 1
    except Exception as e:
        logging.error(f'Ошибка при определении количества страниц: {e}')
        return 1

def parse_page(driver, url):
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.table-row'))
        )
        rows = driver.find_elements(By.CSS_SELECTOR, 'div.table-row')
        draw_data = []
        for row in rows:
            try:
                draw_number_elements = row.find_elements(By.CSS_SELECTOR, 'div.table-cell.w22 .table-cell__value')
                draw_number = draw_number_elements[0].text.strip() if draw_number_elements else None
                if not draw_number:
                    continue

                draw_date_elements = row.find_elements(By.CSS_SELECTOR, 'div.table-cell.w12 .table-cell__value')
                draw_date = draw_date_elements[0].text.strip() if draw_date_elements else None
                if not draw_date:
                    continue

                combination_elements = row.find_elements(By.CSS_SELECTOR, 'div[style="letter-spacing: 1px;"]')
                if not combination_elements:
                    continue

                combination_raw = combination_elements[0].text.strip()
                combination_parts = combination_raw.split(', ')
                combination = ', '.join(combination_parts[:-1])
                field = int(combination_parts[-1])

                draw_data.append((draw_date, draw_number, combination, field))
            except Exception as row_error:
                logging.error(f"Ошибка обработки строки: {row_error}")

        return draw_data
    except Exception as e:
        logging.error(f'Ошибка при парсинге страницы: {e}')
        return []



def parse_data(pages_to_parse=1):
    driver = None
    try:
        driver = setup_driver()
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # Используем единое соединение для всех операций
        with DatabaseManager() as db:
            # Получаем информацию о БД
            cursor = db.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM results")
            result = cursor.fetchone()  # Получаем результат один раз
            total_records = result[0] if result else 0  # Безопасное извлечение
            
            last_update_time = db.get_last_update_time()
            last_update_date = None
            
            if last_update_time:
                try:
                    if isinstance(last_update_time, str):
                        last_update_time = safe_fromisoformat(last_update_time)
                    last_update_date = last_update_time.strftime('%Y-%m-%d')
                except (AttributeError, ValueError) as e:
                    logging.warning(f"Ошибка обработки времени обновления: {str(e)}")

            # Определяем параметры парсинга
            if total_records == 0:
                logging.info("Обнаружена пустая БД. Запуск полного парсинга...")
                dates_to_parse = [today, yesterday]
                total_pages_today = get_total_pages(driver, BASE_URL.format(date=today, page=1))
                total_pages_yesterday = get_total_pages(driver, BASE_URL.format(date=yesterday, page=1))
            else:
                if last_update_date == today:
                    logging.info("Последнее обновление было сегодня. Парсим только сегодняшние данные.")
                    dates_to_parse = [today]
                    total_pages_today = min(get_total_pages(driver, BASE_URL.format(date=today, page=1)), pages_to_parse)
                    total_pages_yesterday = 0
                else:
                    logging.info("Последнее обновление было вчера или раньше. Парсим сегодняшние и вчерашние данные.")
                    dates_to_parse = [today, yesterday]
                    total_pages_today = min(get_total_pages(driver, BASE_URL.format(date=today, page=1)), pages_to_parse)
                    total_pages_yesterday = min(get_total_pages(driver, BASE_URL.format(date=yesterday, page=1)), pages_to_parse)

            # Парсинг данных
            for parse_date in dates_to_parse:
                logging.info(f"Обработка данных за {parse_date}...")
                data_to_insert = []
                
                total_pages = total_pages_today if parse_date == today else total_pages_yesterday
                logging.info(f'Дата: {parse_date}, Всего страниц: {total_pages}')

                for page in range(1, total_pages + 1):
                    url = BASE_URL.format(date=parse_date, page=page)
                    try:
                        draw_data = parse_page(driver, url)
                        if not draw_data:
                            continue
                            
                        for draw_date, draw_number, combination, field in draw_data:
                            # Используем текущее соединение для проверки
                            cursor.execute("SELECT 1 FROM results WHERE draw_number = ?", (draw_number,))
                            if not cursor.fetchone():
                                data_to_insert.append((draw_date, draw_number, combination, field))
                    except Exception as e:
                        logging.error(f"Ошибка при парсинге страницы {page} за {parse_date}: {str(e)}", exc_info=True)
                        continue

                # Сохранение данных
                if data_to_insert:
                    try:
                        # Используем текущее соединение для вставки
                        cursor.executemany(
                            "INSERT INTO results (draw_date, draw_number, combination, field) VALUES (?, ?, ?, ?)",
                            data_to_insert
                        )
                        db.connection.commit()
                        logging.info(f'Успешно сохранено {len(data_to_insert)} записей за {parse_date}')
                        
                        # Обновляем время последнего обновления
                        if data_to_insert:
                            last_draw_number = data_to_insert[-1][1]
                            db.update_last_update_time(last_draw_number)
                    except Exception as e:
                        db.connection.rollback()
                        logging.error(f"Ошибка сохранения данных за {parse_date}: {str(e)}", exc_info=True)

    except Exception as e:
        logging.error(f"Критическая ошибка в parse_data: {str(e)}", exc_info=True)
        raise
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception as e:
                logging.warning(f"Ошибка при закрытии драйвера: {str(e)}")

def is_draw_exists(draw_number):
    with DatabaseManager() as db:
        cursor = db.connection.cursor()
        cursor.execute("SELECT 1 FROM results WHERE draw_number = ?", (draw_number,))
        return cursor.fetchone() is not None   