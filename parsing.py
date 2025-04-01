import os
import config
import sqlite3
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.manager import DriverManager
from webdriver_manager.core.driver_cache import DriverCacheManager
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
import logging
from config import DATETIME_FORMAT, DAYS_TO_PARSE, CHROMEDRIVER_PATH, BASE_URL, LOGGING_CONFIG
datetime.now().strftime(DATETIME_FORMAT)
from database import DatabaseManager, save_to_database_batch, safe_fromisoformat
from database import DatabaseManager
# Устанавливаем переменные окружения
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
# def setup_driver():
#     service = Service(CHROMEDRIVER_PATH)
#     options = Options()
#     options.add_argument('--headless')
#     options.add_argument('--disable-gpu')
#     options.add_argument('--no-sandbox')
#     options.add_argument('--log-level=3')
#     options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
#     options.add_experimental_option('excludeSwitches', ['enable-logging'])
#     return webdriver.Chrome(service=service, options=options)
def setup_driver():
    """Настраивает драйвер с сохранением в папку проекта (для webdriver-manager 4.0+)"""
    try:
        # Подготовка папки
        driver_dir = Path("drivers")
        driver_dir.mkdir(exist_ok=True)
        
        if not config.AUTO_UPDATE_DRIVER:
            return webdriver.Chrome(service=ChromeService(
                executable_path=config.CHROMEDRIVER_PATH))
        
        # Настройка кеша для сохранения в папку проекта
        cache_manager = DriverCacheManager(root_dir=str(driver_dir))
        
        # Автоматическое управление версиями
        driver_path = ChromeDriverManager(
            cache_manager=cache_manager,
            driver_version=config.DRIVER_VERSION  # None для автоматического определения
        ).install()
        
        # Очистка старых версий (если нужно)
        clean_old_drivers(driver_dir)
        
        # Создаем сервис и драйвер
        service = ChromeService(executable_path=driver_path)
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        
        return webdriver.Chrome(service=service, options=options)
        
    except Exception as e:
        logging.error(f"Ошибка инициализации драйвера: {str(e)}")
        raise

def get_chrome_version():
    """Получаем версию установленного Chrome (Windows)"""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                          r"Software\Google\Chrome\BLBeacon") as key:
            version = winreg.QueryValueEx(key, "version")[0]
        return version.split('.')[0]  # Возвращаем major версию
    except Exception as e:
        logging.warning(f"Не удалось определить версию Chrome: {str(e)}")
        return None  # Пусть webdriver-manager сам определит версию
        
def clean_old_drivers(driver_dir, keep_last=2):
    """Оставляет только последние 2 версии драйверов"""
    try:
        drivers = sorted(Path(driver_dir).glob("chromedriver*"))
        for old_driver in drivers[:-keep_last]:
            try:
                old_driver.unlink()
            except Exception as e:
                logging.warning(f"Не удалось удалить {old_driver}: {str(e)}")
    except Exception as e:
        logging.warning(f"Ошибка очистки старых драйверов: {str(e)}")
 

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
        # logging.error(f'Ошибка при определении количества страниц: {e}')
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



import logging
from datetime import datetime, timedelta
from config import DAYS_TO_PARSE  # Импортируем настройку из config.py

import logging
from datetime import datetime, timedelta
from config import DAYS_TO_PARSE  # Импортируем настройку из config.py

def parse_data():
    """Парсит данные лотереи с учетом времени последнего обновления и настроек"""
    driver = None
    try:
        driver = setup_driver()
        current_time = datetime.now()
        today = current_time.date()
        current_hour_minute = current_time.strftime("%H:%M")
        
        with DatabaseManager() as db:
            cursor = db.connection.cursor()
            
            # Проверяем наличие таблицы results
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM sqlite_master 
                    WHERE type='table' AND name='results'
                )
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logging.error("Таблица results не найдена в базе данных. Завершение работы.")
                return
            
            # Получаем время последнего обновления
            cursor.execute("SELECT last_update FROM results ORDER BY last_update DESC LIMIT 1")
            result = cursor.fetchone()
            last_update = datetime.strptime(result[0], DATETIME_FORMAT) if result else None
            
            # Логируем информацию о последнем обновлении
            if last_update:
                logging.info(f"Последнее обновление результатов: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                logging.info("В таблице результатов нет записей (первый запуск или таблица пустая)")
            
            # Определяем даты для парсинга
            if last_update:
                last_update_date = last_update.date()
                last_update_time = last_update.time()
                time_diff = current_time - last_update
                minutes_diff = time_diff.total_seconds() / 60
                
                # Проверяем особый случай: обновление вчера после 23:55 и сейчас до 00:05
                is_special_case = (
                    last_update_date == today - timedelta(days=1) and
                    last_update_time.hour == 23 and last_update_time.minute >= 55 and
                    current_hour_minute < "00:05"
                )
                
                if is_special_case:
                    logging.info("Особый случай: обновление вчера после 23:55, сейчас до 00:05")
                    parse_dates = [last_update_date.strftime('%Y-%m-%d')]
                    parse_pages = {parse_dates[0]: 1}  # Парсим только 1 страницу за вчера
                
                elif last_update_date == today:
                    # Вычисляем количество страниц (min 1, max 15)
                    pages_to_parse = max(1, min(15, int(minutes_diff // 95)))
                    logging.info(f"Разница с последним обновлением: {int(minutes_diff)} минут")
                    logging.info(f"Будем парсить {pages_to_parse} страниц за сегодня")
                    
                    parse_dates = [today.strftime('%Y-%m-%d')]
                    parse_pages = {parse_dates[0]: pages_to_parse}
                    
                elif last_update_date == today - timedelta(days=1):
                    max_date = today - timedelta(days=DAYS_TO_PARSE)
                    parse_dates = []
                    current_date = today
                    
                    while current_date >= max_date and current_date >= last_update_date:
                        parse_dates.append(current_date.strftime('%Y-%m-%d'))
                        current_date -= timedelta(days=1)
                    
                    parse_pages = {date: None for date in parse_dates}
                    logging.info(f"Будем парсить данные за {len(parse_dates)} дней: с {parse_dates[-1]} по {parse_dates[0]}")
                    
                else:
                    max_date = today - timedelta(days=DAYS_TO_PARSE)
                    parse_dates = []
                    current_date = today
                    
                    while current_date >= max_date:
                        parse_dates.append(current_date.strftime('%Y-%m-%d'))
                        current_date -= timedelta(days=1)
                    
                    parse_pages = {date: None for date in parse_dates}
                    days_diff = (today - last_update_date).days
                    logging.info(f"Последнее обновление было {days_diff} дней назад")
                    logging.info(f"Будем парсить данные за {DAYS_TO_PARSE} дней: с {parse_dates[-1]} по {parse_dates[0]}")
            else:
                max_date = today - timedelta(days=DAYS_TO_PARSE)
                parse_dates = []
                current_date = today
                
                while current_date >= max_date:
                    parse_dates.append(current_date.strftime('%Y-%m-%d'))
                    current_date -= timedelta(days=1)
                
                parse_pages = {date: None for date in parse_dates}
                logging.info(f"Будем парсить данные за {DAYS_TO_PARSE} дней: с {parse_dates[-1]} по {parse_dates[0]}")
            
            # Парсинг данных для каждой даты
            for parse_date in parse_dates:
                logging.info(f"Обрабатываем данные за {parse_date}...")
                data_to_insert = []
                
                try:
                    total_available_pages = get_total_pages(driver, BASE_URL.format(date=parse_date, page=1))
                except Exception as e:
                    logging.error(f"Ошибка при получении количества страниц для даты {parse_date}: {str(e)}")
                    continue
                
                pages_limit = parse_pages.get(parse_date)
                if pages_limit is not None:
                    total_pages = min(total_available_pages, pages_limit)
                    logging.info(f"Ограничение парсинга: {pages_limit} страниц из доступных {total_available_pages}")
                else:
                    total_pages = total_available_pages
                    logging.info(f"Будем парсить все доступные страницы ({total_available_pages})")
                
                for page in range(1, total_pages + 1):
                    url = BASE_URL.format(date=parse_date, page=page)
                    try:
                        draw_data = parse_page(driver, url)
                        if not draw_data:
                            continue
                            
                        for draw_date, draw_number, combination, field in draw_data:
                            cursor.execute("SELECT 1 FROM results WHERE draw_number = ?", (draw_number,))
                            if not cursor.fetchone():
                                now = datetime.now().strftime(DATETIME_FORMAT)
                                data_to_insert.append((
                                    draw_date, 
                                    draw_number, 
                                    combination, 
                                    field,
                                    now,
                                    now
                                ))
                    except Exception as e:
                        logging.error(f"Ошибка при парсинге страницы {page} за {parse_date}: {str(e)}", exc_info=True)
                        continue
                
                if data_to_insert:
                    try:
                        cursor.executemany(
                            """INSERT INTO results 
                            (draw_date, draw_number, combination, field, created_at, last_update) 
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            data_to_insert
                        )
                        db.connection.commit()
                        logging.info(f"Добавлено {len(data_to_insert)} новых записей за {parse_date}")
                        
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
