# parsing.py

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
from config import CHROMEDRIVER_PATH, BASE_URL, LOGGING_CONFIG  
from database import save_to_database_batch, update_last_update_time, get_last_update_time, safe_fromisoformat
import logging

# Настройка логгера
logging.basicConfig(**LOGGING_CONFIG)

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
    Если страницы не загружены или структура изменилась, возвращает 1.
    """
    try:
        driver.get(url)
        # Ожидаем загрузки элементов пагинации
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'li.pagination-item a.pagination-link span.pagination-link__text'))
        )
        # Находим все элементы пагинации
        page_elements = driver.find_elements(By.CSS_SELECTOR, 'li.pagination-item a.pagination-link span.pagination-link__text')
        # Извлекаем номера страниц
        pages = [int(element.text.strip()) for element in page_elements if element.text.strip().isdigit()]
        # Возвращаем максимальный номер страницы
        return max(pages) if pages else 1
    except Exception as e:
        # Логируем ошибку с подробной информацией
        logging.error(f'Ошибка при определении количества страниц: {e}')
        logging.error(f'URL: {url}')
        logging.error(f'Страница не загружена или структура изменилась.')
        return 1  # Возвращаем 1, если не удалось определить количество страниц

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

# parsing.py

def parse_data():
    driver = setup_driver()
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Получаем время последнего обновления
    last_update_time = get_last_update_time()
    if last_update_time:
        last_update_time = safe_fromisoformat(last_update_time)  # Используем safe_fromisoformat
        time_since_last_update = datetime.now() - last_update_time
    else:
        time_since_last_update = timedelta(hours=2)  # Если время обновления отсутствует, парсим все данные

    # Если с последнего обновления прошло меньше 1 часа 35 минут, парсим только первую страницу сегодняшней даты
    if time_since_last_update < timedelta(hours=1, minutes=35):
        logging.info("Последнее обновление было менее 1 часа 35 минут назад. Парсим только первую страницу сегодняшней даты.")
        dates_to_parse = [today]
        total_pages_today = 1  # Ограничиваем парсинг одной страницей для сегодня
        total_pages_yesterday = 0  # Не парсим вчерашние данные
    else:
        logging.info("Последнее обновление было более 1 часа 35 минут назад. Парсим все данные.")
        dates_to_parse = [today, yesterday]
        # Определяем количество страниц отдельно для сегодня и вчера
        url_today = BASE_URL.format(date=today, page=1)
        total_pages_today = get_total_pages(driver, url_today)
        url_yesterday = BASE_URL.format(date=yesterday, page=1)
        total_pages_yesterday = get_total_pages(driver, url_yesterday)

    for parse_date in dates_to_parse:
        logging.info(f"Обработка данных за {parse_date}...")
        data_to_insert = []
        url = BASE_URL.format(date=parse_date, page=1)

        # Определяем количество страниц для текущей даты
        if parse_date == today:
            total_pages = total_pages_today
        else:
            total_pages = total_pages_yesterday

        logging.info(f'Дата: {parse_date}, Всего страниц: {total_pages}')

        # Парсим данные только с первой страницы, если total_pages = 1
        for page in range(1, total_pages + 1):
            url = BASE_URL.format(date=parse_date, page=page)
            draw_data = parse_page(driver, url)

            for draw_date, draw_number, combination, field in draw_data:
                if not any(d[1] == draw_number and d[2] == combination and d[3] == field for d in data_to_insert):
                    data_to_insert.append((draw_date, draw_number, combination, field))

        if data_to_insert:
            save_to_database_batch(data_to_insert)
            logging.info(f'Сохранение завершено для {parse_date}.')
        else:
            logging.info(f"Нет новых данных для сохранения за {parse_date}.")

    # Обновляем время последнего обновления
    update_last_update_time()
    driver.quit()