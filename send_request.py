import requests

# Отправка POST-запроса к веб-серверу с предсказанием для заданного текста
# и вывод ответа в формате JSON

# Задаем текст для анализа
text = "Я очень удивлен происходящим!"

# Формируем URL для запроса
url = f"http://127.0.0.1:8000/predict?text={text}"

# Создаем словарь с текстом для передачи в виде JSON
payload = {"text": text}

try:
    # Отправляем POST-запрос
    response = requests.post(url, json=payload)

    # Проверяем статус код ответа
    if response.status_code == 200:
        # Выводим ответ в формате JSON
        print(response.json())
    else:
        print(f"Ошибка при запросе. Код статуса: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Ошибка при отправке запроса: {e}")
