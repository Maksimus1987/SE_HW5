# Import model

from fastapi import FastAPI  # import `FastAPI` from `fastapi`
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)  # import class `AutoTokenizer` и `AutoModelForSequenceClassification`
# from `transformers`
import torch  # import `torch`

# make `/predict` in app FastAPI

app = FastAPI()  # create object 'app' of class FastAPI

model = AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny2-cedr-emotion-detection"
)  # загружаем предобученную модель для классификации последовательностей
# из пакета "cointegrated/rubert-tiny2-cedr-emotion-detection"
# с помощью метода `from_pretrained`
# класса `AutoModelForSequenceClassification`
# и присваивает ее переменной `model`
tokenizer = AutoTokenizer.from_pretrained(
    "cointegrated/rubert-tiny2-cedr-emotion-detection"
)  # загружаем токенизатор для модели из того же пакета
# с помощью метода `from_pretrained класса `AutoTokenizer`
# и присваивает его переменной `tokenizer`


@app.post(
    "/predict"
)  # определяем декоратор `@app.post`, который указывает, что следующая
# функция будет обрабатывать POST-запросы на маршрут "/predict"
def predict(
    text: str,
):  # объявляем функцию `predict`, которая принимае
    # один аргумент `text` типа `str`
    inputs = tokenizer(
        text, return_tensors="pt"
    )  # используем токенизатор `tokenizer` для токенизации текста `text`
    # с помощью метода `tokenizer` и преобразуем его в тензор PyTorch
    outputs = model(
        **inputs
    )  # строка передает входной тензор `inputs` в модель `model`
    # и получает выходной тензор `outputs`
    probabilities = torch.nn.functional.softmax(
        outputs.logits, dim=-1
    )  # строка применяет функцию softmax к выходным данным модели,
    # чтобы получить вероятности эмоций.
    # Результат присваивается переменной `probabilities`
    return (
        probabilities.tolist()
    )  # преобразуем список вероятностей в простой список Python
    # и возвращаем его в качестве ответа на запрос
