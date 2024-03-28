from fastapi import FastAPI
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch

app = FastAPI()

model = AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny2-cedr-emotion-detection"
    )
tokenizer = AutoTokenizer.from_pretrained(
    "cointegrated/rubert-tiny2-cedr-emotion-detection"
    )


@app.post(
    "/predict"
    )
def predict(text: str):
    if not text:
        return {"error": "No text provided"}  # Валидация наличия текста
    try:
        inputs = tokenizer(
            text, return_tensors="pt"
        )
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities.tolist()
    except Exception as e:
        return {"error": str(e)}  # Обработка ошибок


@app.get(
    "/"
    )  # Добавлен новый маршрут для базовой страницы
def read_root():
    return {"message": "Welcome to the emotion detection API"}
