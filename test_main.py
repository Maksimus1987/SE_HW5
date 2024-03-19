# Developer #1 @aleksrf1 aleksrf@gmail.com - test for api
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict():
    # Preparing test data
    text = "Я очень удивлен происходящим!"

    # Execute the request
    response = client.post("/predict?text=" + text, json={"text": text})

    # Checking response status
    assert response.status_code == 200

    predictions = response.json()

    # Checking response content type
    assert response.headers["content-type"] == "application/json"

    # Checking the correctness of the answer
    assert isinstance(predictions, list)
    if predictions:  # Checking for elements in the predictions list
        assert isinstance(predictions[0], list)
    for prob in predictions[0]:
        assert isinstance(prob, float)
