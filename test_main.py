# Importing necessary modules
from fastapi.testclient import TestClient
from main import app

# Initializing the test client
client = TestClient(app)

# Test case for endpoint /predict
def test_predict():
    # Prepare test data
    text = "Я очень удивлен происходящим!"

    # Make the request
    response = client.post("/predict?text=" + text, json={"text": text})

    # Check response status code
    assert response.status_code == 200

    # Validate content type of response
    assert response.headers["content-type"] == "application/json"

    # Check the correctness of the predictions
    predictions = response.json()
    assert isinstance(predictions, list)

    if predictions:
        assert isinstance(predictions[0], list)
        
        for prob in predictions[0]:
            assert isinstance(prob, float)
