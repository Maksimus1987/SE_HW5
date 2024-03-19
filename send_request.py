# Developer 3: Maksimus1987, send a POST request to the web server,
# using URL and text to get a prediction for a given text.
# Then print the response in JSON format

import requests  # import the `requests` module,
# which provides functions for sending HTTP requests

text = "Я очень удивлен происходящим!"  # create a `text` variable
# assigns it the value `"I'm very surprised by what's happening!"`
url = "http://127.0.0.1:8000/predict?text=" + text  # create a variable `url
# and assign it the value `"http://localhost:8080/predict"`

payload = {"text": text}  # create a dictionary `payload` with the key `"text"
# with the value `text`. The `payload` dictionary will be converted to JSON format
# and passed as the request body
response = requests.post(url, json=payload)  # line sends a POST request
# to the specified URL `url` using the `requests` module
print(response.json())
