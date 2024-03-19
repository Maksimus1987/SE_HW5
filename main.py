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
    )
# loading model for classification sequences
# from package "cointegrated/rubert-tiny2-cedr-emotion-detection"
# with use method `from_pretrained`
# class `AutoModelForSequenceClassification`
# and assign to a variable `model`
tokenizer = AutoTokenizer.from_pretrained(
    "cointegrated/rubert-tiny2-cedr-emotion-detection"
)  # load the tokenizer for the model from the same package
# with use `from_pretrained class `AutoTokenizer`
# and assign to a variable `tokenizer`


@app.post(
    "/predict"
)  # define a decorator `@app.post`, which indicates that the next
# the function will process POST-requests to the route "/predict"
def predict(
    text: str,
):  # declare a function `predict`, which accepts
    # one argument `text` type of `str`
    inputs = tokenizer(
        text, return_tensors="pt"
    )  # use `tokenizer` for toking text `text`
    # with use method `tokenizer` and transform it into a tensor PyTorch
    outputs = model(
        **inputs
    )  # the string passes the input tensor `inputs` in model `model`
    # and gets the output tensor `outputs`
    probabilities = torch.nn.functional.softmax(
        outputs.logits, dim=-1
    )  # line applies function "softmax" to model output,
    # to get the probabilities of emotions.
    # The result is assigned to a variable `probabilities`
    return (
        probabilities.tolist()
    )  # convert the list of probabilities into a simple list Python
    # and return it as a response to the request
