from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from service.model import get_labels

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str = Field(..., title="Input Text",
                      description="The text to be processed by the ML model")


class PredictionType(Enum):
    O = 0
    B_discount = 1
    B_value = 2
    I_value = 3


@app.post("/predict",
          response_model=list[PredictionType],
          summary="Get model predictions")
async def predict(request: TextRequest):
    """
    Endpoint to get model predictions.

    This endpoint takes a text input and returns a list of predicted labels
    for each token in the text.
    The labels can be 'O', 'B-discount', 'B-value', or 'I-value'.

    :param request: Request containing text for classification

    :return: List of predicted labels
    """
    try:
        text = request.text
        return get_labels(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse, summary="Get UI")
async def get_ui():
    """
    Endpoint to open the index.html file.

    This endpoint returns the content of the HTML file located in the `static` folder.
    It is used to display the main page of your web application.
    Open this endpoint in a browser to view the main page.

    :return: HTML content of the `index.html` file.
    """
    with open("service/index.html", "r", encoding='utf-8') as file:
        content = file.read()
    return HTMLResponse(content=content)
