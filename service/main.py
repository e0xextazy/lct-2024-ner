from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from service.model import get_labels

app = FastAPI()


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
          summary="Get model predictions",
          description="This endpoint takes a text input and returns a list of predicted labels "
                      "for each token in the text. "
                      "The labels can be 'O', 'B-discount', 'B-value', or 'I-value'.")
async def predict(request: TextRequest):
    try:
        text = request.text
        word_count = len(text.split())
        result = get_labels(text)
        return result[:word_count]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/",
         summary="Get UI",
         description="")
async def get_ui():
    with open("service/index.html", "r") as file:
        content = file.read()
    return HTMLResponse(content=content)
