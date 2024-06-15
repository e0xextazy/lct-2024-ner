from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch


app = FastAPI()


class PredictionType(Enum):
    # O = 0
    # B_discount = 1
    # B_value = 2
    # I_value = 3
    O = "O"
    B_discount = "B-discount"
    B_value = "B-value"
    I_value = "I-value"


def mock_ml_model(text: str) -> list[PredictionType]:
    return [PredictionType.O, PredictionType.B_discount]
    # return ["O", "B-discount"]  # тоже пройдет


class TextRequest(BaseModel):
    text: str = Field(..., title="Input Text", description="The text to be processed by the ML model")


@app.post("/predict",
          response_model=list[PredictionType],
          summary="Get model predictions",
          description="This endpoint takes a text input and returns a list of predicted labels "
                      "for each token in the text. "
                      "The labels can be 'O', 'B-discount', 'B-value', or 'I-value'.")
async def predict(request: TextRequest):
    try:
        text = request.text
        result = mock_ml_model(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
