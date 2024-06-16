import onnxruntime
import numpy as np
from transformers import AutoTokenizer
from onnxruntime import (
    InferenceSession,
    SessionOptions
)


def create_onnx_session(
        model_path: str,
        provider: str = "CPUExecutionProvider"
) -> InferenceSession:
    """Создание сессии для инференса модели с помощью ONNX Runtime.

    @param model_path: путь к модели в формате ONNX
    @param provider: инференс на ЦП
    @return: ONNX Runtime-сессия
    """
    options = SessionOptions()
    options.graph_optimization_level = \
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session


def onnx_inference(
        text: str,
        session: InferenceSession,
        tokenizer: AutoTokenizer,
        max_length: int
) -> np.ndarray:
    """Инференс модели с помощью ONNX Runtime.

    @param text: входной текст для классификации
    @param session: ONNX Runtime-сессия
    @param tokenizer: токенизатор
    @param max_length: максимальная длина последовательности в токенах
    @return: логиты на выходе из модели
    """
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    input_feed = {
        "input_ids": inputs["input_ids"].astype(np.int64)
    }
    outputs = session.run(
        output_names=["output"],
        input_feed=input_feed
    )[0][0]
    return outputs.argmax(1).tolist()


session = create_onnx_session('model/quantized_onnx_labse_model.onnx')
tokenizer = AutoTokenizer.from_pretrained('tokenizer/LaBSE', local_files_only=True)


def get_labels(text: str):
    """Получение меток для входного текста с использованием ONNX модели.

   @param text: входной текст для классификации
   @return: список меток, предсказанных моделью
   """
    return onnx_inference(text, session, tokenizer, max_length=512)

