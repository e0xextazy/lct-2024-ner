import onnxruntime
import numpy as np
from transformers import AutoTokenizer
from onnxruntime import (
    InferenceSession,
    SessionOptions
)


def most_frequent(List):
    return max(set(List), key=List.count)


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
) -> np.ndarray:
    """Инференс модели с помощью ONNX Runtime.

    @param text: входной текст для классификации
    @param session: ONNX Runtime-сессия
    @param tokenizer: токенизатор
    @param max_length: максимальная длина последовательности в токенах
    @return: логиты на выходе из модели
    """
    text_len = len(text.split())
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens = tokens[1:-1]
    input_feed = {
        "input_ids": inputs["input_ids"].astype(np.int64)
    }
    outputs = session.run(
        output_names=["output"],
        input_feed=input_feed
    )[0][0]
    outputs = outputs.argmax(1)
    outputs = outputs[1:-1]

    convert_labels = []
    for label, token in zip(outputs, tokens):
        if token.startswith("##"):
            convert_labels[-1].append(label)
        else:
            convert_labels.append([label])

    convert_labels = [int(most_frequent(el)) for el in convert_labels]
    convert_labels = convert_labels + [0] * (text_len - len(convert_labels))

    return convert_labels


session = create_onnx_session('service/quantized_onnx_model_int8.onnx')
tokenizer = AutoTokenizer.from_pretrained(
    'ai-forever/sbert_large_mt_nlu_ru')


def get_labels(text: str):
    """Получение меток для входного текста с использованием ONNX модели.

    @param text: входной текст для классификации
    @return: список меток, предсказанных моделью
    """

    return onnx_inference(text, session, tokenizer)
