import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


def convert_from_torch_to_onnx(
        onnx_path: str,
        tokenizer: AutoTokenizer,
        model: AutoModelForTokenClassification
) -> None:
    """Конвертация модели из формата PyTorch в формат ONNX.

    @param onnx_path: путь к модели в формате ONNX
    @param tokenizer: токенизатор
    @param model: модель
    """
    dummy_model_input = tokenizer(
        "текст для конвертации",
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to("cpu")
    torch.onnx.export(
        model,
        dummy_model_input["input_ids"],
        onnx_path,
        opset_version=14,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
                1: "sequence_len"
            },
            "output": {
                0: "batch_size"
            }
        }
    )


model = AutoModelForTokenClassification.from_pretrained(
    '/home/pc/Desktop/lct-2024-ner/ml/baseline_ai-forever-sbert_large_mt_nlu_ru_v2/final',
    # "LaBSE",
    local_files_only=True,
)

model.eval()


tokenizer = AutoTokenizer.from_pretrained(
    '/home/pc/Desktop/lct-2024-ner/ml/baseline_ai-forever-sbert_large_mt_nlu_ru_v2/final',
    # "LaBSE",
    local_files_only=True
)


convert_from_torch_to_onnx(
    'onnx_model.onnx',
    tokenizer,
    model
)
