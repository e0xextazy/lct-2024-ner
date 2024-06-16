from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)


def convert_from_onnx_to_quantized_onnx(
        onnx_model_path: str,
        quantized_onnx_model_path: str
) -> None:
    """Квантизация модели в формате ONNX до Int8
    и сохранение кванитизированной модели на диск.

    @param onnx_model_path: путь к модели в формате ONNX
    @param quantized_onnx_model_path: путь к квантизированной модели
    """
    quantize_dynamic(
        onnx_model_path,
        quantized_onnx_model_path,
        weight_type=QuantType.QInt4
    )


convert_from_onnx_to_quantized_onnx(
    '/home/pc/Desktop/lct-2024-ner/ml/convert_models/onnx_model.onnx',
    '/home/pc/Desktop/lct-2024-ner/ml/convert_models/quantized_onnx_model_int8.onnx')
