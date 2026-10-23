"""Nạp artifact chính thức và suy luận trên CPU, không huấn luyện lại."""

import importlib.util
import json
import math
from pathlib import Path
import uuid


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
LABEL_NAMES = ("Neutral", "Positive", "Negative", "Toxic")
LABEL_VI = ("Trung tính", "Tích cực", "Tiêu cực", "Độc hại")
_REQUIRED = ("model.keras", "preprocessing.py", "label_map.json", "config.json")


class ArtifactError(RuntimeError):
    """Artifact thiếu, không tương thích hoặc không hợp lệ."""


def missing_artifacts(artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Liệt kê các tệp bắt buộc chưa có."""
    root = Path(artifact_dir)
    return [name for name in _REQUIRED if not (root / name).is_file()]


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_environment(root):
    try:
        value = _read_json(root / "environment.json")
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _check_vectorizers(model, max_length):
    """Đối chiếu độ dài nếu model công bố cấu hình TextVectorization."""
    for layer in getattr(model, "layers", ()):
        if type(layer).__name__ == "TextVectorization":
            length = layer.get_config().get("output_sequence_length")
            if length is not None and length != max_length:
                raise ValueError("MAX_LENGTH không khớp TextVectorization trong model.")
        if layer is not model and hasattr(layer, "layers"):
            _check_vectorizers(layer, max_length)


def load_artifacts(artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Kiểm tra tệp trước khi nạp TensorFlow và model trên CPU."""
    root = Path(artifact_dir).resolve()
    try:
        missing = missing_artifacts(root)
        if missing:
            raise FileNotFoundError("Thiếu artifact: " + ", ".join(missing))
        config = _read_json(root / "config.json")
        if not isinstance(config, dict):
            raise ValueError("config.json phải là một đối tượng JSON.")
        max_length = config.get("MAX_LENGTH")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("MAX_LENGTH phải là số nguyên dương.")
        label_map = _read_json(root / "label_map.json")
        if not isinstance(label_map, dict) or set(label_map) != {"0", "1", "2", "3"}:
            raise ValueError("label_map.json phải chứa đúng bốn ID 0, 1, 2, 3.")
        for index, (name, vietnamese) in enumerate(zip(LABEL_NAMES, LABEL_VI)):
            item = label_map[str(index)]
            if not isinstance(item, dict) or item.get("name") != name or item.get("vietnamese") != vietnamese:
                raise ValueError(f"Nhãn ID {index} không khớp thứ tự hoặc tên của notebook.")
        spec = importlib.util.spec_from_file_location(
            "_neu_artifact_preprocessing_" + uuid.uuid4().hex, root / "preprocessing.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Không thể nạp preprocessing.py của artifact.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        normalize_text = getattr(module, "normalize_text", None)
        if not callable(normalize_text):
            raise ValueError("preprocessing.py thiếu hàm normalize_text.")

        import tensorflow as tf

        # Context CPU áp dụng cả khi tiến trình đã khởi tạo TensorFlow ở nơi khác.
        with tf.device("/CPU:0"):
            model = tf.keras.models.load_model(root / "model.keras", compile=False)
        _check_vectorizers(model, max_length)
        return {
            "model": model,
            "normalize_text": normalize_text,
            "config": config,
            "label_map": label_map,
            "max_length": max_length,
            "artifact_dir": root,
            "environment": _optional_environment(root),
        }
    except Exception as exc:
        if "charmap" in str(exc) or "codec can't decode" in str(exc):
            raise ArtifactError(
                "Windows đang đọc từ vựng sai bảng mã. Chạy run_demo.cmd hoặc "
                "python -X utf8 -m streamlit run app.py để bật UTF-8."
            ) from exc
        raise ArtifactError(f"Không thể nạp artifact tại {root}: {exc}") from exc


def predict_text(text, artifacts):
    """Chuẩn hóa bằng hàm đi kèm artifact rồi trả xác suất đúng thứ tự nhãn."""
    if not isinstance(text, str):
        raise TypeError("Văn bản phải là chuỗi.")
    if not text.strip():
        raise ValueError("Hãy nhập một bình luận không rỗng.")
    normalized = artifacts["normalize_text"](text)
    if not isinstance(normalized, str) or not normalized.strip():
        raise ValueError("Văn bản sau chuẩn hóa phải là chuỗi không rỗng.")
    try:
        import tensorflow as tf
        import numpy as np

        with tf.device("/CPU:0"):
            output = artifacts["model"](tf.constant([normalized]), training=False)
        scores_array = np.asarray(output)
        if scores_array.shape != (1, 4):
            raise ValueError("Đầu ra model phải có shape (1, 4).")
        if scores_array.dtype.kind not in "fiu":
            raise ValueError("Đầu ra model phải chứa các xác suất dạng số.")
        scores = [float(value) for value in scores_array[0]]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Xác suất phải hữu hạn và thuộc khoảng [0, 1].")
        if not math.isclose(sum(scores), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Tổng xác suất đầu ra không gần 1.")
    except Exception as exc:
        raise ArtifactError(f"Không thể suy luận bằng model: {exc}") from exc
    label_id = max(range(4), key=scores.__getitem__)
    token_count = len(normalized.split())
    max_length = artifacts["max_length"]
    return {
        "text": text,
        "normalized_text": normalized,
        "label_id": label_id,
        "label": LABEL_VI[label_id],
        "scores": scores,
        "token_count": token_count,
        "max_length": max_length,
        "truncated": token_count > max_length,
    }


def load_metrics(artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Chỉ trả số liệu test hợp lệ; thiếu số liệu không cản suy luận."""
    try:
        metrics = _read_json(Path(artifact_dir) / "metrics.json")
        test = metrics["test"]
        for key in ("accuracy", "macro_f1", "f1_toxic"):
            value = test[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            if not math.isfinite(value) or not 0 <= value <= 1:
                return None
        return metrics
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return None
