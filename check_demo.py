"""Kiểm tra pipeline thật và các thao tác chính, không huấn luyện model."""

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from inference import (
    DEFAULT_ARTIFACT_DIR, LABEL_NAMES, ArtifactError, load_artifacts,
    load_metrics, missing_artifacts, predict_text,
)

ROOT = Path(__file__).resolve().parent


def check_missing_files():
    with tempfile.TemporaryDirectory() as directory:
        missing = missing_artifacts(Path(directory))
        assert set(missing) == {"model.keras", "preprocessing.py", "label_map.json", "config.json"}
        assert load_metrics(Path(directory)) is None
        try:
            load_artifacts(Path(directory))
        except ArtifactError:
            pass
        else:
            raise AssertionError("Thiếu artifact nhưng hàm không báo lỗi")
    print("PASS: thiếu artifact/metric được xử lý")


def load_reference(directory):
    # predict.py gốc import preprocessing theo tên; tạm ưu tiên đúng thư mục ZIP.
    previous = sys.modules.pop("preprocessing", None)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("notebook_predict", directory / "predict.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.modules.pop("preprocessing", None)
        if previous is not None:
            sys.modules["preprocessing"] = previous


def check_real_inference(directory):
    started = time.perf_counter()
    artifacts = load_artifacts(directory)
    load_seconds = time.perf_counter() - started
    reference = load_reference(directory)
    normalize = artifacts["normalize_text"]
    assert normalize("  KHÔNG\t tốt 😊  ") == "không tốt 😊"
    assert normalize("to\u0302i") == "tôi"
    for empty in ("", " \n\t "):
        # Nếu gọi nhầm model, sentinel này làm kiểm tra thất bại ngay.
        def forbidden_call(*args, **kwargs):
            raise AssertionError("Không được gọi model với câu rỗng")
        try:
            predict_text(empty, {**artifacts, "model": forbidden_call})
        except ValueError:
            pass
        else:
            raise AssertionError("Phải từ chối câu rỗng")

    texts = [
        "Giảng viên giải thích rất dễ hiểu, mình học được nhiều điều.",
        "Mình không hài lòng với cách xử lý này.",
        "Lịch học được thông báo vào thứ hai.",
        "Bạn nói chuyện thật vô văn hóa!",
        "  KHÔNG   tốt 😊\nnhưng mình vẫn muốn thử. ",
        "To\u0302i không thích cách giải thích này.",
        " ".join(["tốt"] * artifacts["max_length"]),
        " ".join(["không"] * (artifacts["max_length"] + 1)),
    ]
    timings = []
    max_difference = 0.0
    for text in texts:
        started = time.perf_counter()
        actual = predict_text(text, artifacts)
        timings.append(time.perf_counter() - started)
        expected = reference.predict_text(text)
        scores = np.asarray(actual["scores"])
        reference_scores = np.asarray([expected["probabilities"][name] for name in LABEL_NAMES])
        np.testing.assert_allclose(scores, reference_scores, rtol=1e-5, atol=1e-6)
        assert scores.shape == (4,) and np.isfinite(scores).all()
        assert abs(scores.sum() - 1) < 1e-5
        assert actual["label"] == expected["label"]
        assert actual["truncated"] == expected["truncated"]
        assert actual["truncated"] == (len(normalize(text).split()) > artifacts["max_length"])
        max_difference = max(max_difference, float(np.max(np.abs(scores - reference_scores))))
        print(f"PASS: {actual['label']}; tokens={actual['token_count']}; truncated={actual['truncated']}")
    print(json.dumps({"model_load_seconds": load_seconds, "prediction_seconds": timings,
                      "max_score_difference": max_difference}, indent=2))
    assert load_metrics(directory) is not None
    return artifacts


def check_ui(artifacts=None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.text_area and at.button
    at.button[0].click().run()
    assert not at.exception
    assert at.warning or at.info or at.error
    if artifacts is not None:
        sentence = "Mình không hài lòng với cách xử lý này."
        at.text_area[0].input(sentence).run()
        at.button[0].click().run()
        assert not at.exception, at.exception
        expected = predict_text(sentence, artifacts)
        # So sánh điểm trong session với pipeline độc lập để kiểm tra tích hợp UI.
        np.testing.assert_allclose(at.session_state["result"]["scores"], expected["scores"], rtol=1e-5, atol=1e-6)
        at.text_area[0].input("Câu đã sửa, chưa bấm phân tích").run()
        assert not at.exception
        assert "result" not in at.session_state, "Kết quả cũ phải bị xóa khi thay câu"
        if at.selectbox:
            at.selectbox[0].select_index(1).run()
            assert at.text_area[0].value != "Câu đã sửa, chưa bấm phân tích"
        at.text_area[0].input(" ".join(["không"] * (artifacts["max_length"] + 1))).run()
        with patch("inference.load_artifacts", side_effect=AssertionError("Model bị nạp lại thay vì dùng cache")):
            at.button[0].click().run()
        assert not at.exception
        assert any(str(artifacts["max_length"]) in item.value for item in at.warning), "Thiếu cảnh báo cắt token"
    # Kiểm tra trạng thái thiếu file bằng thay thế hàm kiểm kê, không sửa artifact.
    with patch("inference.missing_artifacts", return_value=["model.keras"]):
        missing = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not missing.exception
        messages = [item.value for item in [*missing.warning, *missing.info, *missing.error]]
        assert any("model.keras" in message for message in messages)
    print("PASS: Streamlit nhập câu, dự đoán thật, chọn mẫu, xóa kết quả cũ và báo thiếu file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-model", action="store_true")
    args = parser.parse_args()
    check_missing_files()
    missing = missing_artifacts()
    if not (DEFAULT_ARTIFACT_DIR / "predict.py").is_file():
        missing.append("predict.py")
    artifacts = None
    if missing:
        if args.require_model:
            raise SystemExit("FAIL: thiếu " + ", ".join(missing))
        print("SKIP: chưa đối chiếu model thật, thiếu " + ", ".join(missing))
    else:
        artifacts = check_real_inference(DEFAULT_ARTIFACT_DIR)
    check_ui(artifacts)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
