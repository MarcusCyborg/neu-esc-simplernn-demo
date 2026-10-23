
"""Suy luận bằng artifact đã tải về; không cần đọc lại dữ liệu train."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from preprocessing import normalize_text

ROOT = Path(__file__).resolve().parent
model = tf.keras.models.load_model(ROOT / "model.keras", compile=False)
label_map = json.loads((ROOT / "label_map.json").read_text(encoding="utf-8"))
config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def predict_text(text):
    clean_text = normalize_text(text)
    if not clean_text:
        raise ValueError("Hãy nhập một bình luận không rỗng.")
    scores = model(tf.constant([clean_text]), training=False).numpy()[0]
    label = int(np.argmax(scores))
    return {
        "label": label_map[str(label)]["vietnamese"],
        "truncated": len(clean_text.split()) > config["MAX_LENGTH"],
        "probabilities": {label_map[str(i)]["name"]: float(p) for i, p in enumerate(scores)},
    }


if __name__ == "__main__":
    print(predict_text(input("Nhập bình luận: ")))
