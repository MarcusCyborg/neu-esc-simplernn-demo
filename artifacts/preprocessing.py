
import unicodedata


def normalize_text(text):
    """Giữ dấu, emoji và từ phủ định; chỉ chuẩn hóa Unicode, chữ và khoảng trắng."""
    if not isinstance(text, str):
        raise TypeError("Văn bản phải là chuỗi.")
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.lower().split())
