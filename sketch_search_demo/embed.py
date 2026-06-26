"""
Этап 2 — CLIP-эмбеддинги галереи (и общий «движок» демо).

Берёт фотографии из data/photos (манифест artifacts/gallery.json от collect.py),
кодирует каждую моделью CLIP в вектор и сохраняет artifacts/embeddings.npy.
Здесь же — функции для дашборда: загрузка модели, подготовка рисунка, эмбеддинг
и поиск ближайших. Весь «смысловой» код в одном месте.

CLIP кодирует И фотографии, И рисунок пользователя в ОДНО пространство — поэтому
рисунок можно искать среди фото по косинусной близости (как в demo №1 текст
искали среди игр).

Запуск:  python embed.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

# --- Пути и модель (общие для collect.py / embed.py / app.py) -------------- #
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
PHOTO_DIR = DATA_DIR / "photos"          # photos/<категория>/<id>.jpg
ART = BASE / "artifacts"
GALLERY_PATH = ART / "gallery.json"      # пишет collect.py
EMB_PATH = ART / "embeddings.npy"        # (N, 512), L2-нормированные
META_PATH = ART / "meta.json"

CLIP_MODEL = "clip-ViT-B-32"             # лёгкий CLIP через sentence-transformers
EMBED_BATCH = 64

_model = None


def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    """Грузим CLIP один раз (скачивается с HuggingFace при первом запуске)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(CLIP_MODEL, device=get_device())
    return _model


# --- Подготовка рисунка ----------------------------------------------------- #
def preprocess_sketch(img: Image.Image, margin: float = 0.10, out: int = 320) -> Image.Image:
    """RGBA-холст → чёрные штрихи на белом, по центру, квадрат, сглаженный ресайз.

    Холст отдаёт прозрачный фон: при наивном .convert('RGB') «не нарисовано»
    стало бы чёрным → мусор. Поэтому сначала кладём на белый фон.
    """
    if img.mode in ("RGBA", "LA"):
        img = Image.alpha_composite(
            Image.new("RGBA", img.size, (255, 255, 255, 255)), img.convert("RGBA"))
    img = img.convert("RGB")

    # обрезаем по рамке нарисованного (разница с белым)
    gray = img.convert("L")
    bbox = ImageChops.difference(gray, Image.new("L", gray.size, 255)).getbbox()
    if bbox:
        img = img.crop(bbox)

    # поля + квадрат по центру (центр-кроп CLIP ничего не срежет)
    w, h = img.size
    side = max(w, h) + 2 * int(max(w, h) * margin)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((out, out), Image.LANCZOS)   # сглаживание: тонкие линии выживут


def sketch_is_empty(image_data, min_ink_px: int = 25) -> bool:
    """Пусто, если тёмных пикселей почти нет (фон холста белый → по альфе не понять)."""
    if image_data is None:
        return True
    arr = np.asarray(image_data)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return True
    ink = arr[..., :3].astype(np.int16).min(axis=-1) < 200     # темнее почти-белого
    if arr.shape[2] == 4:
        ink &= arr[..., 3] > 10                                # и реально нарисовано
    return int(ink.sum()) < min_ink_px


# --- Кодирование и поиск ---------------------------------------------------- #
def embed_images(images: list[Image.Image]) -> np.ndarray:
    """Фотографии → L2-нормированные CLIP-векторы (кодируем как есть)."""
    return load_model().encode(images, batch_size=EMBED_BATCH, convert_to_numpy=True,
                               normalize_embeddings=True).astype(np.float32)


def embed_sketch(img: Image.Image) -> np.ndarray:
    """Рисунок → препроцесс → CLIP-вектор (L2-нормированный)."""
    return load_model().encode(preprocess_sketch(img), convert_to_numpy=True,
                               normalize_embeddings=True).astype(np.float32)


def nearest(query: np.ndarray, matrix: np.ndarray, k: int):
    """Топ-k по косинусной близости (векторы нормированы → это скалярное произв.)."""
    sims = matrix @ query
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def main() -> None:
    photos = json.loads(GALLERY_PATH.read_text(encoding="utf-8"))["photos"]
    print(f"Кодирую {len(photos)} фото моделью {CLIP_MODEL} (устройство: {get_device()})…")
    vecs = []
    for i in range(0, len(photos), EMBED_BATCH):
        imgs = [Image.open(PHOTO_DIR / p["file"]).convert("RGB")
                for p in photos[i:i + EMBED_BATCH]]
        vecs.append(embed_images(imgs))
        print(f"   …{min(i + EMBED_BATCH, len(photos))}/{len(photos)}", end="\r", flush=True)
    matrix = np.vstack(vecs)
    np.save(EMB_PATH, matrix)

    n_cat = len({p["category"] for p in photos})
    META_PATH.write_text(json.dumps({
        "model": CLIP_MODEL, "dims": int(matrix.shape[1]), "n": len(photos),
        "n_categories": n_cat, "device": get_device(),
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Готово: {matrix.shape} · {n_cat} категорий → artifacts/ · "
          "дальше: streamlit run app.py")


if __name__ == "__main__":
    main()
