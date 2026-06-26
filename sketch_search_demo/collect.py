"""
Этап 1 — собрать фото-галерею из датасета Sketchy.

Качаем один parquet (≈260 МБ) с зеркала JamieSJS/sketchy на HuggingFace
(«corpus» = реальные фотографии объектов) и раскладываем по папкам категорий:
data/photos/<категория>/<id>.jpg + манифест artifacts/gallery.json.

  • 25 000 строк = 12 500 фото × 2 кадрирования → берём первый кадр на объект.
  • id фото = ImageNet-synset (n02121620), имя категории берём через nltk.wordnet.

Запуск:  python collect.py
"""

from __future__ import annotations

import io
import json
import re
import shutil
from collections import defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from embed import ART, DATA_DIR, GALLERY_PATH, PHOTO_DIR

HF_REPO = "JamieSJS/sketchy"
HF_FILE = "corpus-00000-of-00001.parquet"
RAW_DIR = DATA_DIR / "_raw"
UNIQUE_PHOTOS = 12500       # первый блок строк = по одному кадру на объект
ROWS_PER_GROUP = 100

MAX_PHOTOS = None           # None → все 12 500; число → диверсный сэмпл
MAX_PER_CATEGORY = None     # ограничить число фото на категорию
KEEP_RAW = False            # удалить 260-МБ parquet после распаковки


def synset_name(synset: str) -> str:
    """n02121620 → 'cat' (через WordNet); при сбое возвращаем сам synset."""
    from nltk.corpus import wordnet as wn
    try:
        return wn.synset_from_pos_and_offset("n", int(synset[1:])).lemmas()[0].name()
    except Exception:
        return synset


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> None:
    import nltk
    nltk.download("wordnet", quiet=True)

    print(f"⬇️  Скачиваю {HF_REPO}/{HF_FILE} …")
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", local_dir=str(RAW_DIR))
    pf = pq.ParquetFile(path)
    keep = min(pf.metadata.num_row_groups // 2,
               (UNIQUE_PHOTOS + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP)

    if PHOTO_DIR.exists():
        shutil.rmtree(PHOTO_DIR)
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    per_cat: dict[str, int] = defaultdict(int)
    names: dict[str, str] = {}
    manifest: list[dict] = []

    print("🖼️  Распаковываю фотографии…")
    for g in range(keep):
        batch = pf.read_row_group(g, columns=["id", "image"])
        for _id, cell in zip(batch.column("id").to_pylist(), batch.column("image").to_pylist()):
            synset = _id.split("_")[0]
            cat = slug(names.setdefault(synset, synset_name(synset)))
            if MAX_PER_CATEGORY and per_cat[cat] >= MAX_PER_CATEGORY:
                continue
            if MAX_PHOTOS and len(manifest) >= MAX_PHOTOS:
                break
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            (PHOTO_DIR / cat).mkdir(exist_ok=True)
            rel = f"{cat}/{_id}.jpg"
            Image.open(io.BytesIO(raw)).convert("RGB").save(PHOTO_DIR / rel, "JPEG", quality=90)
            per_cat[cat] += 1
            manifest.append({"file": rel, "category": names[synset]})
        if MAX_PHOTOS and len(manifest) >= MAX_PHOTOS:
            break

    GALLERY_PATH.write_text(json.dumps({"photos": manifest}, ensure_ascii=False), encoding="utf-8")
    if not KEEP_RAW:
        shutil.rmtree(RAW_DIR, ignore_errors=True)
    print(f"✓ {len(manifest)} фото · {len(per_cat)} категорий → artifacts/gallery.json · "
          "дальше: python embed.py")


if __name__ == "__main__":
    main()
