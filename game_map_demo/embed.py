"""
Этап 2 — эмбеддинги + карта (и общий «движок» демо).

Берёт карточки игр из artifacts/games.json, кодирует их одной моделью
(all-MiniLM-L6-v2) в векторы и раскладывает в 2D для карты. Здесь же лежат
функции, которыми пользуется дашборд: загрузка модели, эмбеддинг запроса и
поиск ближайших соседей — весь «смысловой» код в одном месте.

Запуск:  python embed.py        # читает games.json → пишет эмбеддинги и координаты
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- Пути и модель (общие для collect.py / embed.py / app.py) -------------- #
BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
GAMES_PATH = ART / "games.json"        # пишет collect.py
META_PATH = ART / "meta.json"          # пишет embed.py
EMB_PATH = ART / "embeddings.npy"      # (N, 384), L2-нормированные
COORDS = {                             # 2D-проекции для карты
    "UMAP": ART / "coords_umap.npy",
    "PCA": ART / "coords_pca.npy",
}

# Лёгкая модель (~90 МБ): кодирует и запрос, и описание одинаково (без префиксов).
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def load_model():
    """Грузим модель один раз (скачивается с HuggingFace при первом запуске)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_text(game: dict) -> str:
    """Текст карточки: имя + жанр (дважды, для веса) + теги + описание."""
    parts = [
        game.get("name", ""),
        game.get("genre", ""),
        game.get("genre", ""),
        " ".join(game.get("tags", [])),
        game.get("description", ""),
    ]
    return " . ".join(p for p in parts if p).strip()


def embed(texts: list[str]) -> np.ndarray:
    """Тексты → L2-нормированные векторы (скалярное произведение = косинус)."""
    vecs = load_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def reduce_2d(vecs: np.ndarray, method: str) -> np.ndarray:
    """Многомерные векторы → 2D. PCA — линейная, UMAP — нелинейная (плотнее кластеры)."""
    if method == "PCA":
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(vecs)
    else:  # UMAP
        import umap
        coords = umap.UMAP(n_components=2, metric="cosine", random_state=42).fit_transform(vecs)
    return np.asarray(coords, dtype=np.float32)


def nearest(query_vec: np.ndarray, matrix: np.ndarray, k: int, exclude: int | None = None):
    """Топ-k ближайших по косинусной близости (векторы уже нормированы)."""
    sims = matrix @ query_vec
    if exclude is not None:
        sims[exclude] = -np.inf
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def main() -> None:
    if not GAMES_PATH.exists():
        raise SystemExit("✗ Нет artifacts/games.json — сначала запустите: python collect.py")
    payload = json.loads(GAMES_PATH.read_text(encoding="utf-8"))
    games = payload["games"]

    print(f"Кодирую {len(games)} игр моделью {MODEL_NAME}…")
    vecs = embed([build_text(g) for g in games])
    np.save(EMB_PATH, vecs)
    for method, path in COORDS.items():
        print(f"Проекция {method}…")
        np.save(path, reduce_2d(vecs, method))

    META_PATH.write_text(json.dumps({
        "model": MODEL_NAME,
        "dims": int(vecs.shape[1]),
        "n": len(games),
        "source": payload.get("source", "RAWG"),
        "projections": list(COORDS),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Готово: {vecs.shape} → artifacts/ · дальше: streamlit run app.py")


if __name__ == "__main__":
    main()
