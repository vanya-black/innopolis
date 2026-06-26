"""
Этап 1 — собрать данные.

Выгружает TOP-N игр из RAWG (с описаниями) → artifacts/games.json.
Нужен бесплатный ключ в .env:  RAWG_API_KEY=...   (https://rawg.io/apidocs)

Запуск:  python collect.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from embed import ART, GAMES_PATH

# Ключ ищем и в каталоге демо, и в корне репозитория.
load_dotenv(ART.parent / ".env")
load_dotenv(ART.parent.parent / ".env")
API_KEY = os.environ.get("RAWG_API_KEY", "")

TOP_N = 1000
ORDERING = "-added"        # популярность RAWG; ещё: -metacritic | -rating | -released
RAWG_LIST = "https://api.rawg.io/api/games"
RAWG_DETAIL = "https://api.rawg.io/api/games/{id}"

# genres[0] у RAWG почти всегда "Action" (у него низкий внутренний ID, его вешают
# почти на всё) — берём первый более специфичный со-жанр, иначе карта раскрасится
# в один цвет.
BROAD = {"Action"}


def primary_genre(genres: list[str]) -> str:
    specific = [g for g in genres if g not in BROAD]
    return (specific or genres or ["Unknown"])[0]


def normalize(raw: dict, description: str = "") -> dict:
    genres = [g["name"] for g in raw.get("genres", [])]
    tags = [t["name"] for t in raw.get("tags", [])][:12]
    return {
        "name": raw.get("name", "—"),
        "genre": primary_genre(genres),
        "tags": [t for t in genres + tags if t],
        "description": (description or "").strip(),
        "released": raw.get("released"),
        "rating": raw.get("rating"),
        "metacritic": raw.get("metacritic"),
        "platform": ", ".join(p["platform"]["name"] for p in (raw.get("platforms") or [])),
        "url": f"https://rawg.io/games/{raw.get('slug', '')}",
    }


def fetch_rawg(n: int) -> list[dict]:
    games, page = [], 1
    while len(games) < n:
        r = requests.get(RAWG_LIST, params={
            "key": API_KEY, "ordering": ORDERING,
            "page": page, "page_size": min(40, n - len(games)),
        }, timeout=30)
        r.raise_for_status()
        batch = r.json().get("results", [])
        if not batch:
            break
        games.extend(batch)
        page += 1
    games = games[:n]

    # полные описания тянем параллельно (по одному запросу на игру)
    def full(g):
        try:
            r = requests.get(RAWG_DETAIL.format(id=g["id"]),
                             params={"key": API_KEY}, timeout=30)
            r.raise_for_status()
            return g["id"], r.json().get("description_raw", "")
        except Exception:
            return g["id"], ""

    with ThreadPoolExecutor(max_workers=8) as ex:
        descriptions = dict(ex.map(full, games))

    return [normalize(g, descriptions.get(g["id"], "")) for g in games]


def main() -> None:
    if not API_KEY:
        raise SystemExit("✗ Нет RAWG_API_KEY в .env (бесплатный ключ: https://rawg.io/apidocs)")
    print(f"Выгружаю TOP-{TOP_N} игр из RAWG (ordering={ORDERING})…")
    games = fetch_rawg(TOP_N)

    ART.mkdir(exist_ok=True)
    GAMES_PATH.write_text(json.dumps({
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "RAWG",
        "ordering": ORDERING,
        "count": len(games),
        "games": games,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with_desc = sum(1 for g in games if g["description"])
    print(f"✓ {len(games)} игр → artifacts/games.json (с описанием: {with_desc}) · "
          "дальше: python embed.py")


if __name__ == "__main__":
    main()
