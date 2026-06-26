"""
✏️ Нарисуй — найди похожее (дашборд).

Работает на готовых артефактах (collect.py → embed.py): CLIP-эмбеддинги всех
фотографий посчитаны заранее, в реальном времени — лишь один прогон CLIP по
рисунку и поиск ближайших. «Запрос» здесь — РИСУНОК, корпус — КАРТИНКИ.

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import embed

st.set_page_config(page_title="Нарисуй — найди похожее", page_icon="✏️", layout="wide")


@st.cache_data(show_spinner="Загружаю индекс…")
def load(sig: float):
    photos = json.loads(embed.GALLERY_PATH.read_text(encoding="utf-8"))["photos"]
    meta = json.loads(embed.META_PATH.read_text(encoding="utf-8"))
    matrix = np.load(embed.EMB_PATH)
    return photos, meta, matrix


@st.cache_resource(show_spinner="Загружаю модель CLIP…")
def get_model():
    return embed.load_model()


st.title("✏️ Нарисуй — найди похожее")
st.caption("Рисунок → CLIP-вектор → ближайшие фотографии в том же пространстве смыслов")

if not (embed.EMB_PATH.exists() and embed.META_PATH.exists()
        and embed.GALLERY_PATH.exists() and embed.PHOTO_DIR.exists()):
    st.error("Нет готового индекса. Сначала: `python collect.py` → `python embed.py`")
    st.stop()

photos, meta, matrix = load(embed.EMB_PATH.stat().st_mtime)

st.sidebar.title("✏️ Нарисуй — найди похожее")
st.sidebar.caption(f"Модель **{meta['model']}** · {meta['n']} фото · "
                   f"{meta['n_categories']} категорий · {meta['dims']}d")
k = st.sidebar.slider("Сколько похожих показывать (k)", 4, 24, 12)
stroke = st.sidebar.slider("Толщина кисти", 2, 30, 12)

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader("✏️ Нарисуйте объект")
    if st.button("🗑️ Очистить холст", width="stretch"):
        st.session_state.canvas_key += 1
    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)", stroke_width=stroke, stroke_color="#000000",
        background_color="#FFFFFF", height=340, width=340, drawing_mode="freedraw",
        display_toolbar=True, key=f"canvas_{st.session_state.canvas_key}",
    )
    st.caption("Чёрным по белому, крупно и по центру.")

neighbor_idx = neighbor_sims = None
if canvas.image_data is not None and not embed.sketch_is_empty(canvas.image_data):
    get_model()  # прогрев/кэш
    qvec = embed.embed_sketch(Image.fromarray(canvas.image_data.astype("uint8"), "RGBA"))
    neighbor_idx, neighbor_sims = embed.nearest(qvec, matrix, k)

with right:
    st.subheader("🔍 Похожие фотографии")
    if neighbor_idx is None:
        st.info("Нарисуйте что-нибудь слева — справа появятся самые похожие фото.")
    else:
        top = Counter(photos[i]["category"] for i in neighbor_idx).most_common(3)
        st.caption("Скорее всего это: **" + "**, **".join(c for c, _ in top) + "**")
        ncol = 4
        for r in range(0, len(neighbor_idx), ncol):
            cols = st.columns(ncol)
            for col, i, s in zip(cols, neighbor_idx[r:r + ncol], neighbor_sims[r:r + ncol]):
                p = photos[i]
                col.image(str(embed.PHOTO_DIR / p["file"]), width="stretch")
                col.caption(f"{p['category']} · {s:.2f}")

with st.expander("ℹ️ Как это работает"):
    st.markdown(
        f"1. **CLIP** заранее перевёл каждую из {meta['n']} фотографий в вектор из "
        f"{meta['dims']} чисел — «координату смысла».\n"
        "2. Рисунок проходит через **ту же** модель и попадает в **то же** пространство.\n"
        "3. Ищем фото с ближайшими векторами (косинусная близость) — это и есть «похожие».\n\n"
        "> CLIP ловит **категорию и общую форму** (кошка → кошки), но не точную позу — "
        "рисунок и фото разные «домены». Источник фото — датасет **Sketchy**."
    )
