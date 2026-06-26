"""
🎮 Карта игр в пространстве эмбеддингов (дашборд).

Работает на готовых артефактах (collect.py → embed.py): эмбеддинги всех игр
посчитаны заранее, в реальном времени — только эмбеддинг запроса и поиск
ближайших соседей. Поэтому отвечает мгновенно.

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import embed

st.set_page_config(page_title="Карта игр", page_icon="🎮", layout="wide")


@st.cache_data(show_spinner="Загружаю данные…")
def load(sig: float):
    games = json.loads(embed.GAMES_PATH.read_text(encoding="utf-8"))["games"]
    meta = json.loads(embed.META_PATH.read_text(encoding="utf-8"))
    matrix = np.load(embed.EMB_PATH)
    coords = {m: np.load(p) for m, p in embed.COORDS.items() if p.exists()}
    return games, meta, matrix, coords


@st.cache_resource(show_spinner="Загружаю модель для запроса…")
def get_model():
    return embed.load_model()


st.title("🎮 Карта игр в пространстве эмбеддингов")

if not all(p.exists() for p in (embed.GAMES_PATH, embed.META_PATH, embed.EMB_PATH)):
    st.error("Нет готовых артефактов. Сначала: `python collect.py` → `python embed.py`")
    st.stop()

games, meta, matrix, coords = load(embed.EMB_PATH.stat().st_mtime)
df = pd.DataFrame(games)
df["desc_short"] = df["description"].fillna("").astype(str).str.slice(0, 160)
names = df["name"].tolist()

# --- Боковая панель -------------------------------------------------------- #
st.sidebar.title("🎮 Карта игр")
st.sidebar.caption(f"Модель **{meta['model'].split('/')[-1]}** · "
                   f"{meta['n']} игр · {meta['dims']}d")
proj = st.sidebar.radio("Проекция (2D)", list(coords), horizontal=True,
                        help="PCA — линейная (общая форма облака); UMAP — нелинейная "
                             "(плотнее кластеры).")
k = st.sidebar.slider("Сколько соседей (k)", 3, 20, 8)
all_genres = sorted(df["genre"].fillna("Unknown").unique())
sel_genres = st.sidebar.multiselect("Жанры на карте", all_genres, default=all_genres)

xy = coords[proj]
df["x"], df["y"] = xy[:, 0], xy[:, 1]

c1, c2, c3 = st.columns(3)
c1.metric("Игр", len(df))
c2.metric("Жанров", df["genre"].nunique())
c3.metric("Размерность вектора", meta["dims"])

# --- Поиск ----------------------------------------------------------------- #
st.subheader("🔍 Поиск ближайших соседей")
mode = st.radio("Режим", ["По игре из списка", "Свободный текст"], horizontal=True)

query_idx = None
nb_idx = nb_sims = None
label = ""
if mode == "По игре из списка":
    chosen = st.selectbox("Выберите игру", names)
    query_idx = names.index(chosen)
    label = chosen
    nb_idx, nb_sims = embed.nearest(matrix[query_idx], matrix, k, exclude=query_idx)
else:
    q = st.text_input("Опишите, что хотите найти",
                      placeholder="например: open world fantasy rpg with dragons")
    if q.strip():
        get_model()
        nb_idx, nb_sims = embed.nearest(embed.embed([q.strip()])[0], matrix, k)
        label = f"«{q.strip()}»"


# --- Карта ----------------------------------------------------------------- #
def figure():
    base = df[df["genre"].isin(sel_genres)]
    fig = px.scatter(base, x="x", y="y", color="genre", opacity=0.65,
                     custom_data=["name", "genre", "rating", "desc_short"])
    fig.update_traces(
        marker=dict(size=10, line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · ⭐%{customdata[2]}"
                      "<br><i>%{customdata[3]}</i><extra></extra>",
    )
    if nb_idx is not None:
        nb = df.iloc[nb_idx]
        fig.add_trace(go.Scatter(
            x=nb["x"], y=nb["y"], mode="markers", name="Соседи",
            marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(width=3, color="#E8623D")),
            text=nb["name"], hovertemplate="<b>%{text}</b><extra>сосед</extra>"))
    if query_idx is not None:
        qr = df.iloc[query_idx]
        fig.add_trace(go.Scatter(
            x=[qr["x"]], y=[qr["y"]], mode="markers", name="Запрос",
            marker=dict(size=22, symbol="star", color="#FFD700", line=dict(width=1.5, color="#333")),
            text=[qr["name"]], hovertemplate="<b>%{text}</b><extra>запрос</extra>"))
    fig.update_layout(height=600, legend_title_text="Жанр",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=0, r=0, t=10, b=0))
    return fig


left, right = st.columns([3, 2])
with left:
    st.plotly_chart(figure(), width="stretch")
    st.caption(f"Проекция **{proj}** · точки рядом ⇒ игры похожи по смыслу · цвет = жанр.")
with right:
    if nb_idx is not None:
        st.markdown(f"**Ближайшие к {label}:**")
        st.dataframe(pd.DataFrame({
            "Игра": df.iloc[nb_idx]["name"].values,
            "Жанр": df.iloc[nb_idx]["genre"].values,
            "Похожесть": np.round(nb_sims, 3),
        }), width="stretch", hide_index=True)
    else:
        st.info("Выберите игру или введите текстовый запрос, чтобы увидеть соседей.")
