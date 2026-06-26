#!/usr/bin/env bash
# Полный пайплайн: сбор данных → эмбеддинги → дашборд.
set -e
cd "$(dirname "$0")"

if [ -d .venv ]; then source .venv/bin/activate; fi

echo "==> Этап 1: сбор данных (RAWG)"
python collect.py

echo "==> Этап 2: расчёт эмбеддингов"
python embed.py

echo "==> Этап 3: дашборд"
streamlit run app.py
