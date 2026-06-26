#!/usr/bin/env bash
# Полный пайплайн демо «Нарисуй — найди похожее».
# Запуск:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if [ -d .venv ]; then PY=.venv/bin/python; fi

echo "==> 1/3  Сбор фото-галереи Sketchy"
$PY collect.py

echo "==> 2/3  CLIP-эмбеддинги галереи"
$PY embed.py

echo "==> 3/3  Запуск дашборда"
exec $PY -m streamlit run app.py
