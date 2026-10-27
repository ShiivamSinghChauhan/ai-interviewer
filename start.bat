@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
if not exist models\silero_vad.onnx .venv\Scripts\python.exe scripts\download_models.py
if not exist .env copy .env.example .env
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe -m app.main
