@echo off
setlocal
cd /d "%~dp0"

REM OmniRAG AI - start every service on Windows in separate windows.
REM First-time setup (see README):
REM   winget install Ollama.Ollama
REM   ollama pull qwen3-vl:2b-instruct
REM   ollama pull qwen2.5-coder:3b        (optional, better coding model)
REM   python -m venv .venv && .venv\Scripts\pip install -e .
REM   npm --prefix frontend install

where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama is not installed or not on PATH. Install it with:
  echo   winget install Ollama.Ollama
  pause
  exit /b 1
)

REM Start the model server unless it is already running.
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if errorlevel 1 start "Ollama" /min ollama serve

REM Optional better model for the coding assistant; harmless if absent.
set OMNIRAG_OLLAMA_CODE_MODEL=qwen2.5-coder:3b

start "OmniRAG API" cmd /k ".venv\Scripts\python.exe -m uvicorn omnirag.main:app --host 127.0.0.1 --port 8000"
start "OmniRAG worker" cmd /k ".venv\Scripts\python.exe -m omnirag.worker"
start "OmniRAG frontend" cmd /k "npm --prefix frontend run dev"

echo.
echo OmniRAG is starting:
echo   UI        http://127.0.0.1:5173
echo   API       http://127.0.0.1:8000  (docs at /docs)
echo   Model     qwen3-vl:2b-instruct (knowledge) + qwen2.5-coder:3b (coding)
echo.
echo Close the service windows to stop the app.
pause
