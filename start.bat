@echo off
echo Iniciando Agente IA - Asistencia Visual para Pago de Facturas...
echo.

if not exist venv (
    echo Creando entorno virtual...
    python -m venv venv
)

if not exist .env (
    echo ERROR: No se encontro el archivo .env
    echo Copialo desde .env.example y completalo con tu GEMINI_API_KEY
    pause
    exit /b 1
)

echo Instalando dependencias...
venv\Scripts\pip install -r requirements.txt --quiet

echo.
echo Iniciando servidor en http://localhost:8000
echo Presiona Ctrl+C para detener.
echo.

venv\Scripts\python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
