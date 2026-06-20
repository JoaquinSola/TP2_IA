@echo off
title Asistente de Pagos IA - Servidor
cd /d "%~dp0"

echo.
echo  ==========================================
echo   ASISTENTE DE PAGOS IA - UTN 2026
echo  ==========================================
echo.

:: Quitar BOM del .env si existe (Windows guarda con BOM y rompe dotenv)
venv\Scripts\python -c "c=open('.env','r',encoding='utf-8-sig').read();open('.env','w',encoding='utf-8',newline='\n').write(c)" 2>nul

echo  Servidor iniciando en http://localhost:8000
echo  Para detenerlo: Ctrl+C
echo.

venv\Scripts\python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

pause
