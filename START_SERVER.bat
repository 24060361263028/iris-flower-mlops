@echo off
title Iris Flower Classification - MLOps API Server
cd /d d:\mloops
echo ==============================================================
echo     Starting Iris Flower Classification MLOps API Server
echo ==============================================================
echo.
echo Server running at: http://127.0.0.1:8000
echo Swagger UI Docs:  http://127.0.0.1:8000/docs
echo.
echo NOTE: Keep this window OPEN while using the API or Web Docs!
echo Press Ctrl + C to stop the server anytime.
echo.
.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
pause
