@echo off
title MLflow Dashboard - Iris Classification
cd /d d:\mloops
echo ==============================================================
echo     Starting MLflow Tracking Dashboard
echo ==============================================================
echo.
echo Dashboard URL: http://127.0.0.1:5000
echo Database:      sqlite:///D:/mloops/mlflow.db
echo.
echo NOTE: Keep this window OPEN while viewing MLflow!
echo Press Ctrl + C to stop MLflow anytime.
echo.
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///D:/mloops/mlflow.db --host 127.0.0.1 --port 5000
pause
