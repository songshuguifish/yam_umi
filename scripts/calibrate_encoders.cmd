@echo off
setlocal
cd /d "%~dp0\.."
".venv\Scripts\python.exe" -m sim_teleop.data_collection.calibrate_encoders %*
