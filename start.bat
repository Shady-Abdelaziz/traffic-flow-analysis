@echo off
rem Double-click to open the Traffic Flow Analysis dashboard in your browser.
cd /d "%~dp0"
python run.py serve
pause
