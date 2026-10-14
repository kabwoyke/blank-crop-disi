@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm --clean --onefile --windowed --name AutoCropBlankRemover --exclude-module matplotlib --exclude-module scipy --exclude-module pandas --hidden-import pymysql app.py
echo.
echo Done: dist\AutoCropBlankRemover.exe
