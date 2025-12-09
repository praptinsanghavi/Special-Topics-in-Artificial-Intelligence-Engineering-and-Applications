@echo off
echo ===================================================
echo 🛠️  Starting Technical Documentation Assistant
echo ===================================================

echo [1/2] Checking requirements...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error installing requirements.
    pause
    exit /b %ERRORLEVEL%
)

echo [2/2] Launching Streamlit App...
streamlit run app.py

pause
