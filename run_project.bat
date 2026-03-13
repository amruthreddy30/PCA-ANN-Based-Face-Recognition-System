@echo off
echo ==============================================
echo   Face Recognition Project Setup and Execution
echo ==============================================

echo [1/3] Installing Required Libraries...
pip install -r requirements.txt

echo.
echo [2/3] Checking Dataset...
if not exist "dataset\" (
    echo [WARNING] Dataset folder not found. A complete run requires images.
    echo Please run 'python generate_dummy_dataset.py' to capture faces using your webcam.
) else (
    echo [OK] Dataset folder exists.
)

echo.
echo [3/3] Running Main Face Recognition Script...
python face_recognition.py

echo.
pause
