@echo off
echo ========================================
echo   NNFactory - Neural Network Blueprint Maker
echo ========================================
echo.

echo [1/3] Setting up backend dependencies...
cd backend
pip install -r requirements.txt
cd ..

echo.
echo [2/3] Starting backend server...
start "NNFactory Backend" cmd /k "cd backend && uvicorn main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo.
echo [3/3] Starting frontend server...
start "NNFactory Frontend" cmd /k "cd frontend && python -m http.server 3000"

echo.
echo ========================================
echo   NNFactory is running!
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Press any key to stop all servers...
pause >nul

echo Stopping servers...
taskkill /FI "WindowTitle eq NNFactory Backend" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq NNFactory Frontend" /T /F >nul 2>&1

echo Servers stopped.
