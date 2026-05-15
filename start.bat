@echo off
cd /d "%~dp0"
echo ========================================
echo   NNFactory - Neural Network Blueprint Maker
echo ========================================
echo.

echo [1/3] Setting up backend dependencies...
cd backend
call pip install -r requirements.txt
cd ..
echo.

echo [2/3] Starting backend server...
start /b cmd /c "cd backend && uvicorn main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo [3/3] Starting frontend server...
start /b cmd /c "cd frontend && python -m http.server 4000"

echo.
echo ========================================
echo   NNFactory is running!
echo   Frontend: http://localhost:4000
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo All processes running in this console window.
echo Press any key to stop all servers...
pause >nul

echo.
echo Stopping servers...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /PID %%a /T /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":4000" ^| findstr "LISTENING"') do taskkill /PID %%a /T /F >nul 2>&1
echo Servers stopped.
echo.
