@echo off
echo Starting Backend (port 5000)...
start "Backend" cmd /k "cd /d %~dp0ui_handoff && python app.py"

timeout /t 3 /nobreak >nul

echo Starting Organizer Frontend (port 5001)...
start "Organizer Frontend" cmd /k "cd /d %~dp0frontend_organizer && python app.py"

echo Starting Public Frontend Flask (port 5002)...
start "Public Frontend Flask" cmd /k "cd /d %~dp0frontend_public && python app.py"

echo Starting Public React App (port 5174)...
start "Public React" cmd /k "cd /d %~dp0frontend-public && npm run dev"

echo.
echo All servers started:
echo   Backend:                  http://localhost:5000
echo   Organizer Frontend:       http://localhost:5001
echo   Public Frontend (Flask):  http://localhost:5002
echo   Public React App:         http://localhost:5174
