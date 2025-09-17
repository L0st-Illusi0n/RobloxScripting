@echo off
setlocal

REM Ensure we run from the script's directory
pushd "%~dp0"

echo Building ForsakenEntanglementBotGUI standalone executable...
pyinstaller ForsakenEntanglementBotGUI.py --clean --onefile --noconsole --icon Icon.ico --add-data "Images;Images"
if errorlevel 1 goto :error

echo Rebuilding using ForsakenEntanglementBotGUI.spec...
pyinstaller ForsakenEntanglementBotGUI.spec --clean
if errorlevel 1 goto :error

echo Build completed seccessfully! Press any button to exit...
pause >nul
popd
exit /b 0

:error
echo Build failed. See PyInstaller output above for details.
popd
exit /b 1

