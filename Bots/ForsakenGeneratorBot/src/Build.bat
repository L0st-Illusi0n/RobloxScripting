@echo off
setlocal

REM Ensure we run from the script's directory
pushd "%~dp0"

echo Building ForsakenGenBotGUI standalone executable...
pyinstaller ForsakenGenBotGUI.py --clean --onefile --noconsole --icon Icon.ico
if errorlevel 1 goto :error

echo Rebuilding using ForsakenGenBotGUI.spec...
pyinstaller ForsakenGenBotGUI.spec --clean
if errorlevel 1 goto :error

echo Build completed successfully.
popd
exit /b 0

:error
echo Build failed. See PyInstaller output above for details.
popd
exit /b 1
