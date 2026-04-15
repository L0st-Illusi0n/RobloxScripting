@echo off
setlocal

pushd "%~dp0"

set "SCRIPT=DBFGenBot.py"
set "CONFIG_FILE=DBFGenBot.timing.json"
set "OUTPUT_DIR=dist"
set "OUTPUT_NAME=DBFGenBot.exe"
set "BUILD_STEM=DBFGenBot"

echo Building DoomByFateGenerator Bot standalone executable...
python -m nuitka --onefile --output-dir="%OUTPUT_DIR%" --output-filename="%OUTPUT_NAME%" --include-package=interception "%SCRIPT%"
if errorlevel 1 goto :error

echo Copying editable timing config...
copy /y "%CONFIG_FILE%" "%OUTPUT_DIR%\%CONFIG_FILE%" >nul
if errorlevel 1 goto :error

echo Cleaning temporary Nuitka folders...
if exist "%OUTPUT_DIR%\%BUILD_STEM%.build" rmdir /s /q "%OUTPUT_DIR%\%BUILD_STEM%.build"
if exist "%OUTPUT_DIR%\%BUILD_STEM%.dist" rmdir /s /q "%OUTPUT_DIR%\%BUILD_STEM%.dist"
if exist "%OUTPUT_DIR%\%BUILD_STEM%.onefile-build" rmdir /s /q "%OUTPUT_DIR%\%BUILD_STEM%.onefile-build"

echo Build completed successfully.
echo Output: %CD%\%OUTPUT_DIR%\%OUTPUT_NAME%
echo Config: %CD%\%OUTPUT_DIR%\%CONFIG_FILE%
pause >nul
popd
exit /b 0

:error
echo Build failed. See Nuitka output above for details.
popd
exit /b 1
