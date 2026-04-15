@echo off
setlocal

:menu
echo ==============================
echo  Select which bot to update:
echo ==============================
echo 1. ForsakenGenBot
echo 2. DoomByFateGeneratorBot
echo 3. Update ALL
echo 4. Exit
echo ==============================
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto genbot
if "%choice%"=="2" goto dbfbot
if "%choice%"=="3" goto all
if "%choice%"=="4" goto end
echo Invalid choice. Try again.
goto menu

:genbot
if exist "ForsakenGenBot" (
    echo Removing old ForsakenGenBot...
    rmdir /s /q "ForsakenGenBot"
)

if exist "RobloxScripting" (
    echo Cleaning up old RobloxScripting repo...
    rmdir /s /q "RobloxScripting"
)

git clone https://github.com/L0st-Illusi0n/RobloxScripting

move "RobloxScripting\Bots\ForsakenGeneratorBot" ".\ForsakenGenBot"

rmdir /s /q "RobloxScripting"

echo ForsakenGenBot updated successfully!
call :build_prompt GEN
pause
goto end

:dbfbot
if exist "DoomByFateGeneratorBot" (
    echo Removing old DoomByFateGeneratorBot...
    rmdir /s /q "DoomByFateGeneratorBot"
)

if exist "RobloxScripting" (
    echo Cleaning up old RobloxScripting repo...
    rmdir /s /q "RobloxScripting"
)

git clone https://github.com/L0st-Illusi0n/RobloxScripting

move "RobloxScripting\Bots\DoomByFateGeneratorBot" ".\DoomByFateGeneratorBot"

rmdir /s /q "RobloxScripting"

echo DoomByFateGeneratorBot updated successfully!
call :build_prompt DBF
pause
goto end

:all
if exist "ForsakenGenBot" (
    echo Removing old ForsakenGenBot...
    rmdir /s /q "ForsakenGenBot"
)
if exist "DoomByFateGeneratorBot" (
    echo Removing old DoomByFateGeneratorBot...
    rmdir /s /q "DoomByFateGeneratorBot"
)

if exist "RobloxScripting" (
    echo Cleaning up old RobloxScripting repo...
    rmdir /s /q "RobloxScripting"
)

git clone https://github.com/L0st-Illusi0n/RobloxScripting

move "RobloxScripting\Bots\ForsakenGeneratorBot" ".\ForsakenGenBot"
move "RobloxScripting\Bots\DoomByFateGeneratorBot" ".\DoomByFateGeneratorBot"

rmdir /s /q "RobloxScripting"

echo All supported bots updated successfully!
call :build_prompt ALL
pause
goto end

:build_prompt
set "target=%~1"
echo.
choice /M "Build the updated bot(s) now?"
if errorlevel 2 (
    echo Skipping build step.
    goto build_prompt_end
)

if /I "%target%"=="GEN" (
    call :build_gen
    goto build_prompt_end
)
if /I "%target%"=="DBF" (
    call :build_dbf
    goto build_prompt_end
)
if /I "%target%"=="ALL" (
    call :build_gen
    if errorlevel 1 goto build_prompt_end
    call :build_dbf
    goto build_prompt_end
)

echo Unrecognized build target: %target%

:build_prompt_end
exit /b

:build_gen
if not exist "ForsakenGenBot\src\ForsakenGenBotGUI.py" (
    echo ForsakenGenBot files not found. Skipping build.
    exit /b 1
)

pushd "ForsakenGenBot\src"
pyinstaller ForsakenGenBotGUI.py --clean --onefile --noconsole --icon Icon.ico
if errorlevel 1 (
    echo PyInstaller failed for ForsakenGenBotGUI.py.
    popd
    exit /b 1
)
pyinstaller ForsakenGenBotGUI.spec --clean
if errorlevel 1 (
    echo PyInstaller failed for ForsakenGenBotGUI.spec.
    popd
    exit /b 1
)
popd
echo ForsakenGenBot build completed successfully.
exit /b 0

:build_dbf
if not exist "DoomByFateGeneratorBot\src\DBFGenBot.py" (
    echo DoomByFateGeneratorBot files not found. Skipping build.
    exit /b 1
)

pushd "DoomByFateGeneratorBot\src"
python -m nuitka --onefile --output-dir="dist" --output-filename="DBFGenBot.exe" --include-package=interception "DBFGenBot.py"
if errorlevel 1 (
    echo Nuitka failed for DBFGenBot.py.
    popd
    exit /b 1
)
copy /y "DBFGenBot.timing.json" "dist\DBFGenBot.timing.json" >nul
if errorlevel 1 (
    echo Failed to copy DBFGenBot.timing.json into dist.
    popd
    exit /b 1
)
if exist "dist\DBFGenBot.build" rmdir /s /q "dist\DBFGenBot.build"
if exist "dist\DBFGenBot.dist" rmdir /s /q "dist\DBFGenBot.dist"
if exist "dist\DBFGenBot.onefile-build" rmdir /s /q "dist\DBFGenBot.onefile-build"
popd
echo DoomByFateGeneratorBot build completed successfully.
exit /b 0

:end
exit
