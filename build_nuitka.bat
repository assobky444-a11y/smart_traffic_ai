@echo off
setlocal

REM ===== Project Entry =====
set ENTRY=app_wrapper.py
set BASENAME=app_wrapper
set OUTDIR=dist
set EXE=TraffiCountPro.exe

echo Cleaning old build...
if exist %OUTDIR% rd /s /q %OUTDIR%

echo Starting Nuitka build...

python -m nuitka "%ENTRY%" ^
    --standalone ^
    --onefile ^
    --output-dir=%OUTDIR% ^
    --windows-console-mode=disable ^
    --enable-plugin=tk-inter ^
    --assume-yes-for-downloads ^
    --remove-output ^
    --include-data-dir=templates=templates ^
    --include-data-dir=static=static ^
    --include-data-dir=modal=modal ^
    --include-data-file=device.json=device.json ^
    --include-data-file=user_config.json=user_config.json ^
    --include-data-file=license.json=license.json

echo.

if exist "%OUTDIR%\%BASENAME%.exe" (
    move "%OUTDIR%\%BASENAME%.exe" "%OUTDIR%\%EXE%"
    echo.
    echo =========================================
    echo Build finished successfully!
    echo Output: %OUTDIR%\%EXE%
    echo =========================================
) else (
    echo.
    echo Build failed!
)

endlocal