@echo off
REM Train every fold locally, detached, surviving the editor session.
REM
REM cmd rather than PowerShell on purpose. PowerShell wraps a native command's
REM stderr in NativeCommandError records, so the first run died on a harmless
REM HuggingFace "set HF_TOKEN" warning and produced zero epochs. cmd redirection
REM just writes the bytes.
REM
REM Launch:  start "" /B cmd /c scripts\train_all_local.cmd
REM
REM Resumable: a fold whose checkpoint exists is skipped, so a restart continues
REM the sweep instead of redoing finished work.

setlocal
set PY=D:\kaggle_c\.venv\Scripts\python.exe
set REPO=D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection
set OUT=D:\kaggle_c\work\models384
set LOG=D:\kaggle_c\work\train_local.log
set PYTHONUNBUFFERED=1
set HF_HUB_DISABLE_TELEMETRY=1

if not exist "%OUT%" mkdir "%OUT%"
cd /d "%REPO%"

echo === started %DATE% %TIME% >> "%LOG%" 2>&1

for %%F in (0 1 2 3 4) do (
    if exist "%OUT%\fold%%F_resnet34.pt" (
        echo === fold %%F already done, skipping >> "%LOG%" 2>&1
    ) else (
        echo === fold %%F  %TIME% >> "%LOG%" 2>&1
        "%PY%" -u scripts\train.py --fold %%F --folds 5 --epochs 8 --batch 1 --accum 16 --workers 4 --out "%OUT%" >> "%LOG%" 2>&1
    )
)

echo === all folds done %DATE% %TIME% >> "%LOG%" 2>&1
endlocal
