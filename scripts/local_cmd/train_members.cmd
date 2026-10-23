@echo off
REM Train ensemble members, one per line, resumable.
REM
REM Each entry is BACKBONE|SIZE|FOLD|BACKBONE_LR_MULT. A member whose checkpoint
REM already exists is skipped, so this can be restarted freely.
REM
REM The mult is the fix for an outright training failure, not a tuning knob. At a
REM uniform 3e-4 the DINOv3 encoder collapsed to chance by epoch 2 -- weak AUC ran
REM 0.660 -> 0.632 -> 0.540 while the loss stayed flat, which is the signature of a
REM head-sized step overwriting pretrained features rather than adapting them. The
REM OneCycle peak at pct_start=0.25 lands on exactly that epoch. 0.05 puts the
REM backbone at 1.5e-5, inside the usual 1e-5..5e-5 fine-tuning band.
REM
REM cmd rather than PowerShell, on a semicolon-free path: PowerShell wraps native
REM stderr in error records and cmd /c splits the repo path at ';'.

setlocal enabledelayedexpansion
set PY=D:\kaggle_c\.venv\Scripts\python.exe
set REPO=D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection
set OUT=D:\kaggle_c\work\members
set LOG=D:\kaggle_c\work\members.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\kaggle_c\hf

if not exist "%OUT%" mkdir "%OUT%"
cd /d "%REPO%"

echo === members started %DATE% %TIME% >> "%LOG%" 2>&1

for %%M in (
    convnext_tiny.dinov3_lvd1689m^|384^|0^|0.03
) do (
    for /f "tokens=1,2,3,4 delims=|" %%A in ("%%M") do (
        set CK=%OUT%\fold%%C_%%A.pt
        if exist "!CK!" (
            echo === %%A size %%B fold %%C already done >> "%LOG%" 2>&1
        ) else (
            echo === %%A size %%B fold %%C mult %%D  %TIME% >> "%LOG%" 2>&1
            "%PY%" -u scripts\train.py --fold %%C --folds 5 --epochs 8 ^
                --batch 1 --accum 16 --workers 4 --backbone %%A --size %%B ^
                --lr 1e-3 --backbone-lr-mult %%D ^
                --grad-checkpointing --out "%OUT%" >> "%LOG%" 2>&1
        )
    )
)

echo === members done %DATE% %TIME% >> "%LOG%" 2>&1
endlocal
