@echo off
REM Fold 0 with a DINOv3-pretrained ConvNeXt, against the resnet34 baseline.
REM
REM Gradient checkpointing is not optional here. Without it this backbone peaks
REM at 12+ GB on a 10.7 GB card, spills to host memory, and runs 17x slower --
REM which is what made an earlier benchmark report 25 h/fold and nearly got the
REM whole idea discarded. With it: 3.6 GB, 2.7 h/fold.
REM
REM cmd rather than PowerShell, on a semicolon-free path, because PowerShell
REM wraps native stderr in error records and cmd /c splits the repo path at ';'.

setlocal
set PY=D:\kaggle_c\.venv\Scripts\python.exe
set REPO=D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection
set OUT=D:\kaggle_c\work\models_dinov3
set LOG=D:\kaggle_c\work\train_dinov3.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\kaggle_c\hf

if not exist "%OUT%" mkdir "%OUT%"
cd /d "%REPO%"

echo === dinov3 fold 0  %DATE% %TIME% >> "%LOG%" 2>&1
"%PY%" -u scripts\train.py --fold 0 --folds 5 --epochs 8 --batch 1 --accum 16 ^
    --workers 4 --backbone convnext_tiny.dinov3_lvd1689m --grad-checkpointing ^
    --out "%OUT%" >> "%LOG%" 2>&1
echo === done %DATE% %TIME% >> "%LOG%" 2>&1
endlocal
