@echo off
REM Fold 4 locally (chunk 16 from epoch 1 on), in parallel with folds 2-3 on the Google Cloud A100 (the
REM project's GPU quota is 1, so the cloud cannot run two at once).
REM 6 epochs to match the cloud folds. --resume: a reboot or power cut resumes
REM from fold4_last.pt instead of starting over -- just run this file again.
REM chunk 16 + cap 0.75: ~4.8 GB allocated, ~6.5 GB process, ~4 GB left for Boulder Vision (2 GB).
REM DO NOT edit anything under src/ or scripts/ while this runs.
set PY=D:\kaggle_c\.venv\Scripts\python.exe
set REPO=D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection
set OUT=D:\kaggle_c\work\coat
set LOG=D:\kaggle_c\work\coat_f4.log
set HF_HOME=D:\kaggle_c\hf
set PYTHONUNBUFFERED=1
cd /d "%REPO%"
echo === fold 4 starting %DATE% %TIME% >> "%LOG%" 2>&1
"%PY%" -u scripts\train.py --fold 4 --folds 5 --epochs 6 --batch 1 ^
    --accum 16 --workers 4 --backbone coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k ^
    --size 384 --lr 1e-3 --backbone-lr-mult 0.03 --grad-checkpointing ^
    --encoder-chunk 16 --max-vram-frac 0.75 --resume --out "%OUT%" >> "%LOG%" 2>&1
echo === fold 4 finished %DATE% %TIME% >> "%LOG%" 2>&1
