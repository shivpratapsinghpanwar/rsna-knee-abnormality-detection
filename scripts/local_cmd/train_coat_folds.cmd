@echo off
REM Folds 1-4 of CoAtNet, non-ROI. Resumable: a fold whose checkpoint exists is
REM skipped, so this survives a restart.
REM
REM Compartment ROI was tested on fold 0 and REJECTED: worse at all 7 epochs,
REM -0.021 weak AUC and -0.038 gold against the identical non-ROI fold. Rescaling
REM a crop interpolates rather than adding information, and splitting into 6 views
REM over the same 16 slices cost global context for nothing. Real resolution would
REM need re-cropping from DICOM. Archived under work\coat_roi_NEGATIVE.
REM
REM --max-vram-frac 0.75 (~8 GB): a safety ceiling only, set well ABOVE what chunk 8
REM actually uses (~3.1 GB allocated, ~5 GB process). At 0.60 the ceiling sat close
REM to the working set, the caching allocator kept hitting it and freeing/retrying
REM every step, and epochs took ~18 h instead of ~2.4 h. Boulder Vision (~2 GB) is
REM protected by real headroom (~5 GB free) plus this ceiling as a backstop.
REM
REM chunk 8 rather than 16: 3.13 GB instead of 4.84 GB peak, 17% slower per step.
REM That leaves ~5 GB of the 10 GB card free so another job (MaskRCNN inference,
REM ~2 GB) can share it without risking an OOM that would cost a whole fold.
REM
REM DO NOT edit anything under src/ or scripts/ while this runs: Windows spawns
REM DataLoader workers that re-import the module, so an edit mid-run crashes the
REM job against the already-constructed dataset object. That is how fold 1 was
REM lost the first time.
setlocal enabledelayedexpansion
set PY=D:\kaggle_c\.venv\Scripts\python.exe
set REPO=D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection
set OUT=D:\kaggle_c\work\coat
set LOG=D:\kaggle_c\work\coat.log
set HF_HOME=D:\kaggle_c\hf
set PYTHONUNBUFFERED=1
cd /d "%REPO%"
for %%F in (1 2 3 4) do (
    set CK=%OUT%\fold%%F_coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k.pt
    if exist "!CK!" (
        echo === fold %%F already done >> "%LOG%" 2>&1
    ) else (
        echo === fold %%F starting %DATE% %TIME% >> "%LOG%" 2>&1
        "%PY%" -u scripts\train.py --fold %%F --folds 5 --epochs 8 --batch 1 ^
            --accum 16 --workers 4 --backbone coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k ^
            --size 384 --lr 1e-3 --backbone-lr-mult 0.03 --grad-checkpointing ^
            --encoder-chunk 8 --max-vram-frac 0.75 --out "%OUT%" >> "%LOG%" 2>&1
    )
)
echo === folds 1-4 done %DATE% %TIME% >> "%LOG%" 2>&1
endlocal
