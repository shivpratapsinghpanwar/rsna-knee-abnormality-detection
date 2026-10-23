@echo off
REM CoAtNet-2 on the rebuilt corpus (laterality-mirrored, per-series windowed)
REM and the public v4_blend labels, which score 0.8927 on gold against our own
REM blend's 0.8463. Our labeller adds nothing on top of v4 -- every rank blend
REM including it scored at or below v4 alone -- so v4 is used straight.
REM
REM --encoder-chunk 16 is what makes this backbone fit at all: unchunked it peaks
REM at 15.07 GB on a 10.7 GB card and thrashes at 50.8 s/step; chunked it is
REM 4.84 GB and 1.88 s/step.
setlocal
set PY=D:\kaggle_c\.venv\Scripts\python.exe
set REPO=D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection
set HF_HOME=D:\kaggle_c\hf
set PYTHONUNBUFFERED=1
cd /d "%REPO%"
"%PY%" -u scripts\train.py --fold 0 --folds 5 --epochs 8 --batch 1 --accum 16 ^
    --workers 4 --backbone coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --size 384 ^
    --lr 1e-3 --backbone-lr-mult 0.03 --grad-checkpointing --encoder-chunk 16 ^
    --out D:\kaggle_c\work\coat >> D:\kaggle_c\work\coat.log 2>&1
endlocal
