@echo off
cd /d "D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection"
set PY=D:\kaggle_c\.venv\Scripts\python.exe -u scripts\oof.py --fold-file D:\kaggle_c\work\oof2\fold_cloud.csv
set O=D:\kaggle_c\work\oof2
if not exist %O%\dinol_f4_h16w.csv %PY% --half-weights --ckpt D:\kaggle_c\work\dinol_gcp\fold4_vit_large_patch14_dinov2.lvd142m.pt --out %O%\dinol_f4_h16w.csv > %O%\dinol_f4_h16w.log 2>&1
if not exist %O%\dinol_f4_fp32.csv %PY% --fp32 --batch 1 --ckpt D:\kaggle_c\work\dinol_gcp\fold4_vit_large_patch14_dinov2.lvd142m.pt --out %O%\dinol_f4_fp32.csv > %O%\dinol_f4_fp32.log 2>&1
if not exist %O%\coat_f4_fp32.csv %PY% --fp32 --batch 1 --ckpt D:\kaggle_c\work\arm_weights\fold4_coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k.pt --out %O%\coat_f4_fp32.csv > %O%\coat_f4_fp32.log 2>&1
echo done > %O%\PREC_DONE
