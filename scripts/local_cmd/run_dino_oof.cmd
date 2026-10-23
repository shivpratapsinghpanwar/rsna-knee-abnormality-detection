@echo off
cd /d "D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection"
D:\kaggle_c\.venv\Scripts\python.exe -u scripts\oof.py --fold-file D:\kaggle_c\work\oof2\fold_cloud.csv --ckpt D:\kaggle_c\work\dinol_gcp\fold4_vit_large_patch14_dinov2.lvd142m.pt --out D:\kaggle_c\work\oof2\dinol_f4.csv > D:\kaggle_c\work\oof2\dinol_f4.log 2>&1
echo done > D:\kaggle_c\work\oof2\DINO_DONE
