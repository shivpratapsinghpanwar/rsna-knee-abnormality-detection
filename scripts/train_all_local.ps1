# Train every fold locally, detached from any editor session.
#
# Kaggle's 30 h weekly GPU quota is spent, but nothing here needs Kaggle any
# more: the 384 mosaic corpus and the blended labels are both on local disk, and
# the RTX 3080 runs ~13 min/epoch against the T4's 25.
#
# Launch it so it outlives the shell that started it:
#   Start-Process powershell -ArgumentList '-File scripts\train_all_local.ps1' `
#       -WindowStyle Hidden
#
# A previous attempt ran as a session-bound background task and was killed the
# moment the session ended, after producing zero epochs. Detaching is the point.

$ErrorActionPreference = 'Continue'
$py   = 'D:\kaggle_c\.venv\Scripts\python.exe'
$repo = 'D:\Shivpratap_Singh_Panwar\Synthetic\kagg;le_comp\rsna-knee-abnormality-detection'
$out  = 'D:\kaggle_c\work\models384'
$log  = 'D:\kaggle_c\work\train_local.log'

New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location -LiteralPath $repo

"=== started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8

foreach ($fold in 0, 1, 2, 3, 4) {
    # Skip a fold whose checkpoint already exists, so a restart resumes the
    # sweep instead of redoing hours of finished work.
    $ckpt = Join-Path $out ("fold{0}_resnet34.pt" -f $fold)
    if (Test-Path $ckpt) {
        "=== fold $fold already done, skipping ===" | Out-File -FilePath $log -Append -Encoding utf8
        continue
    }
    "=== fold $fold  $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
    & $py -u scripts\train.py --fold $fold --folds 5 --epochs 8 `
        --batch 1 --accum 16 --workers 4 --out $out *>> $log
}

"=== all folds done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
