@echo off
cd /d D:\kaggle_c\work
set PYTHONIOENCODING=utf-8
D:\kaggle_c\.venv\Scripts\python.exe -u D:\kaggle_c\work\tools\probe_chain.py "rsna-knee-958-r7-05=probe: 0.958 stack + R7 at 0.5 of the compact slot" "rsna-knee-958-r7arm-w05=probe: 0.958 stack + R7 0.4 + our CoAtNet 5-fold arm at 0.05=arm" "rsna-knee-958-r7-a224-05=probe: 0.958 stack + R7 0.4, 224 model at 0.5 of the Raptor slot" "rsna-knee-958-r7-wr045=probe: 0.958 stack + R7 0.4, default Raptor weight 0.45" > D:\kaggle_c\work\probe_chain.log 2>&1
