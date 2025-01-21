@echo off
call venv\Scripts\activate
python check.py --vid 001 -cid 01
deactivate
pause
