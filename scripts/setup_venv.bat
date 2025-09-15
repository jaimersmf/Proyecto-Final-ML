@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name nombre_proyecto_final_ML --display-name "Python (nombre_proyecto_final_ML)"
echo ✅ Entorno creado. Actívalo con: .venv\Scripts\activate
