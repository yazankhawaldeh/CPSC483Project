This is our CPSC 483 LLM Project which is a historical AI Bot that gives information about WW2.

IMPORTANT:

If you are using a Windows based machine, then you need to use the ww2_inference_windows.py file because Windows does not support bitsandbytes.

STEPS TO INSTALL AND RUN:

For WINDOWS BASED MACHINES:

git clone https://github.com/yazankhawaldeh/CPSC483Project.git
cd CPSC483Project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch transformers peft accelerate datasets
IF WINDOWS THROWS AN EXCEPTION FOR SCRIPT EXECUTION RUN THIS COMMAND:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python ww2_inference_windows.py

FOR MACOS/LINXUS BASED MACHINES:

git clone https://github.com/yazankhawaldeh/CPSC483Project.git
cd CPSC483Project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers peft accelerate datasets
python inference_chat.py
