@echo off
REM GPU setup using Conda
echo Creating conda environment...
call conda create -n localcopilot python=3.11 -y
call conda activate localcopilot
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
pause
