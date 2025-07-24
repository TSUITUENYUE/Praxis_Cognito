conda env remove -n Praxis_Cognito
conda create -n Praxis_Cognito python=3.12
conda activate Praxis_Cognito

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 #torch 2.7.1 + cuda 12.8

pip install -r requirements.txt