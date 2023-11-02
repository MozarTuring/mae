cd /home/maojingwei/project/mae
if [ ! -d "pyenv" ]; then
    /usr/local/python3.8.16/bin/python3 -m virtualenv pyenv
fi
source pyenv/bin/activate

#python -m pip install torch==1.7.0 -f https://download.pytorch.org/whl/torch_stable.html
#python -m pip uninstall torch
#python -m pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
#python -m pip install tensorboard
#python -m pip install timm==0.3.2
#python -m pip install torchvision==0.8.1
#python -m pip install ipdb
#python -m pip install scipy
#python -m pip install flask
