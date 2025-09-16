bash /mnt/bn/yufei1900/maojianbo/workspace/Miniforge3-Linux-x86_64.sh  -b -u -p ~/miniforge3
source ~/miniforge3/bin/activate
conda create -n areal python=3.12
conda activate areal
bash examples/env/setup-pip-deps.sh
pip install qwen_agent -y
pip uninstall wandb -y
pip install -U byted-wandb -i https://bytedpypi.byted.org/simple