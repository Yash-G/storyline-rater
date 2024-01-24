#CONDA_BASE=$(conda info --base)
#source "$CONDA_BASE"/etc/profile.d/conda.sh
#conda create --name AMLProject -y
#conda activate AMLProject
conda install python=3.11 -y
pip install beautifulsoup4
pip install -U scikit-learn
conda install -c conda-forge pandas
python -m pip install -U pip
python -m pip install -U matplotlib
#conda deactivate
#conda create --name AMLProjectPy3_8 -y
#conda activate AMLProjectPy3_8
#conda install python=3.8 -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers datasets wandb
pip install accelerate -U
wandb login
conda deactivate