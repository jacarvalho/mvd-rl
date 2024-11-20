git submodule update --init --recursive --progress

eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"

conda env create -f environment.yml

conda activate mvd-rl

conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

# Install PyTorch with CUDA 11.8
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install torch==2.2.0 torchvision==0.17.0  --index-url https://download.pytorch.org/whl/cu118

conda env config vars set CUDA_HOME=""
conda activate mvd-rl

conda install -c conda-forge suitesparse -y

conda install anaconda::swig -y

conda activate mvd-rl
cd deps && cd experiment_launcher && pip install --no-use-pep517 -e .[all] && cd ..
#cd mushroom-rl && pip install --no-use-pep517 -e .[all] && cd .. && cd ..
cd mushroom-rl && pip install --no-use-pep517 -e .[gym,bullet,plots] && cd .. && cd ..

pip install -e .

pip install gym==0.25.1
pip install gymnasium

conda install pinocchio -c conda-forge -y  # pinocchio is not available on windows!

pip install einops

pip install imageio

pip install "cython<=0.29.33"

pip install mujoco-py
pip install mujoco

conda install -c conda-forge glew -y
conda install -c conda-forge mesalib -y
conda install -c menpo glfw3 -y
conda install conda-forge::patchelf -y