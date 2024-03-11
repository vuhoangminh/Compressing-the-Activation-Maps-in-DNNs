wget "https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh"
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b

# Create enviroment
conda create --solver=libmamba -y -n compress python=3.9 
conda activate compress

# Install all the dependencies
conda install --solver=libmamba -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install imgui==1.3.0 glfw==2.2.0 pyopengl==3.1.5 imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 click requests psutil
pip install scipy monai
pip install -U albumentations
pip install nilearn
pip install SimpleITK
pip install comet-ml
pip install matplotlib
pip install seaborn
pip install autoflake
pip install termcolor
pip install hyperopt
pip install torchsummary
pip install PyWavelets
pip install torch-dct
pip install tensorly
pip install vit-pytorch

git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
cd ..
rm -rf pytorch_wavelets

# Done
exit