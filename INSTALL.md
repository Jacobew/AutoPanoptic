## Installation

### Environments:
- PyTorch 1.3.0
- torchvision 
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA 10.0.13
- Python 3.6



### Linux: Step-by-step installation

```bash

conda create --name autopanoptic python=3.6
conda activate autopanoptic

conda install ipython pip

pip install ninja yacs cython matplotlib tqdm opencv-python

conda install pytorch=1.3.0 torchvision 

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/Jacobew/AutoPanoptic
cd AutoPanoptic/maskrcnn-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

cd maskrcnn_benchmark/pytorch_distributed_syncbn/lib/gpu
python setup.py install

