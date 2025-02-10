import torch
import subprocess
import sys


def install(package, file_path=None):
    if file_path is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-f", file_path])


def format_pytorch_version(version):
    return version.split('+')[0]


TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)


def format_cuda_version(version):
    return 'cu' + version.replace('.', '')


CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

install(f'torch-scatter', f'https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html')
install(f'torch-sparse', f'https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html')
install(f'torch-cluster', f'https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html')
install(f'torch-spline-conv', f'https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html')
install(f'torch-geometric', f'https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html')
install(f'torchmetrics')
install('ogb')
install('networkx==3.1')
