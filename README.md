# An efficient way to apply Blind Membership Inference Attack

//TODOS

## Training Target Model

    --dataset
    --model
    --batch (optional, default 64)
    --config
    python train_target_model.py --dataset DD --model GCN --config config/graph_classification.json


## Issues
    1. tensorflow.python.framework.errors_impl.InternalError: cudaGetDevice() failed. Status: cudaGetErrorString symbol not found
    Solved: https://developer.nvidia.com/cuda-10.1-download-archive-base to install cuda10.1 toolkit

    2. dgl._ffi.base.DGLError: /opt/dgl/src/runtime/c_runtime_api.cc:87: Check failed: allow_missing: Device API gpu is not enabled 
    https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=runfilelocal
    Reinstall dgl==0.4.2 by pip and dgl-cuda10.2 by conda with -c dglteam


## Dataset Download

    https://chrsmrrs.github.io/datasets/docs/datasets/
    