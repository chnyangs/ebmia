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