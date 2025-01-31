#!/bin/bash
# 激活虚拟环境
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# 启动API服务
cd src/api && uvicorn server:app --host 0.0.0.0 --port 8000 --reload 