from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
import torch
import os
from typing import Tuple, Optional
import gc
from pathlib import Path
from src.chat.model_manager import ModelManager
from src.chat.model_config import SUPPORTED_MODELS

# 设置项目根目录和模型缓存目录
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "src" / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def get_device():
    """
    获取可用的设备，并打印设备信息
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("使用CPU模式")
    return device

def main():
    # 模型名称
    models = [
        "gpt2",
        "deepseek-1.5b",  # 添加新模型
    ]
    
    # 测试提示
    test_prompt = "请解释什么是Python的装饰器模式?"
    
    # 使用ModelManager
    manager = ModelManager()
    
    for model_name in models:
        try:
            print(f"\n使用模型 {model_name} 生成回答:")
            
            # 使用流式输出
            for output in manager.generate_stream(test_prompt, model_name):
                if not output.finished:
                    print(output.text, end="", flush=True)
                if output.finished:
                    print()  # 换行
            
            # 释放显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"处理模型 {model_name} 时发生错误: {str(e)}")

if __name__ == "__main__":
    main() 