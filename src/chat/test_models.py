import sys
import os
from typing import Optional
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.chat.model_manager import ModelManager
from src.chat.model_config import SUPPORTED_MODELS

def run_test_case(manager: ModelManager, model_name: str, test_case: dict) -> None:
    """运行单个测试案例"""
    print(f"\n=== 测试案例: {test_case['name']} ===")
    print(f"提示: {test_case['prompt']}")
    print("回答: ", end="", flush=True)
    
    for output in manager.generate_stream(test_case['prompt'], model_name):
        if not output.finished:
            print(output.text, end="", flush=True)
        if output.finished:
            print()  # 换行

def test_model(manager: ModelManager, model_name: str) -> None:
    """测试指定模型的所有测试案例"""
    if model_name not in SUPPORTED_MODELS:
        print(f"错误: 不支持的模型 {model_name}")
        return
        
    print(f"\n{'='*20} 测试模型: {model_name} {'='*20}")
    
    if not manager.load_model(model_name):
        print(f"错误: 加载模型 {model_name} 失败")
        return
        
    config = SUPPORTED_MODELS[model_name]
    for test_case in config.test_cases:
        try:
            run_test_case(manager, model_name, test_case)
        except Exception as e:
            print(f"测试案例 {test_case['name']} 失败: {str(e)}")

def main():
    """运行所有模型的测试"""
    manager = ModelManager()
    
    # 获取命令行参数中指定的模型名称（如果有）
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    if model_name:
        if model_name in SUPPORTED_MODELS:
            test_model(manager, model_name)
        else:
            print(f"错误: 不支持的模型 {model_name}")
    else:
        # 测试所有支持的模型
        for model_name in SUPPORTED_MODELS:
            test_model(manager, model_name)

if __name__ == "__main__":
    main() 