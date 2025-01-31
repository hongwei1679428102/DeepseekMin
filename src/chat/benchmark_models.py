import sys
import os
import time
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import torch

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.chat.model_manager import ModelManager
from src.chat.model_config import SUPPORTED_MODELS

class ModelBenchmark:
    def __init__(self):
        self.manager = ModelManager()
        self.results: Dict[str, Dict] = {}
        
    def measure_loading_time(self, model_name: str) -> float:
        """测量模型加载时间"""
        start_time = time.time()
        success = self.manager.load_model(model_name)
        loading_time = time.time() - start_time
        
        if not success:
            print(f"错误: 加载模型 {model_name} 失败")
            return -1
            
        return loading_time
        
    def measure_inference_time(self, model_name: str, prompt: str, num_runs: int = 3) -> Dict:
        """测量模型推断时间"""
        if not self.manager.current_model or self.manager.current_model != model_name:
            if not self.manager.load_model(model_name):
                return {"avg": -1, "min": -1, "max": -1}
                
        times = []
        token_counts = []
        outputs = []
        
        for i in range(num_runs):
            start_time = time.time()
            output_text = ""
            token_count = 0
            
            for output in self.manager.generate_stream(prompt):
                if not output.finished:
                    output_text += output.text
                    token_count += 1
                    
            inference_time = time.time() - start_time
            times.append(inference_time)
            token_counts.append(token_count)
            outputs.append(output_text)
            
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 等待一下以避免GPU过热
            time.sleep(1)
            
        return {
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "tokens_per_second": sum(token_counts) / sum(times),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "sample_output": outputs[0][:100] + "..."  # 保存部分输出示例
        }
        
    def run_benchmarks(self):
        """运行所有模型的基准测试"""
        test_prompt = "Explain the concept of recursion in programming and provide a simple example."
        
        print("\n=== 开始模型基准测试 ===")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"测试提示: {test_prompt}")
        
        for model_name in SUPPORTED_MODELS:
            print(f"\n正在测试模型: {model_name}")
            
            # 测量加载时间
            loading_time = self.measure_loading_time(model_name)
            
            # 测量推断时间
            inference_results = self.measure_inference_time(model_name, test_prompt)
            
            # 保存结果
            self.results[model_name] = {
                "loading_time": loading_time,
                "inference_time": inference_results
            }
            
            # 打印结果
            print(f"加载时间: {loading_time:.2f} 秒")
            print(f"平均推断时间: {inference_results['avg']:.2f} 秒")
            print(f"每秒生成令牌数: {inference_results['tokens_per_second']:.2f}")
            print(f"输出示例: {inference_results['sample_output']}")
            
        self.print_summary()
        
    def print_summary(self):
        """打印测试结果总结"""
        print("\n=== 测试结果总结 ===")
        print("\n模型性能对比:")
        print(f"{'模型名称':<20} {'加载时间(秒)':<15} {'推断时间(秒)':<15} {'令牌/秒':<15}")
        print("-" * 65)
        
        for model_name, results in self.results.items():
            loading_time = results["loading_time"]
            inference = results["inference_time"]
            print(
                f"{model_name:<20} "
                f"{loading_time:>14.2f} "
                f"{inference['avg']:>14.2f} "
                f"{inference['tokens_per_second']:>14.2f}"
            )

def main():
    benchmark = ModelBenchmark()
    benchmark.run_benchmarks()

if __name__ == "__main__":
    main() 