from src.chat.model_manager import ModelManager
import time
from typing import Dict, List
import torch

class ModelBenchmark:
    def __init__(self):
        self.manager = ModelManager()
        self.results: Dict[str, Dict] = {}
        
    def run_inference(self, model_name: str, prompt: str) -> Dict:
        """运行单次推断并返回结果"""
        start_time = time.time()
        output_text = ""
        token_count = 0
        
        try:
            for output in self.manager.generate_stream(prompt, model_name):
                if not output.finished:
                    output_text += output.text
                    token_count += 1
                    print(output.text, end="", flush=True)
            print()  # 换行
            
            inference_time = time.time() - start_time
            return {
                "time": inference_time,
                "tokens": token_count,
                "tokens_per_second": token_count / inference_time if inference_time > 0 else 0,
                "output": output_text
            }
            
        except Exception as e:
            print(f"推断失败: {str(e)}")
            return {
                "time": -1,
                "tokens": 0,
                "tokens_per_second": 0,
                "output": f"错误: {str(e)}"
            }

def main():
    # 初始化基准测试器
    benchmark = ModelBenchmark()
    
    # 测试提示列表
    prompts = [
        """请解释Python装饰器的工作原理和使用场景，包括：
1. 什么是装饰器
2. 装饰器的语法
3. 常见使用场景
4. 一个实际的例子""",
        
        """请帮我优化以下Python代码，并解释优化原理：
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
        
        """请解释Python的上下文管理器（with语句），包括：
1. 基本概念
2. 实现原理
3. 常见应用场景
4. 代码示例"""
    ]
    
    # 所有要测试的模型
    models = ["gpt2", "deepseek-1.5b", "deepseek-llama-8b"]
    
    # 存储所有结果
    results = {}
    
    print("\n=== 模型性能对比测试 ===")
    print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"测试用例数量: {len(prompts)}")
    print("-" * 80)
    
    # 对每个模型进行测试
    for model_name in models:
        print(f"\n正在测试模型: {model_name}")
        model_results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- 测试用例 {i} ---")
            print(f"提示: {prompt}\n")
            print("回答: ", end="", flush=True)
            
            result = benchmark.run_inference(model_name, prompt)
            model_results.append(result)
            
            print(f"\n推断时间: {result['time']:.2f} 秒")
            print(f"生成速度: {result['tokens_per_second']:.2f} tokens/s")
            print("-" * 80)
            
        # 计算平均性能
        valid_results = [r for r in model_results if r["time"] > 0]
        if valid_results:
            avg_time = sum(r["time"] for r in valid_results) / len(valid_results)
            avg_speed = sum(r["tokens_per_second"] for r in valid_results) / len(valid_results)
        else:
            avg_time = -1
            avg_speed = 0
            
        results[model_name] = {
            "avg_time": avg_time,
            "avg_speed": avg_speed,
            "success_rate": len(valid_results) / len(prompts)
        }
    
    # 打印性能对比总结
    print("\n=== 性能对比总结 ===")
    print(f"{'模型名称':<20} {'平均推断时间(秒)':<20} {'平均速度(tokens/s)':<20} {'成功率':<10}")
    print("-" * 70)
    
    for model_name, result in results.items():
        if result["avg_time"] > 0:
            print(f"{model_name:<20} {result['avg_time']:>18.2f} {result['avg_speed']:>18.2f} {result['success_rate']:>9.0%}")
        else:
            print(f"{model_name:<20} {'Failed':>18} {'N/A':>18} {result['success_rate']:>9.0%}")

if __name__ == "__main__":
    main() 