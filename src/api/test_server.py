import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
import argparse

class ServerTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        
    def test_models_endpoint(self) -> Dict:
        """测试模型列表接口"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/models")
            time_taken = time.time() - start_time
            
            return {
                "status": "success",
                "time_taken": time_taken,
                "models": response.json()["models"],
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
            
    def test_inference(self, prompt: str, model_name: str, stream: bool = False) -> Dict:
        """测试推理接口"""
        try:
            start_time = time.time()
            
            if stream:
                response = requests.post(
                    f"{self.base_url}/infer/stream",
                    json={
                        "prompt": prompt,
                        "model_name": model_name,
                        "stream": True
                    },
                    stream=True
                )
                
                output = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8').replace('data: ', ''))
                        output += data.get("data", "")
                        
            else:
                response = requests.post(
                    f"{self.base_url}/infer",
                    json={
                        "prompt": prompt,
                        "model_name": model_name,
                        "stream": False
                    }
                )
                output = response.json()["text"]
                
            time_taken = time.time() - start_time
            
            return {
                "status": "success",
                "time_taken": time_taken,
                "output": output,
                "status_code": response.status_code
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
            
    def run_concurrent_tests(self, num_requests: int = 5) -> List[Dict]:
        """并发测试"""
        test_prompts = [
            "解释Python的装饰器模式",
            "什么是Python的生成器?",
            "解释Python的上下文管理器",
            "Python中的多线程和多进程有什么区别?",
            "如何在Python中处理异常?"
        ]
        
        def single_test(prompt: str):
            return self.test_inference(prompt, "deepseek-1.5b", stream=False)
            
        results = []
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(single_test, prompt) for prompt in test_prompts[:num_requests]]
            for future in futures:
                results.append(future.result())
                
        return results

def main():
    parser = argparse.ArgumentParser(description="测试推理服务器")
    parser.add_argument("--url", default="http://localhost:8000", help="服务器URL")
    parser.add_argument("--concurrent", type=int, default=5, help="并发请求数")
    args = parser.parse_args()
    
    tester = ServerTester(args.url)
    
    # 测试模型列表接口
    print("\n=== 测试模型列表接口 ===")
    result = tester.test_models_endpoint()
    print(f"状态: {result['status']}")
    if result['status'] == 'success':
        print(f"可用模型: {result['models']}")
        print(f"响应时间: {result['time_taken']:.3f}秒")
    else:
        print(f"错误: {result['error']}")
        
    # 测试同步推理
    print("\n=== 测试同步推理 ===")
    result = tester.test_inference(
        "请用简单的语言解释什么是Python装饰器",
        "deepseek-1.5b",
        stream=False
    )
    print(f"状态: {result['status']}")
    if result['status'] == 'success':
        print(f"输出: {result['output']}")
        print(f"响应时间: {result['time_taken']:.3f}秒")
    else:
        print(f"错误: {result['error']}")
        
    # 测试流式推理
    print("\n=== 测试流式推理 ===")
    result = tester.test_inference(
        "请解释Python的生成器函数",
        "deepseek-1.5b",
        stream=True
    )
    print(f"状态: {result['status']}")
    if result['status'] == 'success':
        print(f"输出: {result['output']}")
        print(f"响应时间: {result['time_taken']:.3f}秒")
    else:
        print(f"错误: {result['error']}")
        
    # 并发测试
    print(f"\n=== 并发测试 ({args.concurrent}个请求) ===")
    results = tester.run_concurrent_tests(args.concurrent)
    success_count = sum(1 for r in results if r['status'] == 'success')
    avg_time = sum(r['time_taken'] for r in results if r['status'] == 'success') / max(success_count, 1)
    
    print(f"成功请求: {success_count}/{len(results)}")
    print(f"平均响应时间: {avg_time:.3f}秒")
    
    # 打印错误信息
    errors = [(i, r['error']) for i, r in enumerate(results) if r['status'] == 'error']
    if errors:
        print("\n错误详情:")
        for i, error in errors:
            print(f"请求 {i+1}: {error}")

if __name__ == "__main__":
    main() 