import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
import argparse
from src.chat.model_config import SUPPORTED_MODELS

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
                        try:
                            data = json.loads(line.decode('utf-8').replace('data: ', ''))
                            output += data.get("data", "")
                            print(data.get("data", ""), end="", flush=True)
                        except json.JSONDecodeError:
                            continue
                        
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
                print(output)
                
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
            
    def test_model_cases(self, model_name: str, stream: bool = False) -> List[Dict]:
        """测试指定模型的所有测试用例"""
        if model_name not in SUPPORTED_MODELS:
            return [{"status": "error", "error": f"不支持的模型: {model_name}"}]
            
        results = []
        model_config = SUPPORTED_MODELS[model_name]
        
        print(f"\n=== 测试模型: {model_name} ===")
        print(f"测试用例数量: {len(model_config.test_cases)}")
        
        for i, test_case in enumerate(model_config.test_cases, 1):
            print(f"\n--- 测试用例 {i}: {test_case['name']} ---")
            print(f"提示: {test_case['prompt'][:100]}...")
            print("\n回答:")
            
            result = self.test_inference(test_case['prompt'], model_name, stream)
            results.append({
                "case_name": test_case['name'],
                **result
            })
            
            print(f"\n耗时: {result['time_taken']:.2f}秒")
            print("-" * 80)
            
        return results
            
    def run_concurrent_tests(self, model_name: str, num_requests: int = 5) -> List[Dict]:
        """并发测试"""
        if model_name not in SUPPORTED_MODELS:
            return [{"status": "error", "error": f"不支持的模型: {model_name}"}]
            
        model_config = SUPPORTED_MODELS[model_name]
        test_cases = model_config.test_cases[:num_requests]
        
        def single_test(test_case):
            return self.test_inference(test_case['prompt'], model_name, stream=False)
            
        results = []
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(single_test, test_case) for test_case in test_cases]
            for future in futures:
                results.append(future.result())
                
        return results

def main():
    parser = argparse.ArgumentParser(description="测试推理服务器")
    parser.add_argument("--url", default="http://localhost:8000", help="服务器URL")
    parser.add_argument("--concurrent", type=int, default=5, help="并发请求数")
    parser.add_argument("--model", choices=list(SUPPORTED_MODELS.keys()), help="指定要测试的模型")
    parser.add_argument("--stream", action="store_true", help="使用流式输出")
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
        
    # 如果指定了模型，测试该模型的所有用例
    if args.model:
        results = tester.test_model_cases(args.model, args.stream)
        success_count = sum(1 for r in results if r['status'] == 'success')
        avg_time = sum(r['time_taken'] for r in results if r['status'] == 'success') / max(success_count, 1)
        
        print(f"\n=== {args.model} 测试总结 ===")
        print(f"成功用例: {success_count}/{len(results)}")
        print(f"平均响应时间: {avg_time:.3f}秒")
        
        # 打印错误信息
        errors = [(r['case_name'], r['error']) for r in results if r['status'] == 'error']
        if errors:
            print("\n错误详情:")
            for case_name, error in errors:
                print(f"{case_name}: {error}")
    
    # 并发测试
    if args.concurrent > 0:
        model_to_test = args.model or "deepseek-qwen-14b"
        print(f"\n=== 并发测试 ({args.concurrent}个请求) ===")
        results = tester.run_concurrent_tests(model_to_test, args.concurrent)
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