from src.chat.model_manager import ModelManager

def main():
    # 初始化模型管理器
    manager = ModelManager()
    
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
    
    # 使用deepseek-1.5b模型
    model_name = "deepseek-1.5b"
    
    print(f"\n=== 使用 {model_name} 模型进行推断 ===\n")
    
    # 对每个提示进行推断
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- 测试用例 {i} ---")
        print(f"提示: {prompt}\n")
        print("回答: ", end="", flush=True)
        
        try:
            # 使用流式输出生成回答
            for output in manager.generate_stream(prompt, model_name):
                if not output.finished:
                    print(output.text, end="", flush=True)
                if output.finished:
                    print("\n")  # 输出完成后换行
                    
        except Exception as e:
            print(f"\n生成过程中发生错误: {str(e)}")
        
        print("-" * 80)  # 分隔线

if __name__ == "__main__":
    main() 