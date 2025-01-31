from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch

@dataclass
class ModelConfig:
    name: str
    path: str
    max_length: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    use_fast_tokenizer: bool = False
    trust_remote_code: bool = True
    test_cases: List[Dict[str, Any]] = None
    
    # 模型特定配置
    model_kwargs: Dict = None
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.test_cases is None:
            self.test_cases = []

# 支持的模型配置
SUPPORTED_MODELS = {
    "gpt2": ModelConfig(
        name="gpt2",
        path="gpt2",
        max_length=1024,
        use_fast_tokenizer=False,
        temperature=0.7,
        top_p=0.9,
        test_cases=[
            {
                "name": "Python装饰器",
                "prompt": "请解释什么是Python的装饰器模式?",
            },
            {
                "name": "英文翻译",
                "prompt": "Translate to English: 人工智能正在改变世界。",
            },
            {
                "name": "代码生成",
                "prompt": "写一个Python函数来计算斐波那契数列。",
            }
        ]
    ),
    "deepseek-1.5b": ModelConfig(
        name="deepseek-1.5b",
        path="IDEA-CCNL/Wenzhong-GPT2-110M",
        max_length=1024,
        use_fast_tokenizer=False,
        temperature=0.9,
        top_p=0.95,
        test_cases=[
            {
                "name": "中文问答",
                "prompt": "解释一下量子计算机的基本原理。",
            },
            {
                "name": "代码解释",
                "prompt": "解释以下代码的作用：\n```python\n@property\ndef name(self):\n    return self._name\n```",
            },
            {
                "name": "多轮对话",
                "prompt": "你是一个Python专家，请帮我优化以下代码性能：\n```python\nresult = []\nfor i in range(1000000):\n    if i % 2 == 0:\n        result.append(i)\n```",
            }
        ],
        model_kwargs={
            "low_cpu_mem_usage": False,
            "return_dict": False
        }
    ),
    "deepseek-llama-8b": ModelConfig(
        name="deepseek-llama-8b",
        path="facebook/opt-1.3b",
        max_length=2048,
        temperature=0.6,
        top_p=0.95,
        use_fast_tokenizer=False,
        test_cases=[
            {
                "name": "数学推理",
                "prompt": "Please reason step by step, and put your final answer within \\boxed{}.\nSolve the equation: 3x + 5 = 14",
            },
            {
                "name": "代码优化",
                "prompt": "Optimize this Python code and explain the improvements:\n```python\ndef fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```",
            },
            {
                "name": "中英互译",
                "prompt": "Please translate the following text to Chinese:\nQuantum computing leverages the principles of quantum mechanics to process information in fundamentally new ways.",
            },
            {
                "name": "算法设计",
                "prompt": "Design an efficient algorithm to find the longest palindromic substring in a given string. Please provide the solution in Python with detailed explanations.",
            }
        ],
        model_kwargs={
            "revision": "main",
        }
    ),
    "deepseek-qwen-14b": ModelConfig(
        name="deepseek-qwen-14b",
        path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        max_length=4096,
        temperature=0.7,
        top_p=0.9,
        use_fast_tokenizer=True,
        trust_remote_code=True,
        test_cases=[
            {
                "name": "代码生成",
                "prompt": "写一个Python函数，实现快速排序算法，并详细解释代码。"
            },
            {
                "name": "中文问答",
                "prompt": "详细解释一下大语言模型的工作原理，包括Transformer架构和自注意力机制。"
            },
            {
                "name": "数学推理",
                "prompt": "如何求解一个二次方程？请给出具体步骤和示例。"
            },
            {
                "name": "多轮对话",
                "prompt": "我想学习Python，应该从哪里开始？需要掌握哪些基础知识？"
            },
            {
                "name": "代码优化与重构",
                "prompt": """请分析并优化以下Python代码，包括：
1. 性能优化
2. 代码重构
3. 添加类型提示
4. 改进错误处理
5. 添加单元测试

```python
def process_data(data):
    result = []
    for item in data:
        try:
            # 处理数字
            if isinstance(item, (int, float)):
                result.append(item * 2)
            # 处理字符串
            elif isinstance(item, str):
                if item.isdigit():
                    result.append(int(item))
                else:
                    result.append(len(item))
            # 处理列表
            elif isinstance(item, list):
                temp = []
                for x in item:
                    if isinstance(x, (int, float)):
                        temp.append(x * 2)
                    elif isinstance(x, str):
                        temp.append(len(x))
                result.append(sum(temp))
        except:
            result.append(0)
    return result

# 使用示例
data = [1, "23", [1, 2, "abc"], "hello", 2.5]
print(process_data(data))
```"""
            }
        ],
        model_kwargs={
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "pad_token_id": 151643,
            "eos_token_id": 151643,
            "use_cache": True,
            "revision": "main",
            "use_flash_attention_2": True,
            "attn_implementation": "flash_attention_2"
        }
    ),
} 