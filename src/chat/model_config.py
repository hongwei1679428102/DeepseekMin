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
        path="uer/gpt2-chinese-cluecorpussmall",
        max_length=2048,
        use_fast_tokenizer=False,
        temperature=0.7,
        top_p=0.9,
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
        model_kwargs={}
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
} 