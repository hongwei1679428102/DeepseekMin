from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
import torch
import os
from typing import Tuple, Optional

# 设置模型缓存目录
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_model(model_name: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    加载指定的模型和分词器
    
    Args:
        model_name: 模型名称/路径
        
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    print(f"正在加载模型: {model_name}")
    
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,  # 使用Python实现的tokenizer而不是Fast版本
            cache_dir=os.path.join(MODELS_DIR, model_name.split('/')[-1]),  # 设置缓存目录
        )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Python 3.6环境下使用float32而不是bfloat16
            trust_remote_code=True,
            cache_dir=os.path.join(MODELS_DIR, model_name.split('/')[-1]),  # 设置缓存目录
        )
        
        # 手动将模型移到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return model, tokenizer
    except Exception as e:
        print(f"加载模型时发生错误: {str(e)}")
        return None, None

def generate_text(model, tokenizer, prompt: str, 
                 max_length: int = 512,
                 temperature: float = 0.6,
                 top_p: float = 0.95) -> str:
    """
    使用模型生成文本
    
    Args:
        model: 加载的模型
        tokenizer: 加载的分词器
        prompt: 输入提示
        max_length: 最大生成长度
        temperature: 温度参数(0.5-0.7)
        top_p: top-p采样参数
        
    Returns:
        str: 生成的文本
    """
    try:
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成文本
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"生成文本时发生错误: {str(e)}")
        return "生成文本失败"

def main():
    # 模型名称
    models = [
        "gpt2",  # 使用更简单的模型
    ]
    
    # 测试提示
    test_prompt = "请解释什么是Python的装饰器模式?"
    
    for model_name in models:
        try:
            # 加载模型
            model, tokenizer = load_model(model_name)
            
            if model is None or tokenizer is None:
                print(f"跳过模型 {model_name} 因为加载失败")
                continue
            
            print(f"\n使用模型 {model_name} 生成回答:")
            response = generate_text(model, tokenizer, test_prompt)
            print(f"回答: {response}")
            
            # 释放显存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"处理模型 {model_name} 时发生错误: {str(e)}")

if __name__ == "__main__":
    main() 