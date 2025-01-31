from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name: str):
    """
    加载指定的模型和分词器
    
    Args:
        model_name: 模型名称/路径
        
    Returns:
        model: 加载的模型
        tokenizer: 加载的分词器
    """
    print(f"正在加载模型: {model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 使用bfloat16精度
        trust_remote_code=True,
        device_map="auto"  # 自动处理设备映射
    )
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt: str, 
                 max_length: int = 512,
                 temperature: float = 0.6,
                 top_p: float = 0.95):
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

def main():
    # 模型名称
    models = [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ]
    
    # 测试提示
    test_prompt = "请解释什么是Python的装饰器模式?"
    
    for model_name in models:
        try:
            # 加载模型
            model, tokenizer = load_model(model_name)
            
            print(f"\n使用模型 {model_name} 生成回答:")
            response = generate_text(model, tokenizer, test_prompt)
            print(f"回答: {response}")
            
            # 释放显存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"加载模型 {model_name} 时发生错误: {str(e)}")

if __name__ == "__main__":
    main() 