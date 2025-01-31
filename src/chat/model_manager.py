from typing import Optional, Iterator, Dict
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model_config import ModelConfig, SUPPORTED_MODELS
from .stream_handler import BaseStreamHandler, StreamOutput
import os

# 设置模型缓存目录
MODELS_DIR = Path(__file__).parent.parent / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.current_model = None
        self.stream_handler = BaseStreamHandler()
        
        # CUDA设置
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32
            torch.backends.cudnn.benchmark = False  # 禁用cuDNN基准测试
            torch.backends.cudnn.deterministic = True  # 使用确定性算法
            torch.cuda.empty_cache()  # 清理GPU缓存
        
        self.device = self._setup_device()
        
    def _setup_device(self):
        """设置并返回最佳可用设备"""
        if torch.cuda.is_available():
            # 设置CUDA设备
            device = torch.device("cuda")
            # 设置CUDA相关环境变量
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            # 清理GPU缓存
            torch.cuda.empty_cache()
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            print("使用CPU模式")
        return device
        
    def load_model(self, model_name: str) -> bool:
        """加载指定的模型"""
        if model_name not in SUPPORTED_MODELS:
            print(f"不支持的模型: {model_name}")
            return False
            
        if model_name in self.models:
            self.current_model = model_name
            return True
            
        config = SUPPORTED_MODELS[model_name]
        try:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.path,
                use_fast=config.use_fast_tokenizer,
                trust_remote_code=config.trust_remote_code,
                cache_dir=MODELS_DIR / model_name,
            )
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载模型
            print(f"正在加载模型到{self.device}...")
            model = AutoModelForCausalLM.from_pretrained(
                config.path,
                trust_remote_code=config.trust_remote_code,
                cache_dir=MODELS_DIR / model_name,
                **config.model_kwargs
            ).to(self.device)
            
            # 设置为评估模式
            model.eval()
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.current_model = model_name
            return True
            
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {str(e)}")
            return False
            
    def generate_stream(self, prompt: str, model_name: Optional[str] = None) -> Iterator[StreamOutput]:
        """流式生成文本"""
        print(f"开始生成文本，模型: {model_name}")  # 调试信息
        
        if model_name and model_name != self.current_model:
            print(f"需要加载新模型: {model_name}")  # 调试信息
            if not self.load_model(model_name):
                yield StreamOutput(text=f"加载模型 {model_name} 失败", finished=True)
                return
                
        if not self.current_model:
            yield StreamOutput(text="未加载任何模型", finished=True)
            return
            
        model = self.models[self.current_model]
        tokenizer = self.tokenizers[self.current_model]
        config = SUPPORTED_MODELS[self.current_model]
        
        try:
            print(f"编码输入文本: {prompt[:50]}...")
            
            # 根据不同模型添加特定的提示模板
            if "chatglm" in config.path.lower():
                formatted_prompt = f"[INST] {prompt} [/INST]"
            elif "gpt2" in config.path.lower():
                formatted_prompt = f"问题：{prompt}\n答案："
            else:
                formatted_prompt = prompt
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(model.device)
            
            print("使用标准生成方式")
            outputs = model.generate(
                **inputs,
                max_length=config.max_length,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                num_return_sequences=1,
                num_beams=1,                  # 使用贪婪搜索
                no_repeat_ngram_size=3,       # 避免重复
                repetition_penalty=1.2,       # 重复惩罚
                length_penalty=1.0,           # 长度惩罚
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回答部分
            if "[/INST]" in generated_text:
                text = generated_text.split("[/INST]")[1].strip()
            elif "答案：" in generated_text:
                text = generated_text.split("答案：")[1].strip()
            else:
                text = generated_text.strip()
            
            yield self.stream_handler.handle_text(text)
            
            yield self.stream_handler.finish()
            
        except Exception as e:
            print(f"生成过程中发生错误: {str(e)}")  # 调试信息
            yield StreamOutput(text=f"生成失败: {str(e)}", finished=True)

    def measure_inference_time(self, model_name: str, prompt: str, num_runs: int = 3) -> Dict:
        """测量模型推断时间"""
        if not self.current_model or self.current_model != model_name:
            if not self.load_model(model_name):
                return {
                    "avg": -1,
                    "min": -1,
                    "max": -1,
                    "tokens_per_second": 0,
                    "avg_tokens": 0,
                    "sample_output": "加载失败"
                }
                
        if not self.current_model:
            return {
                "avg": -1,
                "min": -1,
                "max": -1,
                "tokens_per_second": 0,
                "avg_tokens": 0,
                "sample_output": "未加载任何模型"
            }
            
        model = self.models[self.current_model]
        tokenizer = self.tokenizers[self.current_model]
        config = SUPPORTED_MODELS[self.current_model]
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 使用模型的stream_generate方法（如果有）
            if hasattr(model, "stream_generate"):
                for output in model.stream_generate(
                    **inputs,
                    max_length=config.max_length,
                    temperature=config.temperature,
                    top_p=config.top_p
                ):
                    yield self.stream_handler.handle_text(output)
            else:
                # 标准生成方式
                outputs = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    streaming=True
                )
                
                for output in outputs:
                    text = tokenizer.decode(output, skip_special_tokens=True)
                    yield self.stream_handler.handle_text(text)
                    
            yield self.stream_handler.finish()
            
        except Exception as e:
            return {
                "avg": -1,
                "min": -1,
                "max": -1,
                "tokens_per_second": 0,
                "avg_tokens": 0,
                "sample_output": f"生成失败: {str(e)}"
            } 