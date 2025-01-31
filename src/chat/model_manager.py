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
            # 获取设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 设置环境变量以禁用警告
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.path,
                use_fast=config.use_fast_tokenizer,
                trust_remote_code=config.trust_remote_code,
                cache_dir=MODELS_DIR / model_name,
                local_files_only=False,
                force_download=True
            )
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                config.path,
                torch_dtype=torch.float32,  # 强制使用float32
                trust_remote_code=config.trust_remote_code,
                cache_dir=MODELS_DIR / model_name,
                local_files_only=False,
                force_download=True,
                **config.model_kwargs
            ).to(device)
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.current_model = model_name
            return True
            
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {str(e)}")
            return False
            
    def generate_stream(self, prompt: str, model_name: Optional[str] = None) -> Iterator[StreamOutput]:
        """流式生成文本"""
        if model_name and model_name != self.current_model:
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