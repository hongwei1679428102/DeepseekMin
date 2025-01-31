from typing import Iterator, Optional
from dataclasses import dataclass

@dataclass
class StreamOutput:
    text: str
    finished: bool = False
    
class BaseStreamHandler:
    def __init__(self):
        self.text = ""
        
    def handle_text(self, text: str) -> StreamOutput:
        """处理生成的文本"""
        print(f"处理文本: {text[:50]}...")  # 调试信息
        self.text += text
        return StreamOutput(text=text, finished=False)
        
    def finish(self) -> StreamOutput:
        """结束生成"""
        print("生成完成")  # 调试信息
        return StreamOutput(text="", finished=True) 