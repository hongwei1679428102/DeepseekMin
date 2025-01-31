from typing import Iterator, Optional
from dataclasses import dataclass

@dataclass
class StreamOutput:
    text: str
    finished: bool = False
    
class BaseStreamHandler:
    def __init__(self):
        self.text = ""
        
    def handle_text(self, new_text: str) -> StreamOutput:
        """处理新的文本片段"""
        self.text += new_text
        return StreamOutput(text=new_text, finished=False)
        
    def finish(self) -> StreamOutput:
        """完成流式输出"""
        return StreamOutput(text="", finished=True) 