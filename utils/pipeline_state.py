from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class PipelineState(BaseModel):
    """流水线状态对象"""
    # 输入信息
    user_id: str
    original_query: str
    chat_history: List[Dict[str, str]] = []

    # 感知层结果
    intent: Optional[Dict[str, Any]] = None
    rewritten_query: Optional[str] = None

    # 信息抽取层结果
    extracted_params: Optional[Dict[str, Any]] = None

    # 执行层结果
    tool_results: Optional[Dict[str, Any]] = None
    rag_context: Optional[List[Dict[str, Any]]] = None

    # 输出结果
    final_answer: Optional[str] = None
    risk_info: Optional[Dict[str, Any]] = None

    # 状态标志
    is_success: bool = True
    error_message: Optional[str] = None

    def update(self, **kwargs):
        """更新状态"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def to_dict(self):
        """转换为字典"""
        return self.model_dump()