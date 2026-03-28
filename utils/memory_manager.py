from typing import List, Dict
from config import settings
from .data_loader import data_loader


class MemoryManager:
    """对话历史管理器"""

    @staticmethod
    def get_recent_history(chat_history: List[Dict[str, str]], max_length: int = None) -> List[Dict[str, str]]:
        """获取最近的对话历史"""
        if max_length is None:
            max_length = settings.MAX_CHAT_HISTORY
        return chat_history[-max_length:] if len(chat_history) > max_length else chat_history

    @staticmethod
    def save_history(chat_history: List[Dict[str, str]], user_id: str):
        """保存对话历史"""
        data_loader.save_chat_history(user_id, chat_history)

    @staticmethod
    def load_history(user_id: str) -> List[Dict[str, str]]:
        """加载对话历史"""
        return data_loader.get_chat_history(user_id)

    @staticmethod
    def add_message(chat_history: List[Dict[str, str]], role: str, content: str) -> List[Dict[str, str]]:
        """添加消息到对话历史"""
        chat_history.append({"role": role, "content": content})
        return chat_history

    @staticmethod
    def format_history(chat_history: List[Dict[str, str]]) -> str:
        """格式化对话历史为字符串"""
        recent_history = MemoryManager.get_recent_history(chat_history)
        return "\n".join([f"{msg['role']}：{msg['content']}" for msg in recent_history])
    
    @staticmethod
    def save_user_structured_info(user_id: str, structured_info: dict):
        """保存用户结构化信息"""
        data_loader.save_user_structured_info(user_id, structured_info)
    
    @staticmethod
    def get_user_structured_info(user_id: str) -> dict:
        """获取用户结构化信息"""
        return data_loader.get_user_structured_info(user_id)
    
    @staticmethod
    def update_user_structured_info(user_id: str, new_info: dict) -> dict:
        """更新用户结构化信息，新信息覆盖旧信息"""
        current_info = MemoryManager.get_user_structured_info(user_id)
        # 用新信息更新当前信息
        updated_info = {**current_info, **new_info}
        # 保存更新后的信息
        MemoryManager.save_user_structured_info(user_id, updated_info)
        return updated_info


# 创建全局内存管理器实例
memory_manager = MemoryManager()