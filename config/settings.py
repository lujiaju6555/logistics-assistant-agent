import os
from dotenv import load_dotenv
from typing import Dict, Any

# 加载.env文件
load_dotenv()


class Settings:
    """配置管理类"""
    # 项目配置
    ROOT_FOLDER = os.getenv("ROOT_FOLDER", "./京东服务细则")
    IMAGE_SUBFOLDER = os.getenv("IMAGE_SUBFOLDER", "图片")
    SAVE_EMBED_PATH = os.getenv("SAVE_EMBED_PATH", "./rag_knowledge_base")
    SAVE_OCR_REPORT = os.getenv("SAVE_OCR_REPORT", "./image_ocr_report.txt")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # 文本切片大小
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # 文本重叠大小

    # 通义千问API配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")  # 你的API Key
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v1")  # 嵌入模型
    VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "qwen3-vl-flash")  # 视觉模型，支持图片识别
    QA_MODEL_NAME = os.getenv("QA_MODEL_NAME", "qwen-flash")  # 问答模型
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-flash")  # LLM模型

    # 业务配置
    QUESTION_FOLDER = os.getenv("QUESTION_FOLDER", "./智能客服问答")  # 问题文件存放路径
    BUSINESS_DB_FOLDER = os.getenv("BUSINESS_DB_FOLDER", "./业务数据库")  # 业务数据库文件夹
    RAG_K = int(os.getenv("RAG_K", "5"))  # RAG检索的切片数量
    
    # 多路召回配置
    TOP_K_PER_ROUTE = int(os.getenv("TOP_K_PER_ROUTE", "20"))  # 每路召回数量
    FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))  # 最终返回数量
    RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")  # 重排序模型名称

    # 对话配置
    MAX_CHAT_HISTORY = 5  # 滑动窗口大小，保留最近5轮对话

    # 数据库配置
    SQL_MAX_EXECUTE_TIME = int(os.getenv("SQL_MAX_EXECUTE_TIME", "5"))  # SQL执行超时时间(秒)
    ORDER_TABLE = os.getenv("ORDER_TABLE", "orders")  # 订单表名
    USER_TABLE = os.getenv("USER_TABLE", "users")  # 用户表名

    # 工具配置
    TOOLS = {
        "query_logistics_price": "查询物流价格",
        "track_order_status": "查询订单状态",
        "create_shipment": "创建物流订单",
        "calculate_delivery_time": "计算预计送达时间"
    }

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            "ROOT_FOLDER": cls.ROOT_FOLDER,
            "IMAGE_SUBFOLDER": cls.IMAGE_SUBFOLDER,
            "SAVE_EMBED_PATH": cls.SAVE_EMBED_PATH,
            "SAVE_OCR_REPORT": cls.SAVE_OCR_REPORT,
            "CHUNK_SIZE": cls.CHUNK_SIZE,
            "CHUNK_OVERLAP": cls.CHUNK_OVERLAP,
            "DASHSCOPE_API_KEY": cls.DASHSCOPE_API_KEY,
            "EMBEDDING_MODEL_NAME": cls.EMBEDDING_MODEL_NAME,
            "VISION_MODEL_NAME": cls.VISION_MODEL_NAME,
            "QA_MODEL_NAME": cls.QA_MODEL_NAME,
            "QUESTION_FOLDER": cls.QUESTION_FOLDER,
            "BUSINESS_DB_FOLDER": cls.BUSINESS_DB_FOLDER,
            "RAG_K": cls.RAG_K,
            "TOP_K_PER_ROUTE": cls.TOP_K_PER_ROUTE,
            "FINAL_TOP_K": cls.FINAL_TOP_K,
            "RERANK_MODEL_NAME": cls.RERANK_MODEL_NAME,
            "MAX_CHAT_HISTORY": cls.MAX_CHAT_HISTORY,
            "SQL_MAX_EXECUTE_TIME": cls.SQL_MAX_EXECUTE_TIME,
            "ORDER_TABLE": cls.ORDER_TABLE,
            "USER_TABLE": cls.USER_TABLE,
            "TOOLS": cls.TOOLS
        }


# 创建全局配置实例
settings = Settings()