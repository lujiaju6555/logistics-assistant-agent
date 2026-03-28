from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import settings
from config import prompts
from utils import memory_manager

class QueryRewriter:
    """问题重写器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=settings.QA_MODEL_NAME,
            temperature=0.1
        )
        self.prompt = PromptTemplate(
            input_variables=["question", "chat_history"],
            template=prompts.QUERY_REWRITING["dynamic"]
        )
        # 系统提示词
        self.system_prompt = prompts.QUERY_REWRITING["system"]
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def rewrite_query(self, question: str, chat_history: List[Dict[str, str]]) -> str:
        """重写问题"""
        try:
            formatted_history = memory_manager.format_history(chat_history)
            rewritten_query = self.chain.run(
                question=question,
                chat_history=formatted_history
            )
            return rewritten_query.strip()
        except Exception as e:
            # 兜底处理，返回原始问题
            return question


# 创建全局问题重写器实例
query_rewriter = QueryRewriter()