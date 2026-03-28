from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import settings
from config import prompts


class AnswerGenerator:
    """回答生成器"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=settings.QA_MODEL_NAME,
            temperature=0.3
        )
        self.prompt = PromptTemplate(
            input_variables=["question", "intent_type", "sub_intent", "extracted_params", "tool_results", "rag_context"],
            template=prompts.ANSWER_GENERATION["dynamic"]
        )
        # 系统提示词
        self.system_prompt = prompts.ANSWER_GENERATION["system"]
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_answer(self, intent: Dict[str, Any], rewritten_query: str,
                        extracted_params: Dict[str, Any], tool_results: Dict[str, Any],
                        rag_context: List[Dict[str, Any]]) -> str:
        """生成最终回答"""
        try:
            # 处理不同意图
            intent_type = intent.get("intent_type", "BUSINESS_QUERY")

            if intent_type == "CHITCHAT":
                return "您好，我是京东物流智能客服，有什么可以帮助您的吗？"
            elif intent_type == "RESET":
                return "对话已重置，请问有什么可以帮助您的？"
            elif intent_type == "HUMAN_TRANSFER":
                return "正在为您转接人工客服，请稍候..."

            # 业务查询
            answer = self.chain.run(
                question=rewritten_query,
                intent_type=intent.get("intent_type", "BUSINESS_QUERY"),
                sub_intent=intent.get("sub_intent", ""),
                extracted_params=extracted_params,
                tool_results=tool_results,
                rag_context=rag_context
            )

            return answer.strip()
        except Exception as e:
            return f"生成回答失败: {str(e)}"


# 创建全局回答生成器实例
answer_generator = AnswerGenerator()