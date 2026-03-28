from typing import Dict, Any, List
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import settings
from config import prompts
import logging
logging.basicConfig(level=logging.INFO)
import json


class IntentRecognizer:
    """意图识别器"""
    
    def __init__(self):
        # 规则配置
        self.reset_keywords = ["重置", "重新开始", "从头开始", "清除历史", "重新来过"]
        self.chat_keywords = []
        self.human_transfer_keywords = ["转人工", "人工客服", "找人工", "人工服务", "需要人工", "我要人工", "人工"]
        self.negative_emotion_keywords = ["垃圾", "傻逼", "混蛋", "生气", "愤怒", "投诉", "差评", "不满"]
        
        # 业务查询子意图关键词
        self.business_sub_intents = {
            "TRACKING": ["查件"],
            "PRICE_QUERY": ["价格", "费用", "多少钱", "收费", "报价"],
            "RULE_QUERY": ["规则", "流程"],
            "ORDER_CREATE": ["下单", "寄件", "发货", "创建订单"],
            "DELIVERY_TIME": ["送达时间"]
        }
        
        # LLM配置
        self.llm = ChatOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=settings.QA_MODEL_NAME,
            temperature=0.1
        )
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template=prompts.INTENT_RECOGNITION["dynamic"]
        )
        # 系统提示词
        self.system_prompt = prompts.INTENT_RECOGNITION["system"]
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _rule_based_recognition(self, question: str) -> Dict[str, Any]:
        """基于规则的意图识别"""
        # 转人工（优先级最高）
        for keyword in self.human_transfer_keywords:
            if keyword in question:
                return {
                    "intent_type": "HUMAN_TRANSFER",
                    "sub_intent": None,
                    "confidence": 0.99,
                    "reason": f"包含转人工关键词: {keyword}"
                }
        
        # 情绪异常（转人工）
        for keyword in self.negative_emotion_keywords:
            if keyword in question:
                return {
                    "intent_type": "HUMAN_TRANSFER",
                    "sub_intent": None,
                    "confidence": 0.95,
                    "reason": f"包含负面情绪关键词: {keyword}"
                }
        
        # 重置
        for keyword in self.reset_keywords:
            if keyword in question:
                return {
                    "intent_type": "RESET",
                    "sub_intent": None,
                    "confidence": 0.99,
                    "reason": f"包含重置关键词: {keyword}"
                }
        
        # 闲聊
        for keyword in self.chat_keywords:
            if keyword in question:
                return {
                    "intent_type": "CHITCHAT",
                    "sub_intent": None,
                    "confidence": 0.99,
                    "reason": f"包含闲聊关键词: {keyword}"
                }
        
        # 业务查询子意图
        for sub_intent, keywords in self.business_sub_intents.items():
            for keyword in keywords:
                if keyword in question:
                    return {
                        "intent_type": "BUSINESS_QUERY",
                        "sub_intent": sub_intent,
                        "confidence": 0.9,
                        "reason": f"包含业务查询关键词: {keyword}"
                    }
        
        # 未匹配到规则
        return None
    
    def recognize_intent(self, question: str) -> Dict[str, Any]:
        """识别意图"""
        # 先使用规则识别
        logging.info(f"开始识别意图: {question}")
        rule_result = self._rule_based_recognition(question)
        logging.info(f"规则识别结果: {rule_result}")
        if rule_result:
            return rule_result
        
        # 规则未匹配，使用LLM识别
        try:
            logging.info(f"📡 LLM开始调用:")
            result = self.chain.run(question=question)
            # 清理结果，去除可能的前缀和后缀
            result = result.strip()
            # 确保结果是有效的JSON
            if not result.startswith('{'):
                # 尝试找到JSON开始的位置
                json_start = result.find('{')
                if json_start != -1:
                    result = result[json_start:]
                else:
                    raise ValueError("No JSON found in result")
            if not result.endswith('}'):
                # 尝试找到JSON结束的位置
                json_end = result.rfind('}')
                if json_end != -1:
                    result = result[:json_end+1]
                else:
                    raise ValueError("No JSON end found in result")
            intent = json.loads(result)
            logging.info(f"✅ LLM识别结果: {intent}")
            return intent
        except Exception as e:
            logging.info(f"❌ 识别失败: {str(e)}")
            # 默认转人工
            return {
                "intent_type": "HUMAN_TRANSFER",
                "sub_intent": None,
                "confidence": 0.5,
                "reason": f"识别失败，默认转人工: {str(e)}"
            }


# 创建全局意图识别器实例
intent_recognizer = IntentRecognizer()