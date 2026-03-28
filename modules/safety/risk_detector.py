from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import settings
from config import prompts

class RiskDetector:
    """风险检测器"""
    
    def __init__(self):
        # 规则检测
        self.sensitive_keywords = [
            "敏感信息", "个人信息", "密码", "银行卡", "身份证", 
            "手机号", "住址", "姓名", "验证码", "账户"
        ]
        
        # LLM检测
        self.llm = ChatOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=settings.QA_MODEL_NAME,
            temperature=0.1
        )
        self.prompt = PromptTemplate(
            input_variables=["answer"],
            template=prompts.RISK_DETECTION["dynamic"]
        )
        # 系统提示词
        self.system_prompt = prompts.RISK_DETECTION["system"]
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _rule_based_detect(self, answer: str) -> Tuple[bool, str]:
        """基于规则的风险检测"""
        # 检测敏感信息
        for keyword in self.sensitive_keywords:
            if keyword in answer:
                return True, f"回答中包含{keyword}，存在风险"
        return False, ""
    
    def _llm_based_detect(self, answer: str) -> Tuple[bool, str]:
        """基于LLM的风险检测"""
        try:
            result = self.chain.run(answer=answer)
            # 解析JSON结果
            import json
            risk_info = json.loads(result)
            if risk_info.get("has_risk", False):
                return True, risk_info.get("risk_info", "存在风险")
            return False, ""
        except Exception as e:
            return False, f"风险检测失败: {str(e)}"
    
    def detect_risk(self, answer: str) -> Tuple[bool, str]:
        """检测风险"""
        # 先使用规则检测
        has_risk, risk_info = self._rule_based_detect(answer)
        if has_risk:
            return has_risk, risk_info
        
        # 规则未检测到风险，使用LLM检测
        return self._llm_based_detect(answer)
    
    def filter_risk(self, answer: str) -> str:
        """过滤风险"""
        has_risk, risk_info = self.detect_risk(answer)
        if has_risk:
            return "抱歉，我无法回答这个问题，请尝试其他方式获取信息。"
        return answer


# 创建全局风险检测器实例
risk_detector = RiskDetector()