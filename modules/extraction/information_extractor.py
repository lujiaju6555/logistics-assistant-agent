from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from config import settings
from config import prompts
import logging

class InformationExtractor:
    """信息抽取器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=settings.QA_MODEL_NAME,
            temperature=0.1
        )
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template=prompts.INFORMATION_EXTRACTION["dynamic"]
        )
        # 系统提示词
        self.system_prompt = prompts.INFORMATION_EXTRACTION["system"]
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def extract_information(self, question: str) -> Dict[str, Any]:
        """抽取结构化信息"""
        logging.info(f"开始抽取信息：{question}")
        result = self.chain.run(question=question)
        logging.info(f"抽取到的信息原始结果：{result}")
        extracted_info = json.loads(result)
        return extracted_info


# 创建全局信息抽取器实例
information_extractor = InformationExtractor()