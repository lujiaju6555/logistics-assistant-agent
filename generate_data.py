"""
自动化构建RAG系统黄金测试集脚本

此脚本会从向量数据库中采样知识切片，使用LLM自动生成问题-答案-来源三元组，
并保存为JSONL格式的黄金测试集。使用真实的point_id作为知识切片的索引。

需要修改的部分：
1. API Key - 从环境变量自动读取
2. 向量数据库路径 - 从配置文件读取
3. 模型名称 - 从配置文件读取
"""

# 设置默认路径
DEFAULT_OUTPUT_PATH = "./data/golden_test_set.jsonl"  # 默认输出文件路径
DEFAULT_SAMPLE_SIZE = 100  # 默认采样大小

import os
import json
import asyncio
import aiohttp
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from config import settings
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入RAG检索器
from modules.execution.rag_retriever import RAGRetriever

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QATriple:
    """问题-答案-来源三元组数据类"""
    question: str
    answer: str
    difficulty: str  # 'Simple' 或 'Hard'
    source_ids: List[str]  # 使用真实的point_id作为source_ids
    reasoning: Optional[str] = None


class GoldenDatasetGenerator:
    """黄金测试集生成器"""

    def __init__(self, output_path: str):
        """
        初始化生成器

        Args:
            output_path: 输出文件路径
        """
        self.output_path = Path(output_path)
        self.api_key = settings.DASHSCOPE_API_KEY
        self.model_name = settings.LLM_MODEL_NAME
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        
        # 初始化RAG检索器
        self.rag_retriever = RAGRetriever()
        logger.info("✅ 已加载RAG检索器和向量数据库")

    def get_knowledge_chunks_from_vector_db(self) -> List[Dict[str, str]]:
        """
        从向量数据库中获取所有知识切片

        Returns:
            知识切片列表，每个元素包含内容和真实的point_id
        """
        documents = []
        
        try:
            # 获取向量存储中的所有文档
            if hasattr(self.rag_retriever.vector_store, 'docstore') and hasattr(self.rag_retriever.vector_store, 'index_to_docstore_id'):
                docs = []
                for doc_id in self.rag_retriever.vector_store.index_to_docstore_id.values():
                    try:
                        doc = self.rag_retriever.vector_store.docstore.search(doc_id)
                        if doc:
                            docs.append(doc)
                    except Exception as e:
                        logger.warning(f"获取文档失败: {e}")
                        continue
                
                logger.info(f"从向量数据库中获取到 {len(docs)} 个知识切片")
                
                # 将文档转换为所需格式，使用真实的point_id
                for doc in docs:
                    # 跳过空内容
                    if not doc.page_content or not doc.page_content.strip():
                        continue
                        
                    # 获取真实的point_id
                    point_id = doc.metadata.get('point_id')
                    if not point_id:
                        logger.warning(f"文档缺少point_id，跳过: {doc.metadata}")
                        continue
                    
                    documents.append({
                        "content": doc.page_content,
                        "source_id": point_id,  # 使用真实的point_id作为source_id
                        "metadata": doc.metadata  # 保留完整的metadata信息
                    })
            else:
                logger.error("向量数据库结构不完整，无法获取文档")
                
        except Exception as e:
            logger.error(f"从向量数据库获取知识切片时出错: {e}")
            
        logger.info(f"从向量数据库中总共获取到 {len(documents)} 个有效知识切片")
        return documents

    def sample_knowledge_chunks(self, chunks: List[Dict[str, str]], sample_size: int = 80) -> List[List[Dict[str, str]]]:
        """
        智能采样知识切片，为多来源问题准备数据

        Args:
            chunks: 知识切片列表
            sample_size: 采样数量

        Returns:
            采样后的知识切片组列表，每个组包含1-5个相关的知识切片
        """
        if len(chunks) <= sample_size:
            logger.info(f"知识切片总数({len(chunks)})小于等于采样数({sample_size})，返回全部")
            # 将单个切片包装成单元素列表
            return [[chunk] for chunk in chunks]

        # 按长度过滤切片（排除过短或过长的）
        filtered_chunks = [chunk for chunk in chunks if 50 <= len(chunk['content']) <= 2000]
        logger.info(f"过滤后剩余 {len(filtered_chunks)} 个有效知识切片")

        if len(filtered_chunks) <= sample_size:
            logger.info(f"过滤后切片数({len(filtered_chunks)})小于等于采样数({sample_size})，返回全部")
            # 将单个切片包装成单元素列表
            return [[chunk] for chunk in filtered_chunks]

        # 为生成多来源问题，随机选择切片并将其组织成组
        sampled_chunks = random.sample(filtered_chunks, sample_size)

        # 将采样切片分组，每组1-5个切片
        grouped_chunks = []
        i = 0
        while i < len(sampled_chunks):
            remaining = len(sampled_chunks) - i
            if remaining == 1:
                # 如果只剩一个切片，单独成组
                group = [sampled_chunks[i]]
                grouped_chunks.append(group)
                i += 1
            else:
                # 每组1-5个切片，或直到切片用完
                max_group_size = min(5, remaining)
                if max_group_size == 1:
                    # 如果最大组大小为1，单独成组
                    group = [sampled_chunks[i]]
                    grouped_chunks.append(group)
                    i += 1
                else:
                    # 随机选择1到最大允许的组大小之间的值
                    group_size = random.randint(1, max_group_size)
                    group = sampled_chunks[i:i+group_size]
                    grouped_chunks.append(group)
                    i += group_size

        logger.info(f"随机采样 {len(sampled_chunks)} 个知识切片，组成 {len(grouped_chunks)} 个切片组")
        return grouped_chunks

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def generate_qa_pair(self, session: aiohttp.ClientSession, doc_group: List[Dict[str, str]]) -> Optional[QATriple]:
        """
        异步生成单个问题-答案对

        Args:
            session: HTTP会话
            doc_group: 文档组，包含多个文档切片

        Returns:
            QATriple对象或None
        """
        system_prompt = """你是一个专业的物流领域评测专家。请基于提供的【文档片段集合】，构造一个问题-答案对。

要求：
1. 问题必须独立完整，包含具体的物流术语或实体名，不能使用"这篇文章"、"上述内容"等指代词
2. 答案必须严格由提供的文档片段内容支撑，不得编造或产生幻觉
3. 问题应反映物流与供应链领域的实际业务场景
4. 如果单个片段能完整回答，标记 difficulty 为 'Simple'，source_ids 包含该片段 ID
5. 如果需要结合 2 个或以上片段的信息才能完整回答（例如对比、总结、因果推理），标记 difficulty 为 'Hard'，source_ids 必须包含所有相关片段的 ID
6. 在 reasoning 字段中说明需要哪些片段来回答问题
7. 如果文档片段集合没有足够的信息量生成有意义的问题，请返回 null

请严格按照以下JSON格式输出：
{"question": "问题内容", "answer": "答案内容", "difficulty": "Simple/Hard", "source_ids": ["来源ID1", "来源ID2"], "reasoning": "推理说明"}
或
null"""

        # 构建用户提示，包含所有知识切片
        content_parts = []
        for i, chunk in enumerate(doc_group):
            content_parts.append(f"知识切片 {i+1} (point_id: {chunk['source_id']}):\n{chunk['content']}\n")

        user_prompt = """【知识切片集合】
""" + "\n".join(content_parts) + "\n请基于以上知识切片集合生成一个高质量的问题-答案对："

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024
        }

        try:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    logger.error(f"API请求失败: {response.status}")
                    return None

                result = await response.json()

                # 提取响应内容
                response_text = result['choices'][0]['message']['content'].strip()

                # 尝试解析JSON
                # 查找JSON部分
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    try:
                        qa_data = json.loads(json_str)

                        # 验证必需字段
                        if 'question' in qa_data and 'answer' in qa_data and 'difficulty' in qa_data and 'source_ids' in qa_data:
                            return QATriple(
                                question=qa_data['question'],
                                answer=qa_data['answer'],
                                difficulty=qa_data['difficulty'],
                                source_ids=qa_data['source_ids'],
                                reasoning=qa_data.get('reasoning', '')
                            )
                        else:
                            logger.warning(f"返回的JSON缺少必要字段: {json_str}")
                            return None

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析失败: {e}, 响应内容: {json_str}")
                        return None
                else:
                    logger.warning(f"未找到有效的JSON格式: {response_text}")
                    return None

        except Exception as e:
            logger.error(f"生成QA对时出错: {e}")
            raise  # 重试装饰器会处理重试逻辑

    async def generate_golden_dataset(self, sample_size: int = 80) -> List[QATriple]:
        """
        生成黄金测试集

        Args:
            sample_size: 采样大小

        Returns:
            QA三元组列表
        """
        logger.info("开始从向量数据库获取知识切片...")
        all_chunks = self.get_knowledge_chunks_from_vector_db()

        if not all_chunks:
            logger.error("未能从向量数据库获取任何知识切片")
            return []

        logger.info("开始采样知识切片...")
        sampled_chunks = self.sample_knowledge_chunks(all_chunks, sample_size)

        logger.info(f"开始异步生成 {len(sampled_chunks)} 个QA对...")

        # 创建异步会话
        connector = aiohttp.TCPConnector(limit=10, force_close=True)  # 限制并发连接数并强制关闭
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for chunk_group in sampled_chunks:
                task = self.generate_qa_pair(session, chunk_group)
                tasks.append(task)

            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤有效结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"任务 {i} 发生异常: {result}")
                continue
            if result is not None:
                valid_results.append(result)

        logger.info(f"成功生成 {len(valid_results)} 个有效的QA对")
        return valid_results

    def save_golden_dataset(self, qa_pairs: List[QATriple]):
        """
        保存黄金测试集到JSONL文件

        Args:
            qa_pairs: QA三元组列表
        """
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                json_line = json.dumps({
                    'question': qa.question,
                    'answer': qa.answer,
                    'difficulty': qa.difficulty,
                    'source_ids': qa.source_ids,
                    'reasoning': qa.reasoning
                }, ensure_ascii=False)
                f.write(json_line + '\n')

        logger.info(f"黄金测试集已保存到 {self.output_path}")

    def print_statistics(self, qa_pairs: List[QATriple]):
        """
        打印统计信息

        Args:
            qa_pairs: QA三元组列表
        """
        if not qa_pairs:
            logger.info("没有生成任何QA对")
            return

        total_count = len(qa_pairs)
        simple_count = sum(1 for qa in qa_pairs if qa.difficulty == 'Simple')
        hard_count = sum(1 for qa in qa_pairs if qa.difficulty == 'Hard')

        # 统计多来源问题
        multi_source_count = sum(1 for qa in qa_pairs if len(qa.source_ids) > 1)

        logger.info("=" * 50)
        logger.info("黄金测试集生成统计:")
        logger.info(f"总QA对数量: {total_count}")
        logger.info(f"简单问题数量: {simple_count} ({simple_count/total_count*100:.1f}%)")
        logger.info(f"困难问题数量: {hard_count} ({hard_count/total_count*100:.1f}%)")
        logger.info(f"多来源问题数量: {multi_source_count} ({multi_source_count/total_count*100:.1f}%)")
        logger.info("=" * 50)


def main():
    """主函数 - 同步版本避免Windows asyncio问题"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='自动化构建RAG系统黄金测试集')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='输出文件路径 (JSONL格式)')
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE,
                        help='采样大小')
    
    args = parser.parse_args()

    # 创建生成器实例
    generator = GoldenDatasetGenerator(args.output)

    # 直接运行异步函数
    import nest_asyncio
    try:
        nest_asyncio.apply()  # 允许嵌套事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        qa_pairs = loop.run_until_complete(generator.generate_golden_dataset(sample_size=args.sample_size))
    except ImportError:
        # 如果nest_asyncio不可用，尝试直接运行
        qa_pairs = asyncio.run(generator.generate_golden_dataset(sample_size=args.sample_size))

    # 保存结果
    generator.save_golden_dataset(qa_pairs)

    # 打印统计信息
    generator.print_statistics(qa_pairs)

    logger.info("黄金测试集生成完成!")


if __name__ == "__main__":
    main()