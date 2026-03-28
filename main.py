import os
import csv
import time
import json
import base64
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import sqlite3  # 新增：用于数据库操作
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from requests.exceptions import Timeout, ConnectionError

# 加载.env文件
load_dotenv()

# --------------------------
# 配置项（从.env文件读取）
ROOT_FOLDER = os.getenv("ROOT_FOLDER", "./京东服务细则")
IMAGE_SUBFOLDER = os.getenv("IMAGE_SUBFOLDER", "图片")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", r"D:\all-MiniLM-L6-v2")
SAVE_EMBED_PATH = os.getenv("SAVE_EMBED_PATH", "./rag_knowledge_base")
SAVE_OCR_REPORT = os.getenv("SAVE_OCR_REPORT", "./image_ocr_report.txt")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # 文本切片大小
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # 文本重叠大小
# 通义千问API配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")  # 你的API Key
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v1")  # 嵌入模型
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "qwen3-vl-flash")  # 视觉模型，支持图片识别
QA_MODEL_NAME = os.getenv("QA_MODEL_NAME", "qwen-flash")  # 问答模型
QUESTION_FOLDER = os.getenv("QUESTION_FOLDER", "./智能客服问答")  # 问题文件存放路径
RAG_K = int(os.getenv("RAG_K", "5"))  # RAG检索的切片数量

# 新增：动态业务办理配置
BUSINESS_DB_FOLDER = os.getenv("BUSINESS_DB_FOLDER", "./业务数据库")  # 业务数据库文件夹
SQL_MAX_EXECUTE_TIME = int(os.getenv("SQL_MAX_EXECUTE_TIME", "5"))  # SQL执行超时时间(秒)
ORDER_TABLE = os.getenv("ORDER_TABLE", "orders")  # 订单表名
USER_TABLE = os.getenv("USER_TABLE", "users")  # 用户表名

# 对话历史配置
MAX_CHAT_HISTORY = 5  # 滑动窗口大小，保留最近5轮对话


def encode_image_to_base64(image_path):
    """将图片转换为base64编码（通义千问图片输入格式）"""
    try:
        with Image.open(image_path).convert("RGB") as img:
            # 压缩图片（避免过大）
            max_size = (1024, 1024)
            img.thumbnail(max_size)

            # 转换为base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"图片编码失败：{str(e)}")
        return None


def call_qwen_vl(image_base64, image_name):
    """调用通义千问视觉模型（带超时与重试）"""
    client = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=VISION_MODEL_NAME,
        temperature=0,
        timeout=30  # 强制超时时间（秒），根据图片复杂度调整
    )

    # 重试装饰器：最多3次，间隔1s→2s→4s
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((Timeout, ConnectionError, Exception))
    )
    def _invoke_model():
        prompt = f"""请识别以下图片中的表格内容，保持原有的行列结构和文字内容。
如果是表格，输出格式应为markdown表格；如果不是表格，直接输出识别的文字。
图片名称：{image_name}"""

        response = client.invoke([
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}
        ])
        return response.content.strip()

    try:
        return _invoke_model()
    except Exception as e:
        return f"❌ 模型调用失败（已重试3次）：{str(e)}"


def extract_table_from_image(image_path, report_file):
    """使用大模型提取表格图片内容"""
    print(f"\n{'=' * 50}")
    print(f"🔍 正在处理表格图片：{os.path.basename(image_path)}")
    print(f"📁 图片路径：{image_path}")

    try:
        # 1. 图片转base64
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return "⚠️ 图片编码失败，无法处理"

        # 2. 调用通义千问识别
        image_name = os.path.basename(image_path)
        print("📡 正在调用通义千问API识别图片...")
        recognition_result = call_qwen_vl(image_base64, image_name)

        if not recognition_result:
            return "⚠️ 未识别到任何内容"

        # 3. 处理识别结果
        final_text = f"【表格图片】{image_name}\n{recognition_result}"
        final_text = final_text.replace('\r', '').strip()

        # 4. 实时打印
        print(f"✅ 识别结果：\n{final_text}")

        # 5. 写入报告
        report_file.write(f"【表格图片信息】\n")
        report_file.write(f"文件名：{image_name}\n")
        report_file.write(f"完整路径：{image_path}\n")
        report_file.write(f"识别结果：\n{final_text}\n")
        report_file.write(f"{'=' * 30}\n\n")

        return final_text

    except Exception as e:
        error_msg = f"❌ 处理失败：{str(e)}"
        print(error_msg)
        report_file.write(f"【表格图片信息】\n")
        report_file.write(f"文件名：{os.path.basename(image_path)}\n")
        report_file.write(f"完整路径：{image_path}\n")
        report_file.write(f"错误信息：{error_msg}\n")
        report_file.write(f"{'=' * 30}\n\n")
        return error_msg


def load_all_knowledge(root_folder):
    """全量读取知识库（文本+表格图片）+ 生成OCR报告"""
    all_documents = []
    with open(SAVE_OCR_REPORT, "w", encoding="utf-8") as ocr_report:
        ocr_report.write(f"# 京东服务细则表格图片识别报告\n")
        ocr_report.write(f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        ocr_report.write(f"总处理文件夹：{root_folder}\n\n")

        for category in os.listdir(root_folder):
            category_path = os.path.join(root_folder, category)
            if not os.path.isdir(category_path):
                continue
            print(f"\n📂 开始处理分类：{category}")
            ocr_report.write(f"## 分类：{category}\n")

            # 读取文本文件
            text_file = None
            for file in os.listdir(category_path):
                if file.endswith(".txt"):
                    text_file = os.path.join(category_path, file)
                    break
            if not text_file:
                print(f"⚠️ 分类{category}无文本文件，跳过")
                ocr_report.write(f"警告：该分类无.txt文本文件，已跳过\n")
                ocr_report.write(f"{'=' * 30}\n\n")
                continue

            # 读取文本内容
            with open(text_file, "r", encoding="utf-8") as f:
                text_content = f.read()
            if not text_content:
                print(f"⚠️ 文本文件{text_file}为空，跳过")
                ocr_report.write(f"警告：文本文件{os.path.basename(text_file)}内容为空，已跳过\n")
                ocr_report.write(f"{'=' * 30}\n\n")
                continue
            print(f"📄 已读取文本文件：{os.path.basename(text_file)}（字数：{len(text_content)}）")

            # 处理表格图片
            image_folder = os.path.join(category_path, IMAGE_SUBFOLDER)
            image_infos = []
            if os.path.exists(image_folder):
                image_files = [f for f in os.listdir(image_folder) if
                               f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
                print(f"🖼️ 该分类下表格图片数量：{len(image_files)}")
                ocr_report.write(f"表格图片数量：{len(image_files)}\n")

                for img_file in image_files:
                    img_path = os.path.join(image_folder, img_file)
                    img_text = extract_table_from_image(img_path, ocr_report)
                    image_infos.append({
                        "image_name": img_file,
                        "image_path": img_path,
                        "extracted_text": img_text
                    })
                    text_content += f"\n\n【关联表格图片】{img_file}：{img_text}（图片路径：{img_path}）"

            else:
                print(f"⚠️ 该分类无{IMAGE_SUBFOLDER}子文件夹，跳过图片处理")
                ocr_report.write(f"警告：无{IMAGE_SUBFOLDER}子文件夹，未处理图片\n")

            # 文本切片
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            split_docs = text_splitter.create_documents(
                texts=[text_content],
                metadatas=[{
                    "category": category,
                    "text_file_path": text_file,
                    "image_count": len(image_infos),
                    "image_paths": [img["image_path"] for img in image_infos]
                }]
            )
            all_documents.extend(split_docs)
            print(f"✅ 分类{category}处理完成，生成知识块：{len(split_docs)}个")
            ocr_report.write(f"知识块数量：{len(split_docs)}个\n")
            ocr_report.write(f"{'=' * 30}\n\n")

    # 打印统计信息
    print(f"\n{'=' * 60}")
    print(f"📊 知识库读取总统计：")
    print(
        f"   - 处理分类数量：{len([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])}")
    print(f"   - 生成知识块总数：{len(all_documents)}")
    print(f"   - OCR报告已保存至：{SAVE_OCR_REPORT}")
    print(f"{'=' * 60}")
    return all_documents


def build_rag_embeddings(documents, save_path):
    """构建RAG嵌入向量"""
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY
    )

    print(f"\n📈 从通义千问API生成嵌入向量...")
    # 创建向量存储
    db = FAISS.from_documents(documents, embeddings)

    # 保存向量库
    os.makedirs(save_path, exist_ok=True)
    db.save_local(save_path)

    print(f"✅ RAG知识库构建完成！")
    print(f"   - 向量库保存路径：{save_path}")
    print(f"   - 知识块总数：{len(documents)}")
    return db


def load_rag_embeddings(save_path):
    """加载已构建的RAG向量库"""
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY
    )
    db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    return db


def get_rag_answer_and_references(db, question, chat_history):
    """结合上下文和RAG进行回答，并返回参考的知识库片段"""
    # 检索相关文档
    docs = db.similarity_search(question, k=RAG_K)

    # 构建上下文和参考信息
    context = "\n".join([doc.page_content for doc in docs])
    references = []

    for i, doc in enumerate(docs, 1):
        # 获取元数据中的分类信息
        category = doc.metadata.get("category", "未知分类")
        # 判断内容来源类型（文本或图片）
        content = doc.page_content
        source_type = "图片" if "【关联表格图片】" in content else "文本"

        # 截取部分内容作为展示（最多200字）
        preview = content[:200] + ("..." if len(content) > 200 else "")

        references.append(
            f"[{i}] 分类：{category}（{source_type}）\n内容：{preview}"
        )

    # 构建对话历史上下文（限制长度）
    recent_chat_history = chat_history[-MAX_CHAT_HISTORY:] if len(chat_history) > MAX_CHAT_HISTORY else chat_history
    history_context = "\n".join([f"用户：{h['question']}\n助手：{h['answer']}" for h in recent_chat_history])

    # 构建提示词
    prompt_template = """
    你是京东物流的智能客服助手，需要根据提供的知识库内容和对话历史回答用户问题。
    请遵循以下规则：
    1. 优先使用知识库中的信息回答
    2. 结合对话历史理解用户当前问题的上下文
    3. 回答要简洁明了，符合中文表达习惯
    4. 如果无法从知识库中找到答案，直接说明"抱歉，我无法回答这个问题"

    知识库内容：
    {context}

    对话历史：
    {history}

    当前问题：{question}

    回答：
    """

    prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=prompt_template
    )

    # 初始化大模型
    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=QA_MODEL_NAME,
        temperature=0.3
    )

    # 创建链并运行
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(context=context, history=history_context, question=question)

    # 合并参考信息为字符串
    references_text = "\n\n".join(references)

    return answer.strip(), references_text


def load_business_database():
    """加载业务数据库表格结构信息"""
    db_info = {}
    if not os.path.exists(BUSINESS_DB_FOLDER):
        print(f"⚠️ 业务数据库文件夹不存在：{BUSINESS_DB_FOLDER}")
        return db_info

    # 读取所有Excel表格作为业务数据库表
    for file in os.listdir(BUSINESS_DB_FOLDER):
        if file.endswith((".xlsx", ".xls")):
            table_name = os.path.splitext(file)[0]
            file_path = os.path.join(BUSINESS_DB_FOLDER, file)

            try:
                # 读取表格获取字段信息
                df = pd.read_excel(file_path)
                fields = df.columns.tolist()
                db_info[table_name] = {
                    "path": file_path,
                    "fields": fields,
                    "sample_data": df.head(3).to_dict(orient="records")  # 示例数据
                }
                print(f"📊 加载业务表：{table_name}，字段：{fields}")
            except Exception as e:
                print(f"❌ 加载业务表{file}失败：{str(e)}")

    return db_info


def generate_sql_query(question, db_schema, chat_history):
    """让大模型根据问题和数据库结构生成SQL查询"""
    # 构建数据库结构描述
    schema_desc = []
    for table, info in db_schema.items():
        schema_desc.append(f"表名：{table}")
        schema_desc.append(f"字段：{', '.join(info['fields'])}")
        schema_desc.append(f"示例数据：{json.dumps(info['sample_data'], ensure_ascii=False)[:200]}...")
        schema_desc.append("---")
    schema_text = "\n".join(schema_desc)

    # 构建对话历史
    history_context = "\n".join([f"用户：{h['question']}\n助手：{h['answer']}" for h in chat_history])

    # SQL生成提示词
    prompt_template = """
    你需要根据用户问题和数据库结构生成正确的SQL查询语句。
    请遵循以下规则：
    1. 仅使用提供的数据库表和字段，不使用不存在的表或字段
    2. 生成标准SQL语法的查询语句
    3. 只返回SQL语句，不包含任何解释或说明文字
    4. 如果无法生成SQL，返回"无法生成SQL查询"

    数据库结构：
    {schema}

    对话历史：
    {history}

    用户问题：{question}

    SQL查询：
    """

    prompt = PromptTemplate(
        input_variables=["schema", "history", "question"],
        template=prompt_template
    )

    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=QA_MODEL_NAME,
        temperature=0.1
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    sql = chain.run(schema=schema_text, history=history_context, question=question)
    return sql.strip()


def execute_sql_query(sql, db_schema):
    """执行SQL查询（简化版：直接让大模型从表格数据中提取答案，不解析SQL）"""
    if not sql or sql == "无法生成SQL查询":
        print('无法生成SQL查询')
        return None, "未生成有效的SQL查询"

    try:
        # 简单提取目标表名（读取所有核心业务表：OMS、TMS、WMS）
        target_tables = ['OMS', 'TMS', 'WMS']
        table_data = ""

        # 读取所有目标表的完整数据并合并
        for table_name in target_tables:
            if table_name not in db_schema:
                print(f"⚠️  未找到{table_name}表，跳过该表数据读取")
                continue

            # 读取对应Excel表数据
            table_info = db_schema[table_name]
            df = pd.read_excel(table_info["path"])

            # 转换为字符串，添加表名标识（便于大模型区分不同表数据）
            table_data += f"【{table_name}表数据】\n"
            table_data += df.to_string(index=False)  # 不显示行索引
            table_data += "\n\n"  # 表之间用空行分隔，提升可读性

        # 处理无数据的情况
        if not table_data.strip():
            table_data = "未获取到OMS、TMS、WMS表的有效数据"

        # 让大模型直接从表格数据中回答问题（忽略SQL语法，只关注问题和数据）
        prompt = f"""
        你是京东物流的智能客服助手，需要根据提供的知识库内容和对话历史回答用户问题。
        请遵循以下规则：
        1. 优先使用表格中的信息回答
        2. 结合对话历史理解用户当前问题的上下文
        3. 回答要简洁明了，符合中文表达习惯，用接近人工客服的语言
        4. 如果无法从表格中找到答案，直接说明"抱歉，我无法回答这个问题"
        请根据以下表格数据回答用户问题，不需要参考SQL语句，仅使用表格中的信息。
        表格名：{table_name}
        表格数据：
        {table_data}

        用户问题（对应SQL意图）：{sql}  # 这里将SQL作为问题意图参考
        回答：
        """

        llm = ChatOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=QA_MODEL_NAME,
            temperature=0.1
        )
        answer = llm.invoke(prompt).content.strip()

        return answer, f"参考表格：{table_name}\n表格数据预览：{table_data[:300]}..."  # 预览前300字符

    except Exception as e:
        return None, f"查询处理失败：{str(e)}"


def extract_structured_info(question, user):
    """从用户输入中提取结构化信息"""
    structured_info = {
        "user_id": user,
        "order_id": None,
        "tracking_number": None,
        "product_name": None,
        "issue_type": None
    }
    
    # 提取订单号
    import re
    order_pattern = r'订单(号|编号)?[:：]?\s*(\w+)'  # 匹配"订单号：ORD123"或"订单ORD123"
    order_match = re.search(order_pattern, question, re.IGNORECASE)
    if order_match:
        structured_info["order_id"] = order_match.group(2)
    
    # 提取物流单号
    tracking_pattern = r'(物流|快递|运单)(号|编号)?[:：]?\s*(\w+)'  # 匹配"物流单号：123456"
    tracking_match = re.search(tracking_pattern, question, re.IGNORECASE)
    if tracking_match:
        structured_info["tracking_number"] = tracking_match.group(3)
    
    # 提取产品名称
    product_pattern = r'(商品|产品|物品)[:：]?\s*([^，。！？]+)'  # 匹配"商品：手机"
    product_match = re.search(product_pattern, question, re.IGNORECASE)
    if product_match:
        structured_info["product_name"] = product_match.group(2)
    
    # 提取问题类型
    issue_keywords = {
        "delay": ["延迟", "晚", "超时", "没到"],
        "damage": ["破损", "损坏", "坏了"],
        "loss": ["丢失", "找不到", "没收到"],
        "refund": ["退货", "退款", "退钱"]
    }
    
    for issue_type, keywords in issue_keywords.items():
        for keyword in keywords:
            if keyword in question:
                structured_info["issue_type"] = issue_type
                break
        if structured_info["issue_type"]:
            break
    
    return structured_info

def get_business_answer(db, business_db, question, chat_history, user):
    """主智能体：使用工具调用机制处理用户问题"""
    # 提取结构化信息
    structured_info = extract_structured_info(question, user)
    print(f"📋 提取的结构化信息：{structured_info}")
    
    # 限制对话历史长度（滑动窗口）
    recent_chat_history = chat_history[-MAX_CHAT_HISTORY:] if len(chat_history) > MAX_CHAT_HISTORY else chat_history
    
    # 定义工具函数
    def static_knowledge_tool(query):
        """处理规则咨询和异常解决类型的问题"""
        answer, reference = handle_static_knowledge(db, query, recent_chat_history)
        return f"回答：{answer}\n参考：{reference}"

    def business_query_tool(query):
        """处理状态查询类型的问题"""
        answer, reference = handle_business_query(business_db, query, recent_chat_history, user)
        return f"回答：{answer}\n参考：{reference}"

    def business_operation_tool(query):
        """处理业务操作类型的问题"""
        answer, reference = handle_business_operation(query)
        return f"回答：{answer}\n参考：{reference}"

    # 定义工具
    tools = [
        Tool(
            name="static_knowledge",
            func=static_knowledge_tool,
            description="处理规则咨询和异常解决类型的问题，输入为问题文本"
        ),
        Tool(
            name="business_query",
            func=business_query_tool,
            description="处理状态查询类型的问题，输入为问题文本"
        ),
        Tool(
            name="business_operation",
            func=business_operation_tool,
            description="处理业务操作类型的问题，输入为问题文本"
        )
    ]

    # 创建LLM
    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=QA_MODEL_NAME,
        temperature=0.1
    )

    # 初始化代理
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # 构建包含结构化信息的问题
    enhanced_question = f"用户ID：{user}\n"
    if structured_info["order_id"]:
        enhanced_question += f"订单号：{structured_info['order_id']}\n"
    if structured_info["tracking_number"]:
        enhanced_question += f"物流单号：{structured_info['tracking_number']}\n"
    if structured_info["product_name"]:
        enhanced_question += f"产品名称：{structured_info['product_name']}\n"
    if structured_info["issue_type"]:
        enhanced_question += f"问题类型：{structured_info['issue_type']}\n"
    enhanced_question += f"用户问题：{question}"

    # 执行代理
    result = agent.run(enhanced_question)

    # 处理结果
    return result, f"工具调用结果\n结构化信息：{structured_info}"


def merge_results_with_llm(original_question, sub_results, chat_history):
    """用大模型合并子智能体结果，生成连贯回答"""
    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=QA_MODEL_NAME,
        temperature=0.3  # 适度随机性，保证流畅度
    )

    # 整理子结果为自然语言描述
    sub_answers = "\n".join([
        f"针对问题片段：'{res['answer'].split('：')[0].strip()}'\n回答内容：{res['answer']}"
        for res in sub_results if res["answer"]
    ])

    # 合并结果的提示词
    merge_prompt = PromptTemplate(
        input_variables=["original_question", "sub_answers", "chat_history"],
        template="""
        请将多个子问题的回答整合成一个连贯、自然的完整回答，满足用户的原始问题。

        要求：
        1. 保留所有子回答的关键信息，不遗漏任何用户关心的点；
        2. 按用户提问的逻辑顺序组织内容（先回答先问的问题）；
        3. 语言简洁流畅，避免重复，用口语化的表达连接不同部分；
        4. 若涉及多个业务类型（如同时查物流+问退货），用过渡句自然衔接（如“关于退货流程及费用”）。

        原始用户问题：{original_question}
        子问题及对应回答：
        {sub_answers}
        对话历史：{chat_history}

        整合后的完整回答：
        """
    )

    # 调用大模型合并
    merge_chain = merge_prompt | llm
    final_answer = merge_chain.invoke({
        "original_question": original_question,
        "sub_answers": sub_answers,
        "chat_history": chat_history
    }).content.strip()

    # 参考信息仍拼接（用于溯源）
    final_reference = "\n".join([res["reference"] for res in sub_results if res["reference"]])

    return final_answer, final_reference


# 子智能体函数（保持不变）
def handle_static_knowledge(db, question, chat_history):
    """静态知识查询子智能体（RAG处理）"""
    # 限制对话历史长度
    recent_chat_history = chat_history[-MAX_CHAT_HISTORY:] if len(chat_history) > MAX_CHAT_HISTORY else chat_history
    return get_rag_answer_and_references(db, question, recent_chat_history)

def handle_business_query(business_db, question, chat_history, user):
    """业务数据查询子智能体（SQL处理）"""
    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=QA_MODEL_NAME,
        temperature=0.1
    )
    sql_prompt = PromptTemplate(
        input_variables=["question", "business_db", "chat_history"],
        template="基于业务数据库结构生成SQL查询语句，仅返回SQL：\n数据库信息：{business_db}\n对话历史：{chat_history}\n用户ID:{user}\n用户问题：{question}\nSQL语句："
    )
    sql_chain = sql_prompt | llm
    sql = sql_chain.invoke({"question": question, "business_db": business_db, "chat_history": chat_history, "user": user}).content.strip()
    print(f"📊 生成SQL：{sql}")
    result, msg = execute_sql_query(sql, business_db)
    return (f"查询结果：\n{result}", f"SQL查询：{sql}") if result else (msg, f"SQL查询：{sql}")


def handle_business_operation(question):
    """业务办理子智能体"""
    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=QA_MODEL_NAME,
        temperature=0.1
    )
    business_prompt = PromptTemplate(
        input_variables=["question"],
        template="识别用户需要办理的具体业务，仅返回以下之一：下单、退单、寄件、修改、其他\n用户问题：{question}"
    )
    business_result = business_prompt | llm
    business_type = business_result.invoke({"question": question}).content.strip()
    business_links = {"下单": "https://example.com/order", "退单": "https://example.com/return", "寄件": "https://example.com/send", "修改": "https://example.com/modify"}
    if business_type in business_links:
        return f"办理{business_type}业务：\n{business_links[business_type]}", f"业务类型：{business_type}"
    return "支持的业务：下单、退单、寄件、修改，请明确需求", "未识别具体业务"


def process_questions():
    """处理问题文件（新增动态业务支持）"""
    if not os.path.exists(QUESTION_FOLDER):
        print(f"❌ 问题文件夹不存在：{QUESTION_FOLDER}")
        return

    question_files = [f for f in os.listdir(QUESTION_FOLDER) if
                      f.startswith("问题") and f.endswith(".xlsx")]

    if not question_files:
        print(f"⚠️ 未在{QUESTION_FOLDER}找到问题文件")
        return

    # 加载向量库和业务数据库
    db = load_rag_embeddings(SAVE_EMBED_PATH)
    business_db = load_business_database()  # 新增：加载业务数据库

    for q_file in question_files:
        t1 = time.time()
        q_path = os.path.join(QUESTION_FOLDER, q_file)
        print(f"\n{'=' * 50}")
        print(f"📝 开始处理问题文件：{q_file}")

        try:
            df = pd.read_excel(q_path)

            chat_history = []
            answers = []

            for _, row in df.iterrows():
                user = row["用户"]
                question = str(row["问题"]).strip()
                print(f"\n🔍 处理用户{user}问题：{question}")

                question = '用户ID=' + user + '||' + question

                # 新增：判断是静态知识还是动态业务
                answer, references = get_business_answer(db, business_db, question, chat_history, user)
                print(f"💬 回答：{answer}")
                print(f"📌 参考信息：\n{references}")

                chat_history.append({"question": question, "answer": answer})
                answers.append({
                    "用户": user,
                    "问题": question,
                    "回答": answer,
                    "参照": references
                })

            # 保存回答结果
            answer_df = pd.DataFrame(answers)
            answer_file = q_file.replace("问题", "回答")
            answer_path = os.path.join(QUESTION_FOLDER, answer_file)
            with pd.ExcelWriter(answer_path, engine='openpyxl') as writer:
                answer_df.to_excel(writer, index=False)
                worksheet = writer.sheets['Sheet1']
                worksheet.column_dimensions['A'].width = 8
                worksheet.column_dimensions['B'].width = 40
                worksheet.column_dimensions['C'].width = 60
                worksheet.column_dimensions['D'].width = 80
            print(f"✅ 回答已保存至：{answer_path}")

        except Exception as e:
            print(f"❌ 处理问题文件失败：{str(e)}")
        t2 = time.time()
        print(f"回答问题文件：{q_file} 用时:{t2 - t1:.2f} 秒")


def export_vectorstore_to_csv(embeddings_path, output_csv_path):
    """
    将向量库中的所有文档切片导出为CSV文件

    参数:
        embeddings_path: 向量库存储路径
        output_csv_path: 导出的CSV文件路径
    """
    # 加载嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY
    )

    # 加载向量库
    db = FAISS.load_local(
        embeddings_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 导出文档到CSV
    with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['切片ID', '原始文本', '分类', '文本文件路径', '图片数量', '图片路径'])

        # 遍历所有文档
        for i, doc in enumerate(db.docstore._dict.values(), 1):
            # 提取元数据
            metadata = doc.metadata
            category = metadata.get('category', '未知')
            text_file_path = metadata.get('text_file_path', '未知')
            image_count = metadata.get('image_count', 0)
            image_paths = metadata.get('image_paths', [])

            # 处理图片路径为字符串
            image_paths_str = ';'.join(image_paths) if image_paths else '无'

            # 写入一行数据
            writer.writerow([
                i,
                doc.page_content.replace('\n', ' '),  # 替换换行符避免CSV格式问题
                category,
                text_file_path,
                image_count,
                image_paths_str
            ])

    print(f"✅ 向量库内容已成功导出至: {output_csv_path}")
    print(f"📊 共导出 {len(db.docstore._dict)} 个文本切片")


if __name__ == "__main__":
    # 检查向量库是否已存在
    vector_db_exists = os.path.exists(os.path.join(SAVE_EMBED_PATH, "index.faiss")) and \
                       os.path.exists(os.path.join(SAVE_EMBED_PATH, "index.pkl"))

    if not vector_db_exists:
        print("🔨 向量数据库不存在，开始构建...")
        t1 = time.time()
        knowledge_docs = load_all_knowledge(ROOT_FOLDER)
        t2 = time.time()
        if knowledge_docs:
            build_rag_embeddings(knowledge_docs, SAVE_EMBED_PATH)
            print(f"构建向量数据库用时:{t2 - t1:.2f} 秒")
        else:
            print("❌ 未读取到知识库内容，无法构建向量库！")
            exit(1)
    else:
        print("📦 向量数据库已存在，直接加载...")

    # export_vectorstore_to_csv(
    #     embeddings_path=SAVE_EMBED_PATH,
    #     output_csv_path="./vectorstore_export.csv"
    # )

    # 处理问题并生成回答
    process_questions()
    print("\n🎉 所有操作完成！")