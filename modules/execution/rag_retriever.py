from typing import List, Dict, Any, Tuple
import os
import base64
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from sentence_transformers import CrossEncoder
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from requests.exceptions import Timeout, ConnectionError
import pandas as pd
from config import settings
import logging

class RAGRetriever:
    """RAG检索器"""

    def __init__(self):
        self.embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            dashscope_api_key=settings.DASHSCOPE_API_KEY
        )
        self.vector_store = self.load_vector_store()
        # 初始化BM25检索器
        self.bm25_retriever = None
        # 初始化重排序模型
        self.rerank_model = None
        self._init_bm25_retriever()
        self._init_rerank_model()

    def load_vector_store(self) -> FAISS:
        """加载向量存储"""
        try:
            logging.info(f"🔍 开始加载向量存储，路径：{settings.SAVE_EMBED_PATH}")
            return FAISS.load_local(
                settings.SAVE_EMBED_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logging.info(f"加载向量存储失败: {e}")
            # 返回空的向量存储
            return FAISS.from_texts([""], self.embeddings)

    def _init_bm25_retriever(self):
        """初始化BM25检索器"""
        try:
            # 从向量存储中提取文档
            if hasattr(self.vector_store, 'docstore') and hasattr(self.vector_store, 'index_to_docstore_id'):
                docs = []
                for doc_id in self.vector_store.index_to_docstore_id.values():
                    try:
                        doc = self.vector_store.docstore.search(doc_id)
                        if doc:
                            docs.append(doc)
                    except:
                        continue
                
                if docs:
                    self.bm25_retriever = BM25Retriever.from_documents(docs)
                    logging.info(f"✅ BM25检索器初始化成功，文档数：{len(docs)}")
                else:
                    logging.warning("⚠️ 无文档可初始化BM25检索器")
            else:
                logging.warning("⚠️ 向量存储结构不完整，无法初始化BM25检索器")
        except Exception as e:
            logging.error(f"❌ 初始化BM25检索器失败: {str(e)}")
    
    def _init_rerank_model(self):
        """初始化重排序模型"""
        try:
            self.rerank_model = CrossEncoder(model_name=settings.RERANK_MODEL_NAME, max_length=512)
            logging.info(f"✅ 重排序模型初始化成功: {settings.RERANK_MODEL_NAME}")
        except Exception as e:
            logging.error(f"❌ 初始化重排序模型失败: {str(e)}")
            self.rerank_model = None
    
    def retrieve_documents(self, query: str, rewritten_query: str = None, k: int = None) -> List[Dict[str, Any]]:
        """多路召回 + 全局重排序检索"""
        if k is None:
            k = settings.RAG_K
        
        # 如果没有重写问题，使用原始问题
        if not rewritten_query:
            rewritten_query = query
        
        logging.info(f"🔍 开始多路召回检索")
        logging.info(f"   原始问题: {query}")
        logging.info(f"   重写问题: {rewritten_query}")
        
        try:
            # 1. 并行执行四路召回
            with ThreadPoolExecutor(max_workers=4) as executor:
                # 路数1: 原问 + Dense
                future1 = executor.submit(self._dense_retrieval, query, settings.TOP_K_PER_ROUTE)
                # 路数2: 原问 + Sparse
                future2 = executor.submit(self._sparse_retrieval, query, settings.TOP_K_PER_ROUTE)
                # 路数3: 重写问 + Dense
                future3 = executor.submit(self._dense_retrieval, rewritten_query, settings.TOP_K_PER_ROUTE)
                # 路数4: 重写问 + Sparse
                future4 = executor.submit(self._sparse_retrieval, rewritten_query, settings.TOP_K_PER_ROUTE)
                
                # 获取结果
                results1 = future1.result()
                results2 = future2.result()
                results3 = future3.result()
                results4 = future4.result()
            
            logging.info(f"[Recall] 原问-Dense 召回 {len(results1)} 条")
            logging.info(f"[Recall] 原问-Sparse 召回 {len(results2)} 条")
            logging.info(f"[Recall] 重写问-Dense 召回 {len(results3)} 条")
            logging.info(f"[Recall] 重写问-Sparse 召回 {len(results4)} 条")
            
            # 2. 智能融合：合并并去重
            merged_docs = self._merge_and_deduplicate([results1, results2, results3, results4])
            logging.info(f"[Merge] 合并后共 {len(merged_docs)} 条，去重后剩余 {len(merged_docs)} 条")
            
            # 3. 全局重排序
            if len(merged_docs) > 0:
                start_time = time.time()
                reranked_docs = self._rerank(merged_docs, rewritten_query, k)
                rerank_time = time.time() - start_time
                logging.info(f"[Rerank] 耗时 {rerank_time:.2f}s，最终返回 {len(reranked_docs)} 条")
                
                # 格式化结果
                formatted_results = []
                for i, doc in enumerate(reranked_docs, 1):
                    formatted_results.append({
                        "id": i,
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                return formatted_results
            else:
                logging.info("[Rerank] 无文档可重排序")
                return []
        except Exception as e:
            logging.error(f"❌ 多路召回失败: {type(e).__name__}: {str(e)}")
            # 降级为单路检索
            try:
                logging.info("🔄 降级为单路向量检索")
                docs = self.vector_store.similarity_search(query, k=k)
                results = []
                for i, doc in enumerate(docs, 1):
                    results.append({
                        "id": i,
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                return results
            except Exception as fallback_error:
                logging.error(f"❌ 降级检索也失败: {type(fallback_error).__name__}: {str(fallback_error)}")
                return []
    
    def _dense_retrieval(self, query: str, k: int) -> List[Document]:
        """向量检索"""
        try:
            if self.vector_store:
                return self.vector_store.similarity_search(query, k=k)
            return []
        except Exception as e:
            logging.error(f"❌ 向量检索失败: {str(e)}")
            return []
    
    def _sparse_retrieval(self, query: str, k: int) -> List[Document]:
        """BM25检索"""
        try:
            if self.bm25_retriever:
                return self.bm25_retriever.get_relevant_documents(query, k=k)
            return []
        except Exception as e:
            logging.error(f"❌ BM25检索失败: {str(e)}")
            return []
    
    def _merge_and_deduplicate(self, results_list: List[List[Document]]) -> List[Document]:
        """合并并去重文档"""
        # 使用point-id作为去重键，如果没有point-id则使用文档内容
        seen_ids = set()
        merged_docs = []
        
        # 按顺序处理，保留首次出现的文档
        for results in results_list:
            for doc in results:
                # 优先使用point-id作为去重键
                if 'point_id' in doc.metadata:
                    doc_id = doc.metadata['point_id']
                else:
                    # 回退到使用文档内容的前100个字符作为去重键
                    doc_id = doc.page_content[:100].strip()
                
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged_docs.append(doc)
        
        return merged_docs
    
    def _rerank(self, docs: List[Document], query: str, k: int) -> List[Document]:
        """全局重排序"""
        if not self.rerank_model:
            # 如果重排序模型未初始化，返回前k个文档
            logging.warning("⚠️ 重排序模型未初始化，返回前k个文档")
            return docs[:k]
        
        try:
            # 准备重排序的输入
            pairs = [[query, doc.page_content] for doc in docs]
            
            # 计算分数
            scores = self.rerank_model.predict(pairs)
            
            # 按分数排序
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前k个文档
            return [doc for doc, score in scored_docs[:k]]
        except Exception as e:
            logging.error(f"❌ 重排序失败: {str(e)}")
            # 失败时返回前k个文档
            return docs[:k]
    
    def encode_image_to_base64(self, image_path):
        """将图片转换为base64编码"""
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
            logging.info(f"图片编码失败：{str(e)}")
            return None
    
    def call_qwen_vl(self, image_base64, image_name):
        """调用通义千问视觉模型"""
        client = ChatOpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name=settings.VISION_MODEL_NAME,
            temperature=0,
            timeout=30
        )

        # 重试装饰器
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
    
    def extract_table_from_image(self, image_path, report_file):
        """使用大模型提取表格图片内容"""
        logging.info(f"\n{'=' * 50}")
        logging.info(f"🔍 正在处理表格图片：{os.path.basename(image_path)}")
        logging.info(f"📁 图片路径：{image_path}")

        try:
            # 1. 图片转base64
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return "⚠️ 图片编码失败，无法处理"

            # 2. 调用通义千问识别
            image_name = os.path.basename(image_path)
            logging.info("📡 正在调用通义千问API识别图片...")
            recognition_result = self.call_qwen_vl(image_base64, image_name)

            if not recognition_result:
                return "⚠️ 未识别到任何内容"

            # 3. 处理识别结果
            final_text = f"【表格图片】{image_name}\n{recognition_result}"
            final_text = final_text.replace('\r', '').strip()

            # 4. 实时打印
            logging.info(f"✅ 识别结果：\n{final_text}")

            # 5. 写入报告
            report_file.write(f"【表格图片信息】\n")
            report_file.write(f"文件名：{image_name}\n")
            report_file.write(f"完整路径：{image_path}\n")
            report_file.write(f"识别结果：\n{final_text}\n")
            report_file.write(f"{'=' * 30}\n\n")

            return final_text

        except Exception as e:
            error_msg = f"❌ 处理失败：{str(e)}"
            logging.info(error_msg)
            report_file.write(f"【表格图片信息】\n")
            report_file.write(f"文件名：{os.path.basename(image_path)}\n")
            report_file.write(f"完整路径：{image_path}\n")
            report_file.write(f"错误信息：{error_msg}\n")
            report_file.write(f"{'=' * 30}\n\n")

            return error_msg
    
    def load_all_knowledge(self, root_folder):
        """全量读取知识库（文本+表格图片）+ 生成OCR报告"""
        all_documents = []
        with open(settings.SAVE_OCR_REPORT, "w", encoding="utf-8") as ocr_report:
            ocr_report.write(f"# 京东服务细则表格图片识别报告\n")
            ocr_report.write(f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            ocr_report.write(f"总处理文件夹：{root_folder}\n\n")

            for category in os.listdir(root_folder):
                category_path = os.path.join(root_folder, category)
                if not os.path.isdir(category_path):
                    continue
                logging.info(f"\n📂 开始处理分类：{category}")
                ocr_report.write(f"## 分类：{category}\n")

                # 读取文本文件
                text_file = None
                for file in os.listdir(category_path):
                    if file.endswith(".txt"):
                        text_file = os.path.join(category_path, file)
                        break
                if not text_file:
                    logging.info(f"⚠️ 分类{category}无文本文件，跳过")
                    ocr_report.write(f"警告：该分类无.txt文本文件，已跳过\n")
                    ocr_report.write(f"{'=' * 30}\n\n")
                    continue

                # 读取文本内容
                with open(text_file, "r", encoding="utf-8") as f:
                    text_content = f.read()
                if not text_content:
                    logging.info(f"⚠️ 文本文件{text_file}为空，跳过")
                    ocr_report.write(f"警告：文本文件{os.path.basename(text_file)}内容为空，已跳过\n")
                    ocr_report.write(f"{'=' * 30}\n\n")
                    continue
                logging.info(f"📄 已读取文本文件：{os.path.basename(text_file)}（字数：{len(text_content)}）")

                # 处理表格图片
                image_folder = os.path.join(category_path, settings.IMAGE_SUBFOLDER)
                image_infos = []
                if os.path.exists(image_folder):
                    image_files = [f for f in os.listdir(image_folder) if
                                   f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
                    logging.info(f"🖼️ 该分类下表格图片数量：{len(image_files)}")
                    ocr_report.write(f"表格图片数量：{len(image_files)}\n")

                    for img_file in image_files:
                        img_path = os.path.join(image_folder, img_file)
                        img_text = self.extract_table_from_image(img_path, ocr_report)
                        image_infos.append({
                            "image_name": img_file,
                            "image_path": img_path,
                            "extracted_text": img_text
                        })
                        text_content += f"\n\n【关联表格图片】{img_file}：{img_text}（图片路径：{img_path}）"

                else:
                    logging.info(f"⚠️ 该分类无{settings.IMAGE_SUBFOLDER}子文件夹，跳过图片处理")
                    ocr_report.write(f"警告：无{settings.IMAGE_SUBFOLDER}子文件夹，未处理图片\n")

                # 文本切片
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP,
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
                
                # 为每个chunk添加point-id作为唯一标识符
                text_file_name = os.path.basename(text_file).replace('.txt', '')
                total_chunks = len(split_docs)
                for i, doc in enumerate(split_docs):
                    # 创建包含时间戳的唯一标识符，确保全局唯一
                    import uuid
                    point_id = str(uuid.uuid4())  # 使用UUID确保全局唯一性
                    doc.metadata["point_id"] = point_id
                    doc.metadata["chunk_index"] = i + 1
                    doc.metadata["total_chunks"] = total_chunks
                    doc.metadata["source_file"] = text_file_name
                    doc.metadata["category"] = category
                    doc.metadata["source_id"] = point_id  # 保持向后兼容性
                all_documents.extend(split_docs)
                logging.info(f"✅ 分类{category}处理完成，生成知识块：{len(split_docs)}个")
                ocr_report.write(f"知识块数量：{len(split_docs)}个\n")
                ocr_report.write(f"{'=' * 30}\n\n")

        # 打印统计信息
        logging.info(f"\n{'=' * 60}")
        logging.info(f"📊 知识库读取总统计：")
        logging.info(
            f"   - 处理分类数量：{len([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])}")
        logging.info(f"   - 生成知识块总数：{len(all_documents)}")
        logging.info(f"   - OCR报告已保存至：{settings.SAVE_OCR_REPORT}")
        logging.info(f"{'=' * 60}")
        return all_documents
    
    def build_embeddings(self):
        """构建嵌入向量库"""
        print("🔨 开始构建向量数据库...")
        # 读取知识库
        knowledge_docs = self.load_all_knowledge(settings.ROOT_FOLDER)
        
        if not knowledge_docs:
            print("❌ 未读取到知识库内容，无法构建向量库！")
            return
        
        print(f"\n📈 从通义千问API生成嵌入向量...")
        
        # 分批处理，每批不超过10个文档
        batch_size = 10
        total_docs = len(knowledge_docs)
        print(f"总文档数：{total_docs}，每批处理：{batch_size}个")
        
        # 初始化向量存储
        if total_docs > 0:
            # 先处理第一批
            first_batch = knowledge_docs[:batch_size]
            print(f"处理第1批，共{len(first_batch)}个文档")
            db = FAISS.from_documents(first_batch, self.embeddings)
            
            # 处理剩余批次
            for i in range(batch_size, total_docs, batch_size):
                end_idx = min(i + batch_size, total_docs)
                batch = knowledge_docs[i:end_idx]
                print(f"处理第{int(i/batch_size)+1}批，共{len(batch)}个文档")
                # 为当前批次创建临时向量存储
                temp_db = FAISS.from_documents(batch, self.embeddings)
                # 合并到主向量存储
                db.merge_from(temp_db)
        else:
            # 如果没有文档，创建空的向量存储
            db = FAISS.from_texts([""], self.embeddings)

        # 保存向量库
        os.makedirs(settings.SAVE_EMBED_PATH, exist_ok=True)
        db.save_local(settings.SAVE_EMBED_PATH)

        # 更新向量存储
        self.vector_store = db

        print(f"✅ RAG知识库构建完成！")
        print(f"   - 向量库保存路径：{settings.SAVE_EMBED_PATH}")
        print(f"   - 知识块总数：{len(knowledge_docs)}")


# 创建全局RAG检索器实例
rag_retriever = RAGRetriever()