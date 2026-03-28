import concurrent.futures
import os
import argparse
from typing import Dict, Any
from utils import PipelineState, memory_manager, data_loader
from modules import (
    intent_recognizer, query_rewriter, information_extractor,
    tool_caller, rag_retriever, answer_generator, risk_detector
)
from config import settings


class MainPipeline:
    """主流水线"""

    def process_user_input(self, user_id: str, question: str) -> Dict[str, Any]:
        """处理用户输入"""
        # 加载对话历史和用户结构化信息
        chat_history = memory_manager.load_history(user_id)
        user_structured_info = memory_manager.get_user_structured_info(user_id)

        # 初始化状态对象
        state = PipelineState(
            user_id=user_id,
            original_query=question,
            chat_history=chat_history
        )
        
        # 设置初始结构化信息为用户历史信息
        if user_structured_info:
            state.update(extracted_params=user_structured_info)

        try:
            # 1. 并行感知层
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 并行执行意图识别和问题重写
                intent_future = executor.submit(intent_recognizer.recognize_intent, question)
                rewrite_future = executor.submit(query_rewriter.rewrite_query, question, chat_history)

                # 获取结果
                intent = intent_future.result()
                rewritten_query = rewrite_future.result()

            # 更新状态
            state.update(intent=intent, rewritten_query=rewritten_query)

            # 2. 信息抽取层
            intent_type = intent.get("intent_type", "HUMAN_TRANSFER")
            if intent_type == "BUSINESS_QUERY":
                # 使用重写后的问题进行信息抽取
                extracted_params = information_extractor.extract_information(rewritten_query)
                
                # 合并新抽取的信息和历史信息
                if user_structured_info:
                    # 用新抽取的非空信息覆盖历史信息
                    for key, value in extracted_params.items():
                        if value is not None:
                            user_structured_info[key] = value
                    extracted_params = user_structured_info
                
                # 更新状态
                state.update(extracted_params=extracted_params)
                
                # 保存更新后的结构化信息
                memory_manager.save_user_structured_info(user_id, extracted_params)
            else:
                # 非业务查询，使用历史结构化信息
                state.update(extracted_params=user_structured_info)

            # 3. 执行与回答层
            if intent_type in ["CHITCHAT", "RESET", "HUMAN_TRANSFER"]:
                # 直接生成回答
                answer = answer_generator.generate_answer(
                    intent=intent,
                    rewritten_query=question,  # 使用原始问题
                    extracted_params={},
                    tool_results={},
                    rag_context=[]
                )
            else:
                # 业务查询
                # 工具调用
                tool_results = tool_caller.route_and_call(rewritten_query, state.extracted_params, intent, data_loader)

                # RAG检索（使用多路召回 + 全局重排序）
                rag_context = rag_retriever.retrieve_documents(question, rewritten_query)
                
                # 更新状态
                state.update(tool_results=tool_results, rag_context=rag_context)
                
                # 生成回答
                answer = answer_generator.generate_answer(
                    intent=intent,
                    rewritten_query=rewritten_query,
                    extracted_params=state.extracted_params,
                    tool_results=tool_results,
                    rag_context=rag_context
                )

            # 4. 后置风控层
            safe_answer = risk_detector.filter_risk(answer)
            state.update(final_answer=safe_answer)

            # 5. 更新对话历史
            updated_history = memory_manager.add_message(chat_history, "user", question)
            updated_history = memory_manager.add_message(updated_history, "assistant", safe_answer)
            memory_manager.save_history(updated_history, user_id)

            state.update(chat_history=updated_history)

        except Exception as e:
            state.update(
                is_success=False,
                error_message=str(e),
                final_answer="抱歉，处理您的请求时出现错误，请稍后重试。"
            )

        return state.to_dict()


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def process_batch(self, input_file, output_file):
        """批处理模式：从JSON文件读取输入并保存输出"""
        print(f"🔔 京东物流智能客服系统启动中...")
        print(f"📥 读取输入文件: {input_file}")
        
        # 检查并构建向量数据库
        if not os.path.exists(settings.SAVE_EMBED_PATH) or not os.listdir(settings.SAVE_EMBED_PATH):
            print("🏗️  向量数据库不存在，开始构建...")
            rag_retriever.build_embeddings()
        else:
            print("📚 向量数据库已存在，直接加载")
        
        # 读取输入文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                import json
                input_data = json.load(f)
        except Exception as e:
            print(f"❌ 读取输入文件失败: {e}")
            return
        
        # 处理每个用户的对话
        output_data = {}
        for user_id, dialogues in input_data.items():
            print(f"\n" + "=" * 60)
            print(f"👤 处理用户: {user_id}")
            
            # 初始化用户对话历史
            user_chat_history = []
            user_results = []
            
            # 处理每轮对话
            for i, dialogue in enumerate(dialogues):
                question = dialogue.get('question')
                if not question:
                    continue
                
                print(f"\n💬 处理第{i+1}轮对话: {question}")
                
                # 处理用户输入
                result = self.pipeline.process_user_input(user_id, question)
                
                # 构建对话历史
                user_chat_history.append({"role": "user", "content": question})
                user_chat_history.append({"role": "assistant", "content": result.get('final_answer', '')})
                
                # 保存结果
                user_results.append({
                    "user_question": question,
                    "system_response": result.get('final_answer', ''),
                    "processing_details": {
                        "intent": result.get('intent', {}),
                        "rewritten_query": result.get('rewritten_query', ''),
                        "extracted_params": result.get('extracted_params', {}),
                        "tool_results": result.get('tool_results', {}),
                        "rag_context": result.get('rag_context', []),
                        "risk_info": result.get('risk_info', {})
                    }
                })
            
            output_data[user_id] = user_results
        
        # 保存输出文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n" + "=" * 60)
            print(f"✅ 处理完成！")
            print(f"📤 输出文件已保存至: {output_file}")
            print(f"=" * 60)
        except Exception as e:
            print(f"❌ 保存输出文件失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="京东物流智能客服系统")
    parser.add_argument('--input', type=str, default='input5.json', help='输入文件路径')
    parser.add_argument('--output', type=str, default='output5.json', help='输出文件路径')
    
    args = parser.parse_args()
    
    pipeline = MainPipeline()

    # 批处理模式
    processor = BatchProcessor(pipeline)
    processor.process_batch(args.input, args.output)