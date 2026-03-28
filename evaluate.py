"""
RAG检索效果评估器

评估RAG系统的检索效果，计算hit@k和recall等指标
"""

import json
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict


def calculate_hit_at_k(retrieved_docs: List[Dict], expected_sources: List[str], k: int) -> bool:
    """
    计算Hit@K指标

    Args:
        retrieved_docs: 检索到的文档列表
        expected_sources: 期望的文档来源列表
        k: K值

    Returns:
        是否在前K个结果中找到期望文档
    """
    if not expected_sources:
        return True  # 如果没有期望来源，认为命中

    # 获取前k个检索到的文档的来源
    retrieved_sources = []
    for doc in retrieved_docs[:k]:
        if 'metadata' in doc and 'source_id' in doc['metadata']:
            retrieved_sources.append(doc['metadata']['source_id'])
        else:
            # 如果没有source_id，尝试从其他字段获取
            retrieved_sources.append(doc.get('id', ''))

    # 检查是否包含任何期望的来源
    for expected_source in expected_sources:
        if expected_source in retrieved_sources:
            return True

    return False


def calculate_recall(retrieved_docs: List[Dict], expected_sources: List[str]) -> float:
    """
    计算召回率

    Args:
        retrieved_docs: 检索到的文档列表
        expected_sources: 期望的文档来源列表

    Returns:
        召回率 (0-1之间)
    """
    if not expected_sources:
        return 1.0  # 如果没有期望来源，召回率为1

    # 获取所有检索到的文档来源
    retrieved_sources = []
    for doc in retrieved_docs:
        if 'metadata' in doc and 'source_id' in doc['metadata']:
            retrieved_sources.append(doc['metadata']['source_id'])
        else:
            retrieved_sources.append(doc.get('id', ''))

    # 计算召回率
    hits = 0
    for expected_source in expected_sources:
        if expected_source in retrieved_sources:
            hits += 1

    return hits / len(expected_sources)


def evaluate_rag_system(test_cases: List[Dict], k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
    """
    评估RAG系统

    Args:
        test_cases: 测试案例列表
        k_values: 用于计算Hit@K的K值列表

    Returns:
        评估结果字典
    """
    results = {
        'total_cases': len(test_cases),
        'hit_at_k': {},
        'recall': [],
        'by_difficulty': defaultdict(lambda: {'hit_at_k': {}, 'recall': [], 'count': 0})
    }

    for case in test_cases:
        expected_sources = case.get('expected_sources', [])
        retrieved_docs = case.get('retrieved_docs', [])
        difficulty = case.get('difficulty', 'Unknown')

        # 计算各种K值下的Hit@K
        for k in k_values:
            hit = calculate_hit_at_k(retrieved_docs, expected_sources, k)
            if f'hit@{k}' not in results['hit_at_k']:
                results['hit_at_k'][f'hit@{k}'] = 0
            results['hit_at_k'][f'hit@{k}'] += hit

            # 按难度分级统计
            if f'hit@{k}' not in results['by_difficulty'][difficulty]['hit_at_k']:
                results['by_difficulty'][difficulty]['hit_at_k'][f'hit@{k}'] = 0
            results['by_difficulty'][difficulty]['hit_at_k'][f'hit@{k}'] += hit

        # 计算召回率
        recall = calculate_recall(retrieved_docs, expected_sources)
        results['recall'].append(recall)
        results['by_difficulty'][difficulty]['recall'].append(recall)
        results['by_difficulty'][difficulty]['count'] += 1

    # 计算平均值
    for k_metric in results['hit_at_k']:
        results['hit_at_k'][k_metric] /= len(test_cases)

    results['avg_recall'] = np.mean(results['recall']) if results['recall'] else 0

    # 计算各难度级别的指标
    for difficulty in results['by_difficulty']:
        difficulty_data = results['by_difficulty'][difficulty]
        for k_metric in difficulty_data['hit_at_k']:
            if difficulty_data['count'] > 0:
                difficulty_data['hit_at_k'][k_metric] /= difficulty_data['count']
        difficulty_data['avg_recall'] = np.mean(difficulty_data['recall']) if difficulty_data['recall'] else 0

    return results


def print_evaluation_results(results: Dict[str, Any]):
    """
    打印评估结果

    Args:
        results: 评估结果字典
    """
    print("=" * 60)
    print("RAG系统评估结果")
    print("=" * 60)
    print(f"总测试案例数: {results['total_cases']}")
    print(f"平均召回率: {results['avg_recall']:.4f}")

    print("\nHit@K 指标:")
    for k_metric, value in results['hit_at_k'].items():
        print(f"  {k_metric}: {value:.4f}")

    print("\n按难度分级的指标:")
    for difficulty, data in results['by_difficulty'].items():
        print(f"\n  {difficulty}: (共{data['count']}个案例)")
        print(f"    平均召回率: {data['avg_recall']:.4f}")
        for k_metric, value in data['hit_at_k'].items():
            print(f"    {k_metric}: {value:.4f}")

    print("=" * 60)


def load_test_cases(input_file_path: str) -> List[Dict]:
    """
    加载测试案例

    Args:
        input_file_path: 输入文件路径

    Returns:
        测试案例列表
    """
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理嵌套的用户结构
    test_cases = []
    if isinstance(data, dict):
        # 遍历所有用户
        for user_id, user_cases in data.items():
            if isinstance(user_cases, list):
                for case in user_cases:
                    test_cases.append(case)
    elif isinstance(data, list):
        # 直接是案例列表
        test_cases = data
    
    return test_cases


def run_evaluation(result_file: str, k_values: List[int] = [1, 3, 5]):
    """
    运行完整评估流程

    Args:
        result_file: 结果文件路径
        k_values: Hit@K的K值列表
    """
    # 加载测试结果
    test_cases = load_test_cases(result_file)

    # 执行评估
    results = evaluate_rag_system(test_cases, k_values)

    # 打印结果
    print_evaluation_results(results)

    # 保存详细结果
    output_path = result_file.replace('.json', '_evaluation.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细评估结果已保存至: {output_path}")


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='评估RAG系统检索结果')
    parser.add_argument('--result_file', type=str, default="./result/output7.json",
                        help='结果文件路径 (JSONL格式)')

    args = parser.parse_args()

    # 运行评估
    run_evaluation(args.result_file)
