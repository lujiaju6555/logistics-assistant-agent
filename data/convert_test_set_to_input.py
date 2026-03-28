"""
黄金测试集转换器

将黄金测试集（JSONL格式）转换为系统输入格式（JSON格式）
"""

import json
import os
from pathlib import Path
import argparse


def convert_golden_to_input(golden_file_path: str, output_file_path: str, users_count: int = 5):
    """
    将黄金测试集转换为系统输入格式

    Args:
        golden_file_path: 黄金测试集文件路径 (JSONL格式)
        output_file_path: 输出文件路径 (JSON格式)
        users_count: 生成的用户数量
    """
    # 读取所有测试案例
    all_cases = []

    with open(golden_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                qa_data = json.loads(line)

                # 创建测试案例，保留所有原始信息
                test_case = {
                    "question": qa_data["question"],
                    "expected_answer": qa_data["answer"],
                    "expected_sources": qa_data["source_ids"],  # 期望的来源
                    "difficulty": qa_data["difficulty"],
                    "reasoning": qa_data.get("reasoning", "")
                }

                all_cases.append(test_case)

            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析失败: {e}")
                continue
    
    # 按用户分组
    user_cases = {}
    cases_per_user = len(all_cases) // users_count
    remainder = len(all_cases) % users_count
    
    start = 0
    for i in range(users_count):
        user_id = f"user{i+1}"
        count = cases_per_user + (1 if i < remainder else 0)
        end = start + count
        user_cases[user_id] = all_cases[start:end]
        start = end

    # 保存为JSON格式
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(user_cases, f, ensure_ascii=False, indent=2)

    print(f"转换完成! 共转换 {len(all_cases)} 个测试案例，分布到 {users_count} 个用户")
    print(f"输出文件: {output_file_path}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='黄金测试集转换器')
    parser.add_argument('--golden-file', type=str, default='./data/golden_test_set.jsonl',
                        help='黄金测试集文件路径 (JSONL格式)')
    parser.add_argument('--output-file', type=str, default='./data/test_input.json',
                        help='输出文件路径 (JSON格式)')
    parser.add_argument('--users-count', type=int, default=10,
                        help='生成的用户数量')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 转换为系统输入格式
    convert_golden_to_input(args.golden_file, args.output_file, users_count=args.users_count)