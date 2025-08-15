#!/usr/bin/env python3
"""
AudioSet ontology id到数字索引转换脚本
读取ontology.json文件，为每个ontology id分配数字索引，保存为CSV格式
"""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List


def load_ontology(ontology_path: str) -> List[Dict]:
    """
    加载ontology.json文件
    
    Args:
        ontology_path: ontology.json文件路径
        
    Returns:
        list: ontology元数据列表
    """
    try:
        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology_data = json.load(f)
        print(f"成功加载ontology文件，包含 {len(ontology_data)} 个条目")
        return ontology_data
    except Exception as e:
        print(f"错误：无法加载ontology文件 {ontology_path}: {e}")
        return []


def extract_all_ontology_ids(ontology_data: List[Dict]) -> List[str]:
    """
    从ontology数据中提取所有ontology id（包括子类别）
    
    Args:
        ontology_data: ontology元数据列表
        
    Returns:
        list: 所有ontology id的列表
    """
    all_ids = set()
    
    def extract_ids_recursive(ontology_list):
        for item in ontology_list:
            if 'id' in item:
                all_ids.add(item['id'])
            
            # 递归处理子类别
            if 'child_ids' in item and item['child_ids']:
                extract_ids_recursive([{'id': child_id} for child_id in item['child_ids']])
    
    extract_ids_recursive(ontology_data)
    
    # 转换为列表并排序，确保输出的一致性
    sorted_ids = sorted(list(all_ids))
    print(f"提取了 {len(sorted_ids)} 个唯一的ontology id")
    return sorted_ids


def save_ontology_mapping(ontology_ids: List[str], output_path: str):
    """
    保存ontology id到数字索引的映射为CSV文件
    
    Args:
        ontology_ids: ontology id列表
        output_path: 输出CSV文件路径
    """
    # 确保输出目录存在
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入映射数据（不含表头）
            for idx, ontology_id in enumerate(ontology_ids):
                writer.writerow([idx, ontology_id])
        
        print(f"成功保存ontology映射到: {output_path}")
        print(f"映射范围: 0 到 {len(ontology_ids) - 1}")
        
    except Exception as e:
        print(f"错误：无法保存CSV文件 {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='AudioSet ontology id到数字索引转换')
    parser.add_argument('--ontology_path', 
                       default='/opt/gpfs/data/raw_data/Audioset/raw_data/ontology.json',
                       help='ontology.json文件路径')
    parser.add_argument('--output_path',
                       default='/opt/gpfs/home/chushu/data/AudioSet/ontology2idx.csv',
                       help='输出CSV文件路径')
    
    args = parser.parse_args()
    
    print("开始处理AudioSet ontology...")
    print(f"输入文件: {args.ontology_path}")
    print(f"输出文件: {args.output_path}")
    
    # 加载ontology数据
    ontology_data = load_ontology(args.ontology_path)
    if not ontology_data:
        print("无法加载ontology数据，程序退出")
        return
    
    # 提取所有ontology id
    ontology_ids = extract_all_ontology_ids(ontology_data)
    if not ontology_ids:
        print("没有找到ontology id，程序退出")
        return
    
    # 保存映射文件
    save_ontology_mapping(ontology_ids, args.output_path)
    
    print("处理完成！")


if __name__ == "__main__":
    main()
