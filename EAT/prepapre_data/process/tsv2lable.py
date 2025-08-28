#!/usr/bin/env python3
"""
TSV文件到标签文件转换脚本
读取TSV文件中的音频顺序，从CSV文件中匹配对应标签，生成.lbl文件
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, List


def load_tsv_audio_order(tsv_path: str) -> List[str]:
    """
    从TSV文件中加载音频文件顺序（只取basename）
    
    Args:
        tsv_path: TSV文件路径
        
    Returns:
        list: 音频文件basename列表，保持TSV中的顺序
    """
    audio_basenames = []
    
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            f.readline() # 跳过第一行公共路径
            for line in f:
                line = line.strip()
                if line:
                    # 分割TSV行，取第一列（音频文件路径）
                    parts = line.split('\t')
                    if parts:
                        audio_path = parts[0]
                        # 提取basename（去掉路径和扩展名）
                        audio_basename = Path(audio_path).stem
                        audio_basenames.append(audio_basename)
        
        print(f"从TSV文件中加载了 {len(audio_basenames)} 个音频文件")
        return audio_basenames
        
    except Exception as e:
        print(f"错误：无法读取TSV文件 {tsv_path}: {e}")
        return []


def load_csv_labels(csv_path: str) -> Dict[str, str]:
    """
    从CSV文件中加载YTID到标签的映射关系
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        dict: YTID到标签字符串的映射字典
    """
    ytid_to_labels = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # 跳过注释行（以#开头）和空行
                if not line or line.startswith('#'):
                    continue
                
                # 手动解析CSV行，正确处理引号内的逗号
                parts = []
                current_part = ""
                in_quotes = False
                i = 0
                
                while i < len(line):
                    char = line[i]
                    
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                    i += 1
                
                # 添加最后一个部分
                parts.append(current_part.strip())
                
                # 确保行有足够的列
                if len(parts) >= 4:
                    ytid = parts[0]
                    labels = parts[3]
                    
                    # 去掉标签中的引号
                    labels = labels.strip('"')
                    
                    if ytid and labels:
                        ytid_to_labels[ytid] = labels
                else:
                    print(f"警告：第 {line_num} 行列数不足: {parts}")
        
        print(f"从CSV文件中加载了 {len(ytid_to_labels)} 个YTID到标签的映射")
        return ytid_to_labels
        
    except Exception as e:
        print(f"错误：无法读取CSV文件 {csv_path}: {e}")
        return {}


def generate_label_file(audio_basenames: List[str], ytid_to_labels: Dict[str, str], 
                        output_path: str, split_name: str):
    """
    生成标签文件
    
    Args:
        audio_basenames: 音频文件basename列表
        ytid_to_labels: YTID到标签的映射字典
        output_path: 输出目录路径
        split_name: 分割名称（用于生成.lbl文件名）
    """
    # 确保输出目录存在
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成.lbl文件路径
    lbl_file_path = output_dir / f"{split_name}.lbl"
    
    try:
        with open(lbl_file_path, 'w', encoding='utf-8') as f:
            for audio_basename in audio_basenames:
                # 查找对应的标签
                if audio_basename[1:] in ytid_to_labels:
                    labels = ytid_to_labels[audio_basename[1:]]
                    # 写入：音频basename + 空格 + 标签
                    f.write(f"{audio_basename} {labels}\n")
                else:
                    # 如果没有找到标签，使用空标签
                    print(f"警告：音频文件 {audio_basename} 没有找到对应的标签")
                    f.write(f"{audio_basename} \n")
        
        print(f"成功生成标签文件: {lbl_file_path}")
        print(f"包含 {len(audio_basenames)} 个音频文件的标签")
        
    except Exception as e:
        print(f"错误：无法写入标签文件 {lbl_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='TSV文件到标签文件转换')
    parser.add_argument('--tsv_path', 
                       default='/opt/gpfs/home/chushu/data/AudioSet/16k_wav_tsv/bal_train.tsv',
                       help='输入TSV文件路径')
    parser.add_argument('--csv_path',
                       default='/opt/gpfs/data/raw_data/Audioset/raw_data/balanced_train_segments.csv',
                       help='标签CSV文件路径')
    parser.add_argument('--output_dir',
                       default='/opt/gpfs/home/chushu/data/AudioSet/16k_wav_tsv',
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    print("开始TSV到标签文件转换...")
    print(f"TSV文件: {args.tsv_path}")
    print(f"CSV文件: {args.csv_path}")
    print(f"输出目录: {args.output_dir}")
    
    # 获取分割名称（TSV文件的basename）
    split_name = Path(args.tsv_path).stem
    print(f"分割名称: {split_name}")
    
    # 加载TSV文件中的音频顺序
    audio_basenames = load_tsv_audio_order(args.tsv_path)
    if not audio_basenames:
        print("无法加载TSV文件中的音频顺序，程序退出")
        return
    
    # 加载CSV文件中的标签映射
    ytid_to_labels = load_csv_labels(args.csv_path)
    if not ytid_to_labels:
        print("无法加载CSV文件中的标签映射，程序退出")
        return
    
    # 生成标签文件
    generate_label_file(audio_basenames, ytid_to_labels, args.output_dir, split_name)
    
    print("转换完成！")


if __name__ == "__main__":
    main()
