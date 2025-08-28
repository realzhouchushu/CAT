#!/usr/bin/env python3
"""
AudioSet数据准备脚本
将tsv和lbl文件按basename匹配生成json文件
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_tsv_data(tsv_path: str) -> Tuple[str, Dict[str, str]]:
    """
    加载tsv文件数据
    返回: (公共路径, {basename: 音频相对路径})
    """
    tsv_data = {}
    common_path = ""
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            raise ValueError("TSV文件为空")
        
        # 第一行为公共路径
        common_path = lines[0].strip()
        
        # 后续每行为音频信息
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                audio_rel_path = parts[0]  # 第一列为音频相对路径
                # 获取basename（去掉扩展名）
                basename = os.path.splitext(os.path.basename(audio_rel_path))[0]
                tsv_data[basename] = audio_rel_path
    
    return common_path, tsv_data


def load_lbl_data(lbl_path: str) -> Dict[str, str]:
    """
    加载lbl文件数据
    返回: {basename: 标签字符串}
    """
    lbl_data = {}
    
    with open(lbl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    basename = parts[0]  # 第一列为basename
                    labels = ' '.join(parts[1:])  # 剩余部分为标签
                    lbl_data[basename] = labels
    
    return lbl_data


def generate_json_data(common_path: str, tsv_data: Dict[str, str], 
                      lbl_data: Dict[str, str]) -> List[Dict]:
    """
    生成json数据
    """
    json_data = []
    
    # 遍历tsv数据，按basename匹配lbl数据
    for basename, audio_rel_path in tsv_data.items():
        # 检查是否有对应的标签
        if basename in lbl_data:
            # 构建完整音频路径
            wav_path = os.path.join(common_path, audio_rel_path)
            
            # 构建数据项
            item = {
                "video_id": basename[1:] if basename.startswith('-') else basename,  # basename[1:]如果以-开头
                "wav": wav_path,
                "image": "",  # 默认为空
                "labels": lbl_data[basename]
            }
            
            json_data.append(item)
        else:
            print(f"警告: 未找到basename '{basename}' 对应的标签")
    
    return json_data


def main():
    parser = argparse.ArgumentParser(description="AudioSet数据准备脚本")
    parser.add_argument('--tsv_path', '-t', 
                       default='/opt/gpfs/home/chushu/data/audioset/setting/AST/unbal_train.tsv',
                       help='TSV文件路径')
    parser.add_argument('--lbl_path', '-l',
                       default='/opt/gpfs/home/chushu/data/audioset/setting/AST/unbal_train.lbl',
                       help='LBL文件路径')
    parser.add_argument('--output_path', '-o',
                       default='/opt/gpfs/home/chushu/data/audioset/setting/AST/unbal_train.json',
                       help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    try:
        print(f"加载TSV文件: {args.tsv_path}")
        common_path, tsv_data = load_tsv_data(args.tsv_path)
        print(f"公共路径: {common_path}")
        print(f"TSV数据条目数: {len(tsv_data)}")
        
        print(f"加载LBL文件: {args.lbl_path}")
        lbl_data = load_lbl_data(args.lbl_path)
        print(f"LBL数据条目数: {len(lbl_data)}")
        
        print("生成JSON数据...")
        json_data = generate_json_data(common_path, tsv_data, lbl_data)
        print(f"匹配成功的数据条目数: {len(json_data)}")
        
        # 构建最终输出格式
        output_data = {
            "data": json_data
        }
        
        # 保存JSON文件
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"JSON文件已保存到: {args.output_path}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
