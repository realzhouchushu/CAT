#!/usr/bin/env python3
"""
将CSV文件转换为fairseq训练用的TSV文件，并统计音频时长
CSV格式: audio_id, hdf5_path
TSV格式: root_dir\nfile_path\tduration
"""

import argparse
import os
import sys
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_audio_duration_from_h5(h5_path, audio_id, sample_rate=32000):
    """从HDF5文件中获取音频时长"""
    # return 10
    try:
        with h5py.File(h5_path, 'r') as f:
            if audio_id in f:
                audio_data = f[audio_id][:]
                duration = len(audio_data) / sample_rate
                return duration
            else:
                logger.warning(f"音频ID {audio_id} 在文件 {h5_path} 中未找到")
                return None
    except Exception as e:
        logger.error(f"读取HDF5文件 {h5_path} 中的 {audio_id} 时出错: {e}")
        return None


def convert_csv_to_tsv(csv_path, output_dir, sample_rate=32000, valid_percent=0.1, seed=42):
    """
    将CSV文件转换为fairseq训练用的TSV文件
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
        sample_rate: 采样率（Hz）
        valid_percent: 验证集比例
        seed: 随机种子
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    logger.info(f"读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path, sep='\t')
    
    # 检查列名
    if 'audio_id' not in df.columns or 'hdf5_path' not in df.columns:
        logger.error("CSV文件必须包含 'audio_id' 和 'hdf5_path' 列")
        return
    
    # 替换路径
    old_path = "/hpc_stor03/public/shared/data/raa/AudioSet"
    new_path = "/opt/gpfs/data/raw_data/AudioSet"
    
    # 统计替换数量
    df['hdf5_path'] = df['hdf5_path'].str.replace(old_path, new_path, regex=False)
    
    logger.info(f"找到 {len(df)} 个音频文件")
    
    # 统计音频时长
    logger.info("开始统计音频时长...")
    durations = []
    entries = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理音频文件"):
        audio_id = row['audio_id']
        h5_path = row['hdf5_path']
        
        # 获取音频时长
        duration = get_audio_duration_from_h5(h5_path, audio_id, sample_rate)
        
        if duration is not None:
            # 转换为帧数（fairseq使用帧数而不是秒数）
            frames = int(duration * sample_rate)
            durations.append(duration)
            entries.append({
                'audio_id': audio_id,
                'hdf5_path': h5_path,
                'duration': duration,
                'frames': frames
            })
        else:
            logger.warning(f"跳过无效的音频文件: {audio_id}")
    
    logger.info(f"成功处理 {len(entries)} 个音频文件")
    
    # 统计信息
    if durations:
        durations = np.array(durations)
        logger.info(f"音频时长统计:")
        logger.info(f"  总时长: {durations.sum():.2f} 秒 ({durations.sum()/3600:.2f} 小时)")
        logger.info(f"  平均时长: {durations.mean():.2f} 秒")
        logger.info(f"  最短时长: {durations.min():.2f} 秒")
        logger.info(f"  最长时长: {durations.max():.2f} 秒")
        logger.info(f"  标准差: {durations.std():.2f} 秒")
    
    # 随机分割训练集和验证集
    np.random.seed(seed)
    n_valid = int(len(entries) * valid_percent)
    indices = np.random.permutation(len(entries))
    valid_indices = indices[:n_valid]
    train_indices = indices[n_valid:]
    
    # 获取根目录（所有HDF5文件的公共父目录）
    h5_paths = [entry['hdf5_path'] for entry in entries]
    root_dir = os.path.commonpath(h5_paths)
    logger.info(f"根目录: {root_dir}")
    
    # 写入训练集TSV文件
    train_tsv_path = os.path.join(output_dir, "train.tsv")
    logger.info(f"写入训练集TSV文件: {train_tsv_path}")
    
    with open(train_tsv_path, 'w', encoding='utf-8') as f:
        # 写入根目录
        print(root_dir, file=f)
        
        # 写入训练数据
        for idx in train_indices:
            entry = entries[idx]
            # 构建相对路径：hdf5文件名/音频ID
            h5_filename = os.path.basename(entry['hdf5_path'])
            relative_path = f"{h5_filename}/{entry['audio_id']}"
            frames = entry['frames']
            print(f"{relative_path}\t{frames}", file=f)
    
    # 写入验证集TSV文件
    if n_valid > 0:
        valid_tsv_path = os.path.join(output_dir, "valid.tsv")
        logger.info(f"写入验证集TSV文件: {valid_tsv_path}")
        
        with open(valid_tsv_path, 'w', encoding='utf-8') as f:
            # 写入根目录
            print(root_dir, file=f)
            
            # 写入验证数据
            for idx in valid_indices:
                entry = entries[idx]
                h5_filename = os.path.basename(entry['hdf5_path'])
                relative_path = f"{h5_filename}/{entry['audio_id']}"
                frames = entry['frames']
                print(f"{relative_path}\t{frames}", file=f)
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, "audio_stats.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("音频文件统计信息\n")
        f.write("=" * 50 + "\n")
        f.write(f"总文件数: {len(entries)}\n")
        f.write(f"训练集文件数: {len(train_indices)}\n")
        f.write(f"验证集文件数: {len(valid_indices)}\n")
        f.write(f"总时长: {durations.sum():.2f} 秒 ({durations.sum()/3600:.2f} 小时)\n")
        f.write(f"平均时长: {durations.mean():.2f} 秒\n")
        f.write(f"最短时长: {durations.min():.2f} 秒\n")
        f.write(f"最长时长: {durations.max():.2f} 秒\n")
        f.write(f"标准差: {durations.std():.2f} 秒\n")
        f.write(f"采样率: {sample_rate} Hz\n")
    
    logger.info(f"统计信息已保存到: {stats_path}")
    logger.info("转换完成！")


def main():
    parser = argparse.ArgumentParser(description="将CSV文件转换为fairseq训练用的TSV文件")
    parser.add_argument("csv_path", help="输入的CSV文件路径")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("--sample-rate", type=int, default=32000, help="采样率（Hz）")
    parser.add_argument("--valid-percent", type=float, default=0, help="验证集比例（0-1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        logger.error(f"CSV文件不存在: {args.csv_path}")
        sys.exit(1)
    
    convert_csv_to_tsv(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        valid_percent=args.valid_percent,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
