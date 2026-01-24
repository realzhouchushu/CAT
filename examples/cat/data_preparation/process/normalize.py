#!/usr/bin/env python3
"""
音频归一化脚本
功能：将指定文件夹下的所有音频文件重采样到相同采样率，统一保存为WAV格式
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time


def setup_logging(log_file: str) -> logging.Logger:
    """设置日志记录（多进程安全）"""
    logger = logging.getLogger('audio_normalize')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def get_audio_files(source_dir: str, logger: logging.Logger) -> List[Path]:
    """获取源文件夹下的所有文件（假设都是音频文件）"""
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"源文件夹不存在: {source_dir}")
    
    if not source_path.is_dir():
        raise NotADirectoryError(f"源路径不是文件夹: {source_dir}")
    
    # 直接获取所有文件，不检查扩展名
    audio_files = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            audio_files.append(os.path.join(root, file))
    
    return audio_files


def normalize_audio_worker(args: Tuple[Path, str, int]) -> Tuple[str, bool, str]:
    """多进程工作函数：归一化单个音频文件"""
    input_path, target_dir, target_sr = args
    input_path = Path(input_path)
    
    try:
        # 读取音频文件
        audio, sr = librosa.load(str(input_path), sr=None)
 
        # 检查音频是否为空或过短
        if audio is None or len(audio) < 0.01 * sr:
            return str(input_path), False, f"文件为空音频文件"
        
        # 重采样
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # 确保音频是float32类型，范围在[-1, 1]之间
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 如果音频值超出范围，进行归一化
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        # 构建输出文件路径
        output_file = Path(target_dir) / f"{input_path.stem}.wav"
        
        # 保存为WAV格式
        sf.write(str(output_file), audio, target_sr, subtype='PCM_16')
        
        return str(input_path), True, "成功"
        
    except Exception as e:
        return str(input_path), False, str(e)


def normalize_audio(input_path: Path, output_path: Path, target_sr: int, logger: logging.Logger) -> bool:
    """归一化单个音频文件（单进程版本，保留兼容性）"""
    try:
        # 读取音频文件
        audio, sr = librosa.load(str(input_path), sr=None)
 
        # 检查音频是否为空或过短
        if audio is None or len(audio) < 0.01 * sr:
            logger.warning(f"文件 {input_path} 为空音频文件")
            return False
        
        # 重采样
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # 确保音频是float32类型，范围在[-1, 1]之间
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 如果音频值超出范围，进行归一化
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        # 保存为WAV格式
        sf.write(str(output_path), audio, target_sr, subtype='PCM_16')
        
        return True
        
    except Exception as e:
        logger.error(f"处理文件 {input_path} 时出错: {str(e)}")
        return False


def process_audio_files(source_dir: str, target_dir: str, target_sr: int, logger: logging.Logger, num_processes: int = None) -> None:
    """处理所有音频文件（支持多进程）"""
    # 获取音频文件列表
    audio_files = get_audio_files(source_dir, logger)
    
    if not audio_files:
        logger.warning(f"在源文件夹 {source_dir} 中未找到音频文件")
        return
    
    logger.info(f"找到 {len(audio_files)} 个音频文件")
    
    # 创建目标文件夹
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 确定进程数
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(audio_files))
    else:
        num_processes = min(num_processes, mp.cpu_count(), len(audio_files))
    
    logger.info(f"使用 {num_processes} 个进程进行处理")
    
    # 准备多进程参数
    worker_args = [(audio_file, target_dir, target_sr) for audio_file in audio_files]
    
    # 统计信息
    success_count = 0
    failed_count = 0
    failed_files = []
    
    # 使用多进程处理
    start_time = time.time()
    
    if num_processes > 1:
        # 多进程处理
        with mp.Pool(processes=num_processes) as pool:
            # 使用imap处理，保持顺序
            results = list(tqdm(
                pool.imap(normalize_audio_worker, worker_args),
                total=len(audio_files),
                desc="处理音频文件"
            ))
        
        # 处理结果
        for file_path, success, message in results:
            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_files.append((file_path, message))
    else:
        # 单进程处理（兼容性）
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            output_file = target_path / f"{audio_file.stem}.wav"
            if normalize_audio(audio_file, output_file, target_sr, logger):
                success_count += 1
            else:
                failed_count += 1
                failed_files.append((str(audio_file), "处理失败"))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 记录失败的文件
    if failed_files:
        logger.info("失败的文件列表:")
        for failed_file, message in failed_files:
            logger.info(f"  - {failed_file}: {message}")

    # 记录统计信息
    logger.info(f"处理完成！成功: {success_count}, 失败: {failed_count}")
    logger.info(f"总处理时间: {processing_time:.2f}秒")
    if success_count > 0:
        avg_time = processing_time / success_count
        logger.info(f"平均每文件处理时间: {avg_time:.3f}秒")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="音频归一化脚本 - 重采样音频到指定采样率并保存为WAV格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
     示例用法:
     python normalize.py --source_dir /path/to/source --target_dir /path/to/target --sample_rate 16000 --log_file /path/to/error.log
     python normalize.py -s /path/to/source -t /path/to/target -r 22050 -l /path/to/error.log
     python normalize.py -s /path/to/source -t /path/to/target -r 16000 -l /path/to/error.log -p 8
        """
    )
    
    parser.add_argument(
        '--source_dir', '-s',
        required=True,
        help='源文件夹路径（包含音频文件）'
    )
    
    parser.add_argument(
        '--target_dir', '-t',
        required=True,
        help='目标文件夹路径（输出WAV文件）'
    )
    
    parser.add_argument(
        '--sample_rate', '-r',
        type=int,
        required=True,
        help='目标采样率（Hz）'
    )
    
    parser.add_argument(
        '--log_file', '-l',
        required=True,
        help='日志文件路径（记录错误信息）'
    )
    
    parser.add_argument(
        '--num_processes', '-p',
        type=int,
        default=None,
        help='使用的进程数（默认使用所有可用CPU核心）'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 验证参数
    if args.sample_rate <= 0:
        print("错误：采样率必须大于0")
        sys.exit(1)
    
    # 设置日志
    logger = setup_logging(args.log_file)
    
    try:
        logger.info(f"开始音频归一化处理")
        logger.info(f"源文件夹: {args.source_dir}")
        logger.info(f"目标文件夹: {args.target_dir}")
        logger.info(f"目标采样率: {args.sample_rate} Hz")
        logger.info(f"日志文件: {args.log_file}")
        if args.num_processes:
            logger.info(f"指定进程数: {args.num_processes}")
        
        # 处理音频文件
        process_audio_files(args.source_dir, args.target_dir, args.sample_rate, logger, args.num_processes)
        
        logger.info("音频归一化处理完成！")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
