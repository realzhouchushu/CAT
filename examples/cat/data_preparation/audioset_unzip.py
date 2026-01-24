import os
import tarfile
import glob
import time
from pathlib import Path

def extract_tar_files():
    # 源目录和目标目录
    source_dir = "/opt/gpfs/data/raw_data/Audioset/audioset_zip/data"
    target_dir = "/opt/gpfs/data/raw_data/Audioset/raw_data"
    count_dir = "/opt/gpfs/data/raw_data/Audioset/raw_data/audio/unbal_train"
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(count_dir, exist_ok=True)
    
    # 获取所有unbal_train*.tar文件
    tar_pattern = os.path.join(source_dir, "unbal_train*.tar")
    tar_files = sorted(glob.glob(tar_pattern))

    skip_count = 277
    tar_files = tar_files[skip_count:]
    
    print(f"找到 {len(tar_files)} 个tar文件需要解压")
    print(f"目标目录: {target_dir}")
    print(f"统计目录: {count_dir}")
    print("-" * 60)
    
    # 记录开始时间
    start_time = time.time()
    
    for i, tar_file in enumerate(tar_files, 1):
        tar_name = os.path.basename(tar_file)
        print(f"[{i:3d}/{len(tar_files)}] 正在解压: {tar_name}")
        
        try:
            # 解压tar文件
            with tarfile.open(tar_file, 'r') as tar:
                tar.extractall(path=target_dir)
            
            # 统计解压后的文件数量(文件数量大统计文件数耗时严重)
            # if os.path.exists(count_dir):
            #     file_count = len([f for f in os.listdir(count_dir) if os.path.isfile(os.path.join(count_dir, f))])
            #     dir_count = len([d for d in os.listdir(count_dir) if os.path.isdir(os.path.join(count_dir, d))])
            #     total_count = file_count + dir_count
            #     print(f"    ✓ 解压完成 | 统计目录文件总数: {total_count} (文件: {file_count}, 目录: {dir_count})")
            # else:
            #     print(f"    ✓ 解压完成 | 统计目录不存在")
                
        except Exception as e:
            print(f"    ✗ 解压失败: {e}")
            continue
        
        # 每解压10个文件显示一次进度
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(tar_files) - i)
            print(f"    进度: {i}/{len(tar_files)} | 已用时: {elapsed:.1f}s | 预计剩余: {remaining:.1f}s")
    
    # 最终统计
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"解压完成! 总共处理 {len(tar_files)} 个文件")
    print(f"总用时: {total_time:.1f} 秒")
    print(f"平均每个文件: {total_time/len(tar_files):.1f} 秒")
    
    # 最终文件统计
    if os.path.exists(count_dir):
        final_file_count = len([f for f in os.listdir(count_dir) if os.path.isfile(os.path.join(count_dir, f))])
        final_dir_count = len([d for d in os.listdir(count_dir) if os.path.isdir(os.path.join(count_dir, d))])
        final_total = final_file_count + final_dir_count
        print(f"最终统计目录文件总数: {final_total} (文件: {final_file_count}, 目录: {final_dir_count})")

def check_disk_space():
    """检查磁盘空间"""
    target_dir = "/opt/gpfs/data/raw_data/Audioset/raw_data"
    if os.path.exists(target_dir):
        stat = os.statvfs(target_dir)
        free_space_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
        print(f"目标目录可用空间: {free_space_gb:.2f} GB")
        return free_space_gb
    return 0

def estimate_total_size():
    """估算解压后的总大小"""
    source_dir = "/opt/gpfs/data/raw_data/Audioset/audioset_zip/data"
    tar_pattern = os.path.join(source_dir, "unbal_train*.tar")
    tar_files = glob.glob(tar_pattern)
    
    total_size = 0
    for tar_file in tar_files:
        if os.path.exists(tar_file):
            total_size += os.path.getsize(tar_file)
    
    total_size_gb = total_size / (1024**3)
    print(f"所有tar文件总大小: {total_size_gb:.2f} GB")
    print(f"解压后预计需要空间: {total_size_gb * 2:.2f} GB (估算)")
    return total_size_gb

if __name__ == "__main__":
    print("AudioSet 不平衡训练集解压脚本")
    print("=" * 60)
    
    # 检查磁盘空间
    # free_space = check_disk_space()
    
    # # 估算解压后大小
    # estimated_size = estimate_total_size()
    
    # if free_space < estimated_size * 2:
    #     print(f"⚠️  警告: 可用空间 ({free_space:.2f} GB) 可能不足以解压所有文件")
    #     response = input("是否继续? (y/N): ")
    #     if response.lower() != 'y':
    #         print("取消解压")
    #         exit()
    
    print("\n开始解压...")
    extract_tar_files()