#!/usr/bin/env python3
"""
AudioSet 不平衡训练集解压脚本
解压分卷压缩文件到指定目录
"""

import os
import subprocess
import sys

SRC_DIR = "/opt/gpfs/data/raw_data/audioset/audioset_zip/unbalanced"
TARGET_DIR = "/opt/gpfs/data/raw_data/audioset/raw_data/unbal_train"

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"正在执行: {description}")
    print(f"命令: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} 失败")
        return False

def extract_audioset_files():
    """解压AudioSet文件"""
    
    # 目标目录
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"源目录: {SRC_DIR}")
    print(f"目标目录: {TARGET_DIR}")
    
    # 切换到目标目录
    # os.chdir(target_dir) # 移除此行，因为不再需要切换到目标目录
    # print(f"已切换到目录: {os.getcwd()}") # 移除此行
    
    # 定义需要解压的文件组（按顺序）
    file_groups = []
    
    # 生成文件组（part00-part40）
    for part_num in range(41):  # 0-40（如果没有40会自动跳过）
        # if part_num < 14 or part_num >= 20:
        #     continue
        part_str = f"{part_num:02d}"  # 格式化为两位数
        
        # 每个部分有三个文件：.z01, .z02, .zip
        z01_file = f"unbalanced_train_segments_part{part_str}_partial.z01"
        z02_file = f"unbalanced_train_segments_part{part_str}_partial.z02"
        zip_file = f"unbalanced_train_segments_part{part_str}_partial.zip"
        
        file_groups.append({
            'part': part_str,
            'z01': z01_file,
            'z02': z02_file,
            'zip': zip_file
        })
    
    print(f"总共需要解压 {len(file_groups)} 个部分")
    
    # 解压每个部分
    for i, group in enumerate(file_groups, 1):
        print("\n" + "="*60)
        print(f"正在处理第 {i}/{len(file_groups)} 部分: part{group['part']}")
        print("="*60)
        
        # 检查文件是否存在
        missing_files = []
        for ext, filename in [('z01', group['z01']), ('z02', group['z02']), ('zip', group['zip'])]:
            if not os.path.exists(os.path.join(SRC_DIR, filename)):
                missing_files.append(filename)
        
        if missing_files:
            print(f"⚠️  缺少文件: {[os.path.basename(x) for x in missing_files]}")
            print(f"跳过部分 part{group['part']}")
            continue
        
        # 使用7z解压（7z可以处理分卷压缩）
        # 只需要指定第一个文件(.z01)，7z会自动找到其他分卷
        extract_cmd = f'7z x "{os.path.join(SRC_DIR, group["zip"])}" -o"{TARGET_DIR}" -y'
        
        if run_command(extract_cmd, f"解压 part{group['part']}"):
            print(f"✓ 部分 part{group['part']} 解压完成")
        else:
            ans = input("是否继续解压下一个部分？(y/n): ").strip().lower()
            if ans != "y":
                break
    
    print("\n" + "="*60)
    print("解压完成！")
    # 统计目标目录文件总数（递归）
    total_files = sum(len(files) for _, _, files in os.walk(TARGET_DIR))
    print(f"解压后的文件数量(含子目录): {total_files}")

def main():
    """主函数"""
    print("AudioSet 不平衡训练集解压脚本")
    print("="*60)
    
    # 检查7z是否安装 - 修复7z版本检查
    try:
        # 7z的正确版本检查方式
        r = subprocess.run(["7z"], capture_output=True, text=True)
        if "7-Zip" not in (r.stdout + r.stderr):
            print("⚠️  未检测到 7z 输出，检查安装")
            sys.exit(1)
        print("✓ 7z 已安装")
    except FileNotFoundError:
        print("✗ 7z 未安装，请先安装 p7zip-full")
        sys.exit(1)
    
    # 检查当前目录是否包含需要解压的文件
    if not os.path.isdir(SRC_DIR):
        print(f"源目录不存在: {SRC_DIR}")
        sys.exit(1)
    
    # os.chdir(current_dir) # 移除此行，因为不再需要切换到包含压缩文件的目录
    # print(f"已切换到目录: {os.getcwd()}") # 移除此行
    
    # 查找第一个文件来确认位置
    # first_file = "unbalanced_train_segments_part00_partial.z01" # 移除此行
    # if not os.path.exists(first_file): # 移除此行
    #     print(f"⚠️  在当前目录未找到文件: {first_file}") # 移除此行
    #     print("请确保脚本在包含压缩文件的目录中运行") # 移除此行
    #     response = input("是否继续？(y/n): ") # 移除此行
    #     if response.lower() != 'y': # 移除此行
    #         sys.exit(1) # 移除此行
    
    # 开始解压
    extract_audioset_files()

if __name__ == "__main__":
    # cd /opt/gpfs/data/raw_data/audioset/audioset_zip/unbalanced and run this scripts
    main()