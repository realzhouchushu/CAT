import os
import subprocess
import sys
from pathlib import Path

# --- 配置 ---
# 请根据您的环境设置此绝对路径
ROOT_DIR = Path("~/raw_datas/aac-datasets/WavCaps")
ZIP_DIR = ROOT_DIR / "Zip_files"
JSON_DIR = ROOT_DIR / "json_files"
AUDIO_DIR = ROOT_DIR / "Audio"

# 预期文件数量 (用于检查)
EXPECTED_COUNTS = {
    # Zip files
    "Zip_files/AudioSet_SL": 8,
    "Zip_files/BBC_Sound_Effects": 26,
    "Zip_files/FreeSound": 123,
    "Zip_files/SoundBible": 1,
    # JSON files
    "json_files/AudioSet_SL": 1,
    "json_files/BBC_Sound_Effects": 1,
    "json_files/FreeSound": 2, # fsd_final_2s.json 和 fsd_final.json
    "json_files/SoundBible": 1,
    "json_files/blacklist": 3,
}

# 数据源列表
SOURCES = ["AudioSet_SL", "BBC_Sound_Effects", "FreeSound", "SoundBible"]

def check_file_counts():
    """
    检查 Zip_files 和 json_files 目录下的文件数量是否符合预期。
    """
    print("\n--- 步骤 1/3: 检查文件数量 ---")
    all_checks_passed = True

    # 检查 Zip_files
    for source in SOURCES:
        path = ZIP_DIR / source
        # 寻找所有以 .z01 或 .zip 结尾的文件
        actual_count = len(list(path.glob(f"{source}.z*")))
        expected_key = f"Zip_files/{source}"
        expected_count = EXPECTED_COUNTS[expected_key]

        if actual_count == expected_count:
            print(f"✅ {expected_key} 数量检查通过: 找到 {actual_count} 个文件。")
        else:
            print(f"❌ {expected_key} 数量不匹配: 预期 {expected_count}, 实际找到 {actual_count}。")
            all_checks_passed = False

    # 检查 json_files
    for expected_key, expected_count in EXPECTED_COUNTS.items():
        if "json_files" in expected_key:
            sub_path = Path(expected_key.replace("json_files/", ""))
            path = JSON_DIR / sub_path
            
            # 如果是 blacklist 目录，统计 .json 文件
            if sub_path.name == "blacklist":
                actual_count = len(list(path.glob("*.json")))
            else:
                # 统计子目录下的 .json 文件
                actual_count = len(list(path.glob("*.json")))

            if actual_count == expected_count:
                print(f"✅ {expected_key} 数量检查通过: 找到 {actual_count} 个文件。")
            else:
                print(f"❌ {expected_key} 数量不匹配: 预期 {expected_count}, 实际找到 {actual_count}。")
                all_checks_passed = False

    if not all_checks_passed:
        print("\n[警告] 部分文件数量不符合预期。请检查后再继续解压。")
    return all_checks_passed

def find_first_zip_part(source_path: Path):
    """查找多卷压缩文件的第一个文件 (.z01 或 .zip)。"""
    # 查找 .z01 作为第一个卷
    first_part = source_path / f"{source_path.name}.z01"
    if first_part.exists():
        return first_part
    
    # 如果没有 .z01，则查找 .zip (可能是单文件或第一个卷)
    single_zip = source_path / f"{source_path.name}.zip"
    if single_zip.exists():
        return single_zip
        
    return None

def extract_zips():
    """
    将 Zip_files 下的压缩文件解压到对应的 Audio 目录下。
    使用 7z x 命令处理多卷压缩文件。
    """
    print("\n--- 步骤 2/3: 执行解压操作 ---")
    
    for source in SOURCES:
        zip_source_path = ZIP_DIR / source
        audio_target_path = AUDIO_DIR / source
        
        # 1. 创建目标目录 (如果不存在)
        audio_target_path.mkdir(parents=True, exist_ok=True)
        print(f"创建或确认目标目录: {audio_target_path}")

        # 2. 查找第一个压缩文件卷
        zip_file_to_extract = find_first_zip_part(zip_source_path)
        
        if not zip_file_to_extract:
            print(f"[跳过] 找不到 {source} 的主要压缩文件 (.z01 或 .zip)。")
            continue
            
        print(f"开始解压 {zip_file_to_extract}...")
        
        # 3. 执行 7z 解压命令
        # x: 提取文件，包括完整路径
        # -o{target_path}: 设置输出目录
        # -aoa: 覆盖所有现有文件
        # -y: 默认回答 'Yes'
        command = [
            "7z", "x", 
            str(zip_file_to_extract), 
            f"-o{audio_target_path}", 
            "-aoa", "-y"
        ]

        try:
            # 执行命令并捕获输出
            result = subprocess.run(
                command, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            print(f"✅ {source} 解压成功。7z 输出片段：\n{result.stdout[:200]}...")
        except subprocess.CalledProcessError as e:
            print(f"❌ {source} 解压失败。错误码: {e.returncode}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print("❌ 错误: 找不到 '7z' 命令。请确保 7z 已正确安装并添加到 PATH 中。")
            sys.exit(1)

def count_audio_files():
    """
    统计 Audio 目录下各子目录中的 .flac 文件数量。
    """
    print("\n--- 步骤 3/3: 统计音频文件数量 ---")
    
    total_audio_count = 0
    
    for source in SOURCES:
        audio_path = AUDIO_DIR / source
        
        if not audio_path.exists():
            print(f"[警告] 目录 {audio_path} 不存在，跳过统计。")
            continue

        # 递归统计所有 .flac 文件
        flac_count = len(list(audio_path.rglob("*.flac")))
        print(f"🎧 {source} 目录下找到 {flac_count} 个 .flac 音频文件。")
        total_audio_count += flac_count

    print(f"\n✨ 所有 Audio 目录下 .flac 文件总数为: {total_audio_count}")


if __name__ == "__main__":
    print(f"WavCaps 数据集处理脚本启动。根目录: {ROOT_DIR}")
    
    # 1. 检查文件数量
    check_file_counts()

    # 2. 执行解压操作
    extract_zips()

    # 3. 统计音频文件数量
    count_audio_files()

    print("\n--- 脚本执行完毕 ---")
