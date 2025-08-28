#!/bin/bash

# 保持脚本运行，防止容器退出
cd /opt/gpfs/home/chushu/codes/2506/EAT
pip install nvitop
pip install --editable ./

echo "Starting long-running script to keep container alive..."

while true; do
    echo "$(date): Container is still running..."
    sleep 60  # 每60秒输出一次状态
done