#!/bin/bash

# 定义不同的学习率
learning_rates=(0.005 0.003 0.001 0.0005 0.0003 0.0001 0.00005 0.00003 0.00001)

# 创建日志目录
mkdir -p /opt/gpfs/home/chushu/codes/2506/EAT/evar/src/le/logs

# 循环运行不同学习率的实验
for lr in "${learning_rates[@]}"; do
    echo "=========================================="
    echo "Running experiment with learning rate: $lr"
    echo "=========================================="
    
    # 修改all_eat.sh中的学习率
    sed -i "s/lr=.*/lr=$lr/" /opt/gpfs/home/chushu/codes/2506/EAT/evar/src/le/all_eat.sh
    
    # 运行实验并重定向输出
    bash /opt/gpfs/home/chushu/codes/2506/EAT/evar/src/le/all_eat.sh > /opt/gpfs/home/chushu/codes/2506/EAT/evar/src/le/logs/log_${lr}.log 2>&1
    
    # 检查运行状态
    if [ $? -eq 0 ]; then
        echo "Experiment with lr=$lr completed successfully"
    else
        echo "Experiment with lr=$lr failed"
    fi
    
    echo "Output saved to: /opt/gpfs/home/chushu/codes/2506/EAT/evar/src/le/logs/log_${lr}.log"
    echo ""
done

echo "All experiments completed!"
echo "Log files are saved in: /opt/gpfs/home/chushu/codes/2506/EAT/evar/src/le/logs/"
