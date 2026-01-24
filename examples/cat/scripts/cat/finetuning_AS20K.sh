#!/bin/bash
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/default_0_2025-09-20_15-33-21/checkpoint_11_100000.pt
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/default_0_2025-09-20_15-33-21/checkpoint_21_200000.pt
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/default_0_2025-09-20_15-33-21/checkpoint_31_300000.pt
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/default_0_2025-09-20_15-33-21/checkpoint_41_400000.pt
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/pre_4_AS2M/conv_0/checkpoint_6_50000.pt
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/default_0_2025-09-20_15-33-21/checkpoint_10_90000.pt
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/pre_4_AS2M/clap_0/checkpoint_10_90000.pt
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/conv_clap_0_2025-09-23_14-49-45/checkpoint_8_70000.pt
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/pre_4_AS2M/conv_clap_6/checkpoint_41_400000.pt
# model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/pre_4_AS2M/audio_mae_0/checkpoint_last.pt
model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/default_0_2025-09-20_15-33-21/checkpoint_last.pt

model_linear_layer=${1:-0}
model_add_bottleneck=${2:-false}
echo "model_linear_layer: ${model_linear_layer}"
# SAVE_DIR_ROOT=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS20K/default_11_100000_${model_linear_layer}
# SAVE_DIR_ROOT=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS20K/default_21_200000_${model_linear_layer}
# SAVE_DIR_ROOT=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS20K/default_31_300000_${model_linear_layer}
# SAVE_DIR_ROOT=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS20K/default_10_90000_${model_linear_layer}
# SAVE_DIR_ROOT=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS20K/clap_0_21_200000_${model_linear_layer}
# SAVE_DIR_ROOT=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS20K/conv_clap_0_8_70000_${model_linear_layer}
SAVE_DIR_ROOT=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS20K/default_0_41_400000_${model_linear_layer}_${model_add_bottleneck}
# 从 model_model_path 提取父目录名与文件名
parent_dir="$(basename -- "$(dirname -- "$model_model_path")")"
ckpt_name="$(basename -- "$model_model_path")"
# 组合保存目录并确保存在
checkpoint_save_dir="${SAVE_DIR_ROOT%/}/${parent_dir}"
mkdir -p -- "$checkpoint_save_dir"
checkpoint_restore_file="${checkpoint_save_dir%/}/${ckpt_name}"
echo "checkpoint_save_dir: ${checkpoint_save_dir}"
echo "checkpoint_restore_file: ${checkpoint_restore_file}"

device=1

CUDA_VISIBLE_DEVICES=${device} python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning  \
    common.user_dir=EAT \
    checkpoint.save_dir=${checkpoint_save_dir} \
    checkpoint.restore_file=${checkpoint_restore_file} \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=48 \
    dataset.num_workers=24 \
    dataset.data_buffer_size=48 \
    task.data=/inspire/hdd/global_user/zhouchushu-253108120180/data/audioset/setting/SFT_AS20k \
    task.target_length=1024 \
    task.roll_aug=true \
    task.load_clap_emb=false \
    +task.load_source_file=true \
    +task.load_mel_file=false \
    +model.linear_classifier=false \
    +model.linear_layer=${model_linear_layer} \
    +model.add_bottleneck=${model_add_bottleneck} \
    optimization.max_update=40000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=4000 \
    model.model_path=${model_model_path} \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN # CLS_TOKEN num_workers: 6 data_buffer_size: 10