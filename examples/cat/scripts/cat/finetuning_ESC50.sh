model_model_path=~/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/default_0_2025-09-20_15-33-21/checkpoint_last.pt

data_fold=5

SAVE_DIR_ROOT=~/exp/cat/sft_4_esc50/default_0_41_400000
# 从 model_model_path 提取父目录名与文件名
parent_dir="$(basename -- "$(dirname -- "$model_model_path")")"
ckpt_name="$(basename -- "$model_model_path")"
# 组合保存目录并确保存在
checkpoint_save_dir="${SAVE_DIR_ROOT%/}/${parent_dir}/fold${data_fold}"
mkdir -p -- "$checkpoint_save_dir"
checkpoint_restore_file="${checkpoint_save_dir%/}/${ckpt_name}"
echo "checkpoint_save_dir: ${checkpoint_save_dir}"
echo "checkpoint_restore_file: ${checkpoint_restore_file}"


device=0

CUDA_VISIBLE_DEVICES=${device} python fairseq_cli/hydra_train.py -m \
    --config-dir examples/cat/config \
    --config-name finetuning  \
    common.user_dir=examples/cat \
    common.log_interval=100 \
    checkpoint.save_dir=${checkpoint_save_dir} \
    checkpoint.restore_file=${checkpoint_restore_file} \
    dataset.batch_size=48 \
    criterion.log_keys=['correct'] \
    task.data=~/codes/2506/CAT/examples/cat/data_manifest/SFT_ESC_50/test0${data_fold} \
    task.esc50_eval=True \
    task.target_length=512 \
    task.roll_aug=true \
    optimization.max_update=4000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=400 \
    model.model_path=${model_model_path} \
    model.num_classes=50 \
    model.esc50_eval=true \
    model.mixup=0.0 \
    model.target_length=512 \
    model.mask_ratio=0.4 \
    model.label_smoothing=0.1 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \