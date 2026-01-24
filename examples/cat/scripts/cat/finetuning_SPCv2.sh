model_model_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/default_0_2025-09-20_15-33-21/checkpoint_last.pt

SAVE_DIR_ROOT=/inspire/hdd/global_user/zhouchushu-253108120180/exp/cat/sft_4_SPCv2/default_0_41_400000

# 从 model_model_path 提取父目录名与文件名
parent_dir="$(basename -- "$(dirname -- "$model_model_path")")"
ckpt_name="$(basename -- "$model_model_path")"
# 组合保存目录并确保存在
checkpoint_save_dir="${SAVE_DIR_ROOT%/}/${parent_dir}"
mkdir -p -- "$checkpoint_save_dir"
checkpoint_restore_file="${checkpoint_save_dir%/}/${ckpt_name}"
echo "checkpoint_save_dir: ${checkpoint_save_dir}"
echo "checkpoint_restore_file: ${checkpoint_restore_file}"

device=0

CUDA_VISIBLE_DEVICES=${device} python fairseq_cli/hydra_train.py -m \
    --config-dir examples/cat/config \
    --config-name finetuning  \
    common.user_dir=examples/cat \
    common.seed=42 \
    checkpoint.save_dir=${checkpoint_save_dir} \
    checkpoint.restore_file=${checkpoint_restore_file} \
    dataset.batch_size=256 \
    criterion.log_keys=['correct'] \
    task.data=/inspire/hdd/global_user/zhouchushu-253108120180/data/audioset/EAT_manifest/SPC_2 \
    task.spcv2_eval=True \
    task.target_length=128 \
    task.noise=true \
    task.roll_aug=true \
    optimization.lr=[0.0002] \
    optimizer.groups.default.lr_float=0.0002 \
    optimization.max_update=40000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=4000 \
    model.model_path=${model_model_path} \
    model.num_classes=35 \
    model.spcv2_eval=true \
    model.mixup=0.8 \
    model.target_length=128 \
    model.mask_ratio=0.2 \
    model.label_smoothing=0.1 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \