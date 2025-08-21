# config options
train_mode=clap
config_option=0

# shared config
SAVE_DIR_ROOT=/opt/gpfs/home/chushu/exp/eat/pre_8_AS2M
checkpoint_save_dir=${SAVE_DIR_ROOT}/${train_mode}_${config_option}_$(date +"%Y-%m-%d_%H-%M-%S")
checkpoint_restore_file=${checkpoint_save_dir}/checkpoint_last.pt

# 脚本自身的绝对路径与文件名（解析符号链接）
script_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
script_name="$(basename -- "$script_path")"
# 创建目录并拷贝（保留权限与时间戳）
mkdir -p -- "$checkpoint_save_dir"
cp -p -- "$script_path" "$save_dir/$script_name"
echo "script_path: ${script_path}"
echo "checkpoint_save_dir: ${checkpoint_save_dir}"

if [[ $train_mode == "default" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/AudioSet/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    model_proj_type=null
elif [[ $train_mode == "clap" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/AudioSet/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=48
fi

python fairseq_cli/hydra_train.py -m \
    --config-dir ./EAT/config \
    --config-name pretraining_AS2M \
    common.user_dir=./EAT \
    checkpoint.save_dir=${checkpoint_save_dir} \
    checkpoint.restore_file=${checkpoint_restore_file} \
    distributed_training.distributed_world_size=1 \
    dataset.num_workers=24 \
    dataset.data_buffer_size=48 \
    dataset.batch_size=${dataset_batch_size} \
    task.data=${task_data} \
    task.h5_format=False \
    task.load_clap_emb=${task_load_clap_emb} \
    model.proj_type=${model_proj_type} \
    model.clone_batch=${model_clone_batch}