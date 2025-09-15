# config options
train_mode=conv_clap
config_option=0

# shared config
SAVE_DIR_ROOT=/opt/gpfs/home/chushu/exp/eat/pre_4_AS2M
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

# default setting
model_clone_batch=4
dataset_batch_size=48
model_clap_loss=1.0
average_top_k_layers=12
model_add_conv=false
model_depth=12
checkpoint_keep_interval_updates=1 
checkpoint_save_interval_updates=10000

if [[ $train_mode == "default" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=48
    model_clap_loss=0
elif [[ $train_mode == "clap" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=48
    model_clap_loss=1.0
    average_top_k_layers=12
    model_add_conv=false
elif [[ $train_mode == "clap" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=48
    model_clap_loss=1.0
    average_top_k_layers=1
elif [[ $train_mode == "ast" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_in
    task_load_clap_emb=true
    model_proj_type=4
    model_clone_batch=4
    model_clap_loss=1.0
    dataset_batch_size=48
elif [[ $train_mode == "ast" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_in
    task_load_clap_emb=true
    model_proj_type=4
    model_clone_batch=4
    model_clap_loss=0.001
    dataset_batch_size=48
elif [[ $train_mode == "ast" && ${config_option} -eq 2 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_in
    task_load_clap_emb=true
    model_proj_type=4
    model_clone_batch=4
    model_clap_loss=0.01
    dataset_batch_size=48
elif [[ $train_mode == "ast" && ${config_option} -eq 3 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_out
    task_load_clap_emb=true
    model_proj_type=6
    model_clone_batch=4
    dataset_batch_size=48
elif [[ $train_mode == "conv_clap" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=16 # original 48 oom on 4090 24G 
    model_clap_loss=1.0
    average_top_k_layers=11 # modify with model depth
    model_add_conv=true
    model_depth=11 # 
    checkpoint_keep_interval_updates=-1 # default 1 
    checkpoint_save_interval_updates=10000
fi

python fairseq_cli/hydra_train.py -m \
    --config-dir ./EAT/config \
    --config-name pretraining_AS2M \
    common.user_dir=./EAT \
    checkpoint.save_dir=${checkpoint_save_dir} \
    checkpoint.restore_file=${checkpoint_restore_file} \
    distributed_training.distributed_world_size=${1:-4} \
    dataset.num_workers=24 \
    dataset.data_buffer_size=48 \
    dataset.batch_size=${dataset_batch_size} \
    task.data=${task_data} \
    task.h5_format=False \
    task.load_clap_emb=${task_load_clap_emb} \
    model.proj_type=${model_proj_type} \
    model.clone_batch=${model_clone_batch} \
    model.clap_loss=${model_clap_loss} \
    model.average_top_k_layers=${average_top_k_layers} \
    +model.add_conv=${model_add_conv} \
    model.depth=${model_depth} \
    checkpoint.keep_interval_updates=${checkpoint_keep_interval_updates} \
    checkpoint.save_interval_updates=${checkpoint_save_interval_updates}