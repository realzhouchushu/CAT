#!/usr/bin/env bash
# config options
train_mode=test
config_option=0
# change world size

world_size=${1:-2}
# shared config
SAVE_DIR_ROOT=~/exp/cat/pre_4_AS2M
checkpoint_save_dir=${SAVE_DIR_ROOT}/${train_mode}_${config_option}
checkpoint_restore_file=${checkpoint_save_dir}/checkpoint_last.pt

script_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
script_name="$(basename -- "$script_path")"

mkdir -p -- "$checkpoint_save_dir"
cp -p -- "$script_path" "$checkpoint_save_dir/$script_name"
echo "script_path: ${script_path}"
echo "checkpoint_save_dir: ${checkpoint_save_dir}"

config_name=pretraining_AS2M
# default setting
model_clone_batch=4
dataset_batch_size=48
model_clap_loss=0
model_clap_loss_type="mse"  # option ce cosine l1
model_clap_loss_layer=0
average_top_k_layers=12
model_add_conv=false
model_depth=12
model_dispersive_loss=0
model_dispersive_loss_layer=0
checkpoint_keep_interval_updates=1 
checkpoint_save_interval_updates=10000
model_modalities_image_conv_option=0
model_modalities_image_patch_size=16
model_modalities_image_mask_prob=0.8
model_add_bottleneck=false
model_bottleneck_dim=768

optimization_max_update=400000

if [[ $train_mode == "test" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/codes/2506/CAT/examples/cat/data_manifest/PRETRAIN_AS2M/test
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=0
    average_top_k_layers=11 
    model_add_conv=true
    model_depth=11 
    checkpoint_keep_interval_updates=-1
    checkpoint_save_interval_updates=10000
elif [[ $train_mode == "default" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=0
    checkpoint_keep_interval_updates=-1
elif [[ $train_mode == "disp" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_dispersive_loss=1
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=1
    dataset_batch_size=384
    model_dispersive_loss=1
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 2 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=1
    dataset_batch_size=384
    model_dispersive_loss=10.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 3 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=1
    dataset_batch_size=384
    model_dispersive_loss=100.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 4 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=1
    dataset_batch_size=384
    model_dispersive_loss=10000.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 5 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=1
    dataset_batch_size=384
    model_dispersive_loss=1000.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 6 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_dispersive_loss=1000.0
    model_dispersive_loss_layer=10
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 7 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_dispersive_loss=1000.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 8 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_dispersive_loss=500.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 9 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_dispersive_loss=100.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 10 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_dispersive_loss=50.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 11 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_dispersive_loss=10.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "disp" && ${config_option} -eq 12 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_dispersive_loss=5000.0
    model_dispersive_loss_layer=0
    checkpoint_keep_interval_updates=1
elif [[ $train_mode == "clap" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    average_top_k_layers=12
    model_add_conv=false
    checkpoint_keep_interval_updates=-1
elif [[ $train_mode == "clap" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=48
    model_clap_loss=1.0
    average_top_k_layers=1
# loss type ablation
elif [[ $train_mode == "clap" && ${config_option} -eq 2 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=48
    model_clap_loss=1.0
    average_top_k_layers=12
    model_clap_loss_type="ce"
elif [[ $train_mode == "clap" && ${config_option} -eq 3 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=48
    model_clap_loss=1.0
    average_top_k_layers=12
    model_clap_loss_type="l1"
elif [[ $train_mode == "clap" && ${config_option} -eq 4 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    average_top_k_layers=12
    model_clap_loss_type="cosine"
# loss layer ablation
elif [[ $train_mode == "clap" && ${config_option} -eq 5 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    average_top_k_layers=12
    model_clap_loss_type="mse"
    model_clap_loss_layer=10
elif [[ $train_mode == "clap" && ${config_option} -eq 6 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    average_top_k_layers=12
    model_clap_loss_type="mse"
    model_clap_loss_layer=8
elif [[ $train_mode == "clap" && ${config_option} -eq 7 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    average_top_k_layers=12
    model_clap_loss_type="mse"
    model_clap_loss_layer=6
elif [[ $train_mode == "clap" && ${config_option} -eq 8 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    model_clap_loss=5.0
    dataset_batch_size=96
    average_top_k_layers=12
    model_clap_loss_type="mse"
    checkpoint_keep_interval_updates=-1
elif [[ $train_mode == "clap" && ${config_option} -eq 9 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    model_clap_loss=0.1
    dataset_batch_size=96
    average_top_k_layers=12
    model_clap_loss_type="mse"
    checkpoint_keep_interval_updates=-1
elif [[ $train_mode == "clap" && ${config_option} -eq 10 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    model_add_bottleneck=false
    average_top_k_layers=12
    model_add_conv=false
    model_depth=12 # 
    config_name=pretraining_AS2M_large
elif [[ $train_mode == "dasheng" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_dasheng
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    model_add_bottleneck=false
    average_top_k_layers=12
    model_add_conv=false
    model_depth=12 # 
elif [[ $train_mode == "beats" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_beats
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    model_add_bottleneck=false
    average_top_k_layers=12
    model_add_conv=false
    model_depth=12 # 
elif [[ $train_mode == "beats" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_beats_pretrain
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    model_add_bottleneck=false
    average_top_k_layers=12
    model_add_conv=false
    model_depth=12
elif [[ $train_mode == "audio_mae" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AUDIO_MAE
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=1.0
    model_add_bottleneck=false
    average_top_k_layers=12 
    model_add_conv=false
    model_depth=12 # 
elif [[ $train_mode == "audio_mae" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AUDIO_MAE_pretrain
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=1.0
    model_add_bottleneck=false
    average_top_k_layers=12 
    model_add_conv=false
    model_depth=12 # 
elif [[ $train_mode == "audio_mae" && ${config_option} -eq 2 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AUDIO_MAE_pretrain
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=0.001
    model_add_bottleneck=false
    average_top_k_layers=12 
    model_add_conv=false
    model_depth=12 # 
elif [[ $train_mode == "audio_mae" && ${config_option} -eq 3 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AUDIO_MAE_pretrain
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=0.001
    model_clap_loss_layer=8
    model_add_bottleneck=false
    average_top_k_layers=12 
    model_add_conv=false
    model_depth=12 # 
elif [[ $train_mode == "audio_mae" && ${config_option} -eq 4 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AUDIO_MAE_pretrain
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=0.001
    model_clap_loss_layer=6
    model_add_bottleneck=false
    average_top_k_layers=12 
    model_add_conv=false
    model_depth=12 # 
elif [[ $train_mode == "audio_mae" && ${config_option} -eq 5 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AUDIO_MAE_pretrain
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=0.001
    model_clap_loss_layer=4
    model_add_bottleneck=false
    average_top_k_layers=12 # modify with model depth
    model_add_conv=false
    model_depth=12 # 
elif [[ $train_mode == "audio_mae" && ${config_option} -eq 6 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AUDIO_MAE_pretrain
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=0.001
    model_clap_loss_layer=10
    model_add_bottleneck=false
    average_top_k_layers=12 
    model_add_conv=false
    model_depth=12 # 
elif [[ $train_mode == "conv_audio_mae" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AUDIO_MAE
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=1.0
    model_add_bottleneck=false
    average_top_k_layers=11 
    model_add_conv=true
    model_modalities_image_conv_option=0
    model_depth=11 # 
elif [[ $train_mode == "conv_audio_mae" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AUDIO_MAE_pretrain
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=0.001
    model_add_bottleneck=false
    average_top_k_layers=11 
    model_add_conv=true
    model_modalities_image_conv_option=0
    model_depth=11 
elif [[ $train_mode == "ast" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_in
    task_load_clap_emb=true
    model_proj_type=4
    model_clone_batch=4
    model_clap_loss=1.0
    dataset_batch_size=48
elif [[ $train_mode == "ast" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_in
    task_load_clap_emb=true
    model_proj_type=4
    model_clone_batch=4
    model_clap_loss=0.001
    dataset_batch_size=48
elif [[ $train_mode == "ast" && ${config_option} -eq 2 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_in
    task_load_clap_emb=true
    model_proj_type=4
    model_clone_batch=4
    model_clap_loss=0.01
    dataset_batch_size=48
elif [[ $train_mode == "ast" && ${config_option} -eq 3 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_out
    task_load_clap_emb=true
    model_proj_type=6
    model_clone_batch=4
    dataset_batch_size=48
elif [[ $train_mode == "ast" && ${config_option} -eq 4 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_AST_AS2M/mlp_head_in
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=4
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=0.001
    model_add_bottleneck=false
    average_top_k_layers=12 
    model_add_conv=false
    model_depth=12 
elif [[ $train_mode == "conv" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=0
    average_top_k_layers=11 
    model_add_conv=true
    model_depth=11 
    checkpoint_keep_interval_updates=-1 
    checkpoint_save_interval_updates=10000
elif [[ $train_mode == "conv" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=0
    average_top_k_layers=12
    model_add_conv=true
    model_modalities_image_conv_option=1
    model_depth=12 
    checkpoint_keep_interval_updates=1 
    checkpoint_save_interval_updates=10000
elif [[ $train_mode == "conv" && ${config_option} -eq 2 ]]; then
    echo "Config ${train_mode} ${config_option}" # H100 80G
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=24
    model_clap_loss=0
    average_top_k_layers=12 
    model_add_conv=true
    model_modalities_image_conv_option=2
    model_modalities_image_patch_size=8
    model_depth=12 
    checkpoint_keep_interval_updates=1 
    checkpoint_save_interval_updates=10000
elif [[ $train_mode == "conv" && ${config_option} -eq 3 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=0
    average_top_k_layers=12 
    model_add_conv=true
    model_modalities_image_conv_option=3
    model_depth=12 
    checkpoint_keep_interval_updates=1 
    checkpoint_save_interval_updates=10000
elif [[ $train_mode == "conv" && ${config_option} -eq 4 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96 
    model_clap_loss=0
    average_top_k_layers=12 
    model_add_conv=true
    model_modalities_image_conv_option=4
    model_depth=12 
    checkpoint_keep_interval_updates=1 
    checkpoint_save_interval_updates=10000
elif [[ $train_mode == "conv" && ${config_option} -eq 5 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=24
    model_clap_loss=0
    average_top_k_layers=12
    model_add_conv=true
    model_modalities_image_conv_option=5
    model_depth=12
    checkpoint_keep_interval_updates=1
    checkpoint_save_interval_updates=10000
elif [[ $train_mode == "conv" && ${config_option} -eq 6 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=false
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=null
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=0
    average_top_k_layers=12
    model_add_conv=true
    model_modalities_image_conv_option=6
    model_modalities_image_patch_size=32
    model_depth=12
    checkpoint_keep_interval_updates=1
    checkpoint_save_interval_updates=10000
elif [[ $train_mode == "conv_clap" && ${config_option} -eq 0 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    average_top_k_layers=11
    model_add_conv=true
    model_depth=11
    checkpoint_keep_interval_updates=-1
    checkpoint_save_interval_updates=10000
    optimization_max_update=400000
elif [[ $train_mode == "conv_clap" && ${config_option} -eq 1 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    average_top_k_layers=11
    model_add_conv=true
    model_modalities_image_mask_prob=0.85
    model_depth=11
    checkpoint_keep_interval_updates=-1 
    checkpoint_save_interval_updates=10000
    optimization_max_update=800000
elif [[ $train_mode == "conv_clap" && ${config_option} -eq 2 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=48
    model_clap_loss=1.0
    average_top_k_layers=11
    model_add_conv=true
    model_depth=11
    checkpoint_keep_interval_updates=-1
    checkpoint_save_interval_updates=10000
    optimization_max_update=800000
elif [[ $train_mode == "conv_clap" && ${config_option} -eq 3 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    model_add_bottleneck=true
    model_bottleneck_dim=128
    average_top_k_layers=11
    model_add_conv=true
    model_depth=11
    checkpoint_keep_interval_updates=-1
    checkpoint_save_interval_updates=100000
    optimization_max_update=400000
elif [[ $train_mode == "conv_clap" && ${config_option} -eq 4 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    model_add_bottleneck=true
    model_bottleneck_dim=64
    average_top_k_layers=11
    model_add_conv=true
    model_depth=11
    checkpoint_keep_interval_updates=-1
    checkpoint_save_interval_updates=100000
    optimization_max_update=400000
elif [[ $train_mode == "conv_clap" && ${config_option} -eq 5 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    model_add_bottleneck=true
    model_bottleneck_dim=32
    average_top_k_layers=11
    model_add_conv=true
    model_depth=11
    checkpoint_keep_interval_updates=-1 
    checkpoint_save_interval_updates=100000
    optimization_max_update=400000
elif [[ $train_mode == "conv_clap" && ${config_option} -eq 6 ]]; then
    echo "Config ${train_mode} ${config_option}"
    task_data=~/data/audioset/setting/PRETRAIN_AS2M_w_CLAP
    task_load_clap_emb=true
    task_load_source_file=true
    task_load_mel_file=false
    model_proj_type=2
    model_clone_batch=4
    dataset_batch_size=96
    model_clap_loss=1.0
    model_add_bottleneck=true
    model_bottleneck_dim=16
    average_top_k_layers=11 
    model_add_conv=true
    model_depth=11 
    checkpoint_keep_interval_updates=-1 
    checkpoint_save_interval_updates=100000
    optimization_max_update=400000
fi

python fairseq_cli/hydra_train.py -m \
    --config-dir examples/cat/config \
    --config-name ${config_name} \
    common.user_dir=examples/cat \
    checkpoint.save_dir=${checkpoint_save_dir} \
    checkpoint.restore_file=${checkpoint_restore_file} \
    distributed_training.distributed_world_size=${world_size} \
    dataset.num_workers=24 \
    dataset.data_buffer_size=48 \
    dataset.batch_size=${dataset_batch_size} \
    task.data=${task_data} \
    task.h5_format=False \
    task.load_clap_emb=${task_load_clap_emb} \
    +task.load_source_file=${task_load_source_file} \
    +task.load_mel_file=${task_load_mel_file} \
    model.proj_type=${model_proj_type} \
    model.clone_batch=${model_clone_batch} \
    model.clap_loss=${model_clap_loss} \
    model.average_top_k_layers=${average_top_k_layers} \
    +model.add_conv=${model_add_conv} \
    +model.clap_loss_type=${model_clap_loss_type} \
    +model.clap_loss_layer=${model_clap_loss_layer} \
    +model.dispersive_loss=${model_dispersive_loss} \
    +model.dispersive_loss_layer=${model_dispersive_loss_layer} \
    +model.add_bottleneck=${model_add_bottleneck} \
    +model.bottleneck_dim=${model_bottleneck_dim} \
    model.depth=${model_depth} \
    +model.modalities.image.conv_option=${model_modalities_image_conv_option} \
    +model.modalities.image.patch_size=${model_modalities_image_patch_size} \
    model.modalities.image.mask_prob=${model_modalities_image_mask_prob} \
    checkpoint.keep_interval_updates=${checkpoint_keep_interval_updates} \
    checkpoint.save_interval_updates=${checkpoint_save_interval_updates} \
    optimization.max_update=${optimization_max_update}