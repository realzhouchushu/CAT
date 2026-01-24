#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7


run_dir=/inspire/hdd/global_user/zhouchushu-253108120180/codes/2506/EAT/SLAM-LLM
cd $run_dir
code_dir=examples/slam_aac

encoder_fairseq_dir=/inspire/hdd/global_user/zhouchushu-253108120180/codes/2506/EAT/EAT            # path to the fairseq directory of the encoder model

encoder_name="ast"
# encoder_name="audiomae"

# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/sft_4_AS2M_w_clap_CLS/clap_0_2025-08-27_09-23-59/checkpoint_best.pt
audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS2M/default_lw1_llayer0_layer12_llayer0/default_0_2025-09-20_15-33-21/checkpoint_best.pt
# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/official/encoder/EAT-base_epoch30_ft.pt
# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS2M/conv_422_lw1_layer11_llayer0/conv_clap_0_2025-09-23_14-49-45/checkpoint_best.pt
# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS2M/conv_clap1_41_400000_lw1_llayer0_layer12_llayer0/conv_clap_1/checkpoint_best.pt
audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/others/audio-mae/finetuned.pth
audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/others/ast/audioset_10_10_0.4593.pth

llm_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/lmsys/vicuna-7b-v1.5

seed=10086
btz=4
lr=8e-6
encoder_projector_ds_rate=5

train_jsonl_path=/inspire/hdd/global_user/zhouchushu-253108120180/data/acc-datasets/clotho/development.jsonl
val_jsonl_path=/inspire/hdd/global_user/zhouchushu-253108120180/data/acc-datasets/clotho/validation.jsonl

# exp_name=slam-aac_Clotho_fine-tune-eat-clap-AS2M
exp_name=slam-aac_Clotho_fine-tune-ast-AS2M

output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/eat_clap/${exp_name}
output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/eat/${exp_name}
output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/conv_clap_0/${exp_name}
output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/conv_clap_1/${exp_name}
output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/our_eat_400k_scratch_all/${exp_name}
output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/ast/${exp_name}

# output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/offical_eat_scratch_all/${exp_name}
# output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/eat_official_encoder/


# ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/eat_clap/slam-aac_pre-train-eat-clap-AS2M/aac_epoch_2_step_2382/model.pt   # path to load the pre-trained model
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/eat/slam-aac_pre-train-eat-clap-AS2M/aac_epoch_2_step_2382/model.pt
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/official/pretrain/model.pt
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/eat_official_encoder/slam-aac_pre-train-eat-clap-AS2M/aac_epoch_2_step_2382/model.pt
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/conv_clap_0/slam-aac_pre-train-eat-clap-AS2M/aac_epoch_2_step_2382/model.pt
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/conv_clap_1/slam-aac_pre-train-eat-clap-AS2M/aac_epoch_2_step_2382/model.pt
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/our_eat_400k_scratch/slam-aac_pre-train-our-eat-AS2M/aac_epoch_2_step_2382/model.pt
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/offical_eat_scratch/slam-aac_pre-train-offcial-eat-AS2M/aac_epoch_2_step_2382/model.pt
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/offical_eat_scratch_all/slam-aac_pre-train-our-eat-AS2M/aac_epoch_3_step_30764/model.pt
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/our_eat_400k_scratch_all/slam-aac_pre-train-our-eat-AS2M/aac_epoch_3_step_30764/model.pt
ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/ast/slam-aac_pre-train-ast-AS2M/aac_epoch_3_step_30764/model.pt
# ckpt_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/audiomae/slam-aac_pre-train-audiomae-AS2M/aac_epoch_3_step_30764/model.pt
peft_ckpt=null
# peft_ckpt=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/official/pretrain
# ↑ This parameter is required for loading the old version of the SLAM-LLM model. Our released checkpoint uses the old version. In the new version, this parameter is no longer needed.

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=${encoder_name} \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
++model_config.encoder_path=$audio_encoder_path \
++model_config.encoder_dim=768 \
++model_config.encoder_projector=linear \
++model_config.encoder_fairseq_dir=$encoder_fairseq_dir \
++dataset_config.encoder_projector_ds_rate=${encoder_projector_ds_rate} \
++dataset_config.dataset=audio_dataset \
++dataset_config.train_data_path=$train_jsonl_path \
++dataset_config.val_data_path=$val_jsonl_path \
++dataset_config.input_type=mel \
++dataset_config.fbank_mean=-4.268 \
++dataset_config.fbank_std=4.569 \
++dataset_config.model_name=${encoder_name} \
++dataset_config.fixed_length=true \
++dataset_config.target_length=1024 \
++train_config.model_name=aac \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=$lr \
++train_config.validation_interval=500 \
++train_config.batch_size_training=$btz \
++train_config.val_batch_size=$btz \
++train_config.num_workers_dataloader=4 \
++train_config.use_fp16=true \
++train_config.output_dir=$output_dir \
++train_config.seed=${seed} \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=lora \
++train_config.specaug=true \
++log_config.log_file="${output_dir}/train.log" \
++log_config.wandb_dir=${output_dir} \
++log_config.wandb_entity_name=wxc12 \
++log_config.wandb_project_name=slam-llm \
++log_config.wandb_exp_name=$exp_name \
++log_config.use_wandb=false \
++metric=acc \
++ckpt_path=$ckpt_path \
++peft_ckpt=$peft_ckpt \
"

# note: to train the linear layer only, you could set '++train_config.use_peft=false' and 'train_config.freeze_llm=true'
# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python $code_dir/finetune_aac.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        $hydra_args
fi

# bash /data/wenxi.chen/SLAM-LLM/examples/slam_aac/scripts/finetune_clotho.sh