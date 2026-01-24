#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

run_dir=/inspire/hdd/global_user/zhouchushu-253108120180/codes/2506/EAT/SLAM-LLM
cd $run_dir
code_dir=examples/slam_aac

encoder_fairseq_dir=/inspire/hdd/global_user/zhouchushu-253108120180/codes/2506/EAT/EAT            # path to the fairseq directory of the encoder model

encoder_name="ast"
encoder_name="audiomae"

# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/zhouchushu/tmp_model_store/sft_4_AS2M_w_clap_CLS/clap_0_2025-08-27_09-23-59/checkpoint_best.pt
# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS2M/default_lw1_llayer0_layer12_llayer0/default_0_2025-09-20_15-33-21/checkpoint_best.pt
# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/official/encoder/EAT-base_epoch30_ft.pt
# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS2M/conv_422_lw1_layer11_llayer0/conv_clap_0_2025-09-23_14-49-45/checkpoint_best.pt
# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS2M/conv_clap1_41_400000_lw1_llayer0_layer12_llayer0/conv_clap_1/checkpoint_best.pt
audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/official/encoder/EAT-base_epoch30_ft.pt
audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/others/audio-mae/finetuned.pth
# audio_encoder_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/others/ast/audioset_10_10_0.4593.pth

llm_path=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/huggingface/lmsys/vicuna-7b-v1.5
clap_dir=/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/others/slam-aac/clap

encoder_projector_ds_rate=5

inference_data_path=/inspire/hdd/global_user/zhouchushu-253108120180/data/acc-datasets/clotho/evaluation_single.jsonl
# output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/eat_clap/slam-aac_Clotho_fine-tune-eat-clap-AS2M/aac_epoch_2_step_702
# output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/eat/slam-aac_Clotho_fine-tune-eat-clap-AS2M/aac_epoch_2_step_702
# output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/official/ckpt
# output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/eat_offical_pretrain/aac_epoch_1_step_4500
# output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/conv_clap_0/slam-aac_Clotho_fine-tune-eat-clap-AS2M/aac_epoch_2_step_202
# output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/conv_clap_1/slam-aac_Clotho_fine-tune-eat-clap-AS2M/aac_epoch_2_step_202
output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/audiomae/slam-aac_Clotho_fine-tune-audiomae-AS2M/aac_epoch_2_step_4702
# output_dir=/inspire/hdd/global_user/zhouchushu-253108120180/exp/aac/ast/slam-aac_Clotho_fine-tune-ast-AS2M/aac_epoch_2_step_4702

# define the beam size range
beam_range=(2 3 4 5 6 7 8)
# beam_range=()

for num_beams in "${beam_range[@]}"; do
    decode_log=$output_dir/decode_beam${num_beams}

    if [ -f "${decode_log}_pred" ]; then
        echo "Decode log ${decode_log}_pred already exists, skipping this beam size..."
        continue
    fi

    echo "Running inference with num_beams=$num_beams"

    python $code_dir/inference_aac_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$output_dir \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=${encoder_name} \
        ++model_config.encoder_path=$audio_encoder_path \
        ++model_config.encoder_dim=768 \
        ++model_config.encoder_projector=linear \
        ++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
        ++model_config.normalize=true \
        ++model_config.encoder_fairseq_dir=$encoder_fairseq_dir \
        ++dataset_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
        ++dataset_config.dataset=audio_dataset \
        ++dataset_config.val_data_path=$inference_data_path \
        ++dataset_config.fbank_mean=-4.268 \
        ++dataset_config.fbank_std=4.569 \
        ++dataset_config.model_name=${encoder_name} \
        ++dataset_config.inference_mode=true \
        ++dataset_config.normalize=true \
        ++dataset_config.input_type=mel \
        ++dataset_config.fixed_length=true \
        ++dataset_config.target_length=1024 \
        ++train_config.model_name=aac \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=4 \
        ++train_config.num_workers_dataloader=0 \
        ++train_config.output_dir=$output_dir \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=false \
        ++train_config.use_peft=true \
        ++ckpt_path=$output_dir/model.pt \
        ++peft_ckpt=null \
        ++decode_log=$decode_log \
        ++model_config.num_beams=$num_beams
done

# note: to inference model trained the linear layer only, you could set '++train_config.use_peft=false' and 'train_config.freeze_llm=true'

echo "Running CLAP-Refine"

# -m debugpy --listen 6666 --wait-for-client
python ${code_dir}/utils/clap_refine.py \
    --start_beam 2 --end_beam 8 \
    --clap_ckpt $clap_dir/best_model.pt \
    --config $clap_dir/clap_config.yaml \
    --test_jsonl $inference_data_path \
    --exp_explorer $output_dir

# bash /data/wenxi.chen/SLAM-LLM/examples/slam_aac/scripts/inference_clotho_CLAP_Refine.sh