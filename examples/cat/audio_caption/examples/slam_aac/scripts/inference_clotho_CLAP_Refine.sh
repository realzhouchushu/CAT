#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

run_dir=~/codes/2506/EAT/SLAM-LLM
cd $run_dir
code_dir=examples/slam_aac

encoder_fairseq_dir=~/codes/2506/EAT/EAT            # path to the fairseq directory of the encoder model

encoder_name="ast"
audio_encoder_path=~/hubs/models/others/ast/audioset_10_10_0.4593.pth

llm_path=~/hubs/models/huggingface/lmsys/vicuna-7b-v1.5
clap_dir=~/hubs/models/others/slam-aac/clap

encoder_projector_ds_rate=5

output_dir=~/exp/aac/ast/slam-aac_Clotho_fine-tune-ast-AS2M/aac_epoch_2_step_4702

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
