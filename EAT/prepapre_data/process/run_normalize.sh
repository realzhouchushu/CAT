# opt split: bal_train  eval  unbal_train; all to momo by librosa
split=unbal_train
python normalize.py \
    --source_dir /opt/gpfs/data/raw_data/Audioset/raw_data/audio/${split} \
    --target_dir  /opt/gpfs/data/raw_data/Audioset/wav_16k/${split} \
    --sample_rate 16000 \
    --log_file /opt/gpfs/home/chushu/codes/2506/fairseq/EAT/prepapre_data/process/log/normalize_${split}.log \
    --num_processes 64