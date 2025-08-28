# opt split: bal_train  eval  unbal_train; all to momo by librosa
split=unbal_train
python normalize.py \
    --source_dir /opt/gpfs/data/raw_data/audioset/raw_data/${split} \
    --target_dir  /opt/gpfs/data/raw_data/audioset/wav_16k/${split} \
    --sample_rate 16000 \
    --log_file /opt/gpfs/home/chushu/codes/2506/EAT/EAT/prepapre_data/process/log/normalize_${split}1.1og \
    --num_processes 64