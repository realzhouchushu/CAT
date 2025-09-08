# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/surge/raw_data /opt/gpfs/data/raw_data/surge/16k 16000 --suffix .ogg
# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/surge/raw_data /opt/gpfs/data/raw_data/surge/22k 22000 --suffix .ogg
# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/surge/raw_data /opt/gpfs/data/raw_data/surge/32k 32000 --suffix .ogg
# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/surge/raw_data /opt/gpfs/data/raw_data/surge/44k 44100 --suffix .ogg
# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/surge/raw_data /opt/gpfs/data/raw_data/surge/48k 48000 --suffix .ogg

python ./evar/evar/utils/make_metadata.py surge /opt/gpfs/data/raw_data/surge/raw_data
