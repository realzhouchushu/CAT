# first modify following scrpits and download data
# cd downloads
# wget https://github.com/karoldvl/ESC-50/archive/master.zip
# unzip master.zip
# mv ESC-50-master esc50
# cd ..

# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/esc50/raw_esc50/audio /opt/gpfs/data/raw_data/esc50/16k/ 16000
# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/esc50/raw_esc50/audio /opt/gpfs/data/raw_data/esc50/22k/ 22000
# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/esc50/raw_esc50/audio /opt/gpfs/data/raw_data/esc50/32k/ 32000
# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/esc50/raw_esc50/audio /opt/gpfs/data/raw_data/esc50/44k/ 44100
# python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/esc50/raw_esc50/audio /opt/gpfs/data/raw_data/esc50/48k/ 48000

python ./evar/evar/utils/make_metadata.py esc50 /opt/gpfs/data/raw_data/esc50