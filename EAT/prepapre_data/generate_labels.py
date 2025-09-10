import json
import numpy as np

# # eval
# json_file = '/data02/data/audio/audioset/audios/data/datafiles/audioset_eval_data.json'
# idx2label = {}
# with open(json_file, 'r') as f:
#     data = json.load(f)
#     for i in range(len(data['data'])):
#         idx = data['data'][i]['wav'].strip().split('/')[-1]
#         label = data['data'][i]['labels']
#         idx2label[idx] = label
# # "dict_keys(['wav', 'length', 'labels'])"
# # print(data['data'][0]['wav'])
# with open('/home/hanbing/audio_pretrain/fairseq/data/finetune_AS2M/eval.tsv', 'r') as f:
#     data = f.readlines()[1:]

# with open('/home/hanbing/audio_pretrain/fairseq/data/finetune_AS2M/eval.lbl', 'w') as f:
#     for i in range(len(data)):
#         idx = data[i].strip().split()[0]
#         label = idx2label[idx]
#         f.write(f"{idx}\t{label}\n")


# train
# json_file = '/data02/data/audio/audioset/audios/data/datafiles/audioset_unbal_train_data.json'
# json_file = '/data02/data/audio/audioset/audios/data/datafiles/audioset_eval_data.json'
# idx2label = {}
# with open('/opt/gpfs/home/chushu/data/audioset/setting/SFT_AS2M/train.lbl', 'r') as f:
#     data = f.readlines()
#     for i in range(len(data)):
#         idx = data[i].split()[0] + '.wav'
#         label = data[i].strip().split()[1]
#         idx2label[idx] = label
# with open(json_file, 'r') as f:
#     data = json.load(f)
#     for i in range(len(data['data'])):
#         idx = data['data'][i]['wav'].strip().split('/')[-1]
#         label = data['data'][i]['labels']
#         idx2label[idx] = label
#     print("len1", len(data['data']), "len2", len(idx2label))
# "dict_keys(['wav', 'length', 'labels'])"
# print(data['data'][0]['wav'])
# with open('/opt/gpfs/home/chushu/data/audioset/setting/SFT_AS2M/train.tsv', 'r') as f:
#     data = f.readlines()[1:]

# with open('/opt/gpfs/home/chushu/data/audioset/setting/SFT_AS2M/train.lbl', 'w') as f:
#     for i in range(len(data)):
#         idx = data[i].strip().split()[0].split('/')[-1]
#         label = idx2label[idx]
#         f.write(f"{idx}\t{label}\n")

# with open('/home/hanbing/audio_pretrain/fairseq/data_manifest/AS2M/weight_train_all.csv', 'r') as f:
#     weights = f.readlines()
weights = np.loadtxt('/opt/gpfs/home/chushu/data/audioset/setting/SFT_AS2M/weight_train_all.csv')
idx2weight = {}
with open('/opt/gpfs/home/chushu/codes/2506/bgaw/finetune_csv/finetune_AS2M/train.tsv', 'r') as f:
    data = f.readlines()[1:]
    for i in range(len(data)):
        idx = data[i].split()[0].split('/')[-1]
        weight = weights[i]
        idx2weight[idx] = weight

with open('/opt/gpfs/home/chushu/data/audioset/setting/SFT_AS2M/train.tsv', 'r') as f:
    data = f.readlines()[1:]

with open('/opt/gpfs/home/chushu/data/audioset/setting/SFT_AS2M/weights.csv', 'w') as f:
    for i in range(len(data)):
        idx = data[i].strip().split()[0].split('/')[-1]
        weight = idx2weight[idx]
        f.write(f"{weight}\n")