cd /opt/gpfs/home/chushu/codes/2506/EAT
pip install nvitop
pip install --editable ./

bash ./src/eat/pretraining_AS2M.sh ${1}
