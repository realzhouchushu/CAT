cd /opt/gpfs/home/chushu/codes/2506/EAT

depth=12
layers=(0 11 10 9 8 7 6 5 4 3 2 1)

for layer in ${layers[@]}; do
    echo "Running experiment with layer: $layer"
    # if [ $layer != 11 ]; then
    #     continue
    # fi
    # bash /opt/gpfs/home/chushu/codes/2506/EAT/src/eat/linear_AS20K_oms_run.sh $layer
    bash /opt/gpfs/home/chushu/codes/2506/EAT/src/eat/finetuning_AS20K_oms_run.sh $layer
    sleep 12m
done