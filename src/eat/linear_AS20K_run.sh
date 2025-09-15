cd /opt/gpfs/home/chushu/codes/2506/EAT

depth=12
layers=(-1 11 10 9 8 7 6 5 4 3 2 1 0)

for layer in ${layers[@]}; do
    echo "Running experiment with layer: $layer"
    if [ $layer != 11 ]; then
        continue
    fi
    bash /opt/gpfs/home/chushu/codes/2506/EAT/src/eat/linear_AS20K.sh $layer
done