oms node list

# lr_float=0.0005
# lr_float=0.0001
lr_float=0.00005
# lr_float=0.00001

name=zcs-cfc-eat-as20k-${lr_float}
echo "##### delete job ${name} #####\n"
oms job delete ${name}
echo "##### delete job ${name} done #####\n"

gpus=1
cpus=32
memgb=128
shmgb=32
replicas=1
echo "##### run parameters #####\n"
echo "gpus: ${gpus}"
echo "cpus: ${cpus}"
echo "memgb: ${memgb}"
echo "shmgb: ${shmgb}"
echo "replicas: ${replicas}"
echo "##### run parameters done #####\n"

# queue tye option: [queue-h100-4n, queue-rtx4090-2n]
echo "##### submit job ${name} #####\n"
oms job submit --image lunalabs-acr-registry.cn-guangzhou.cr.aliyuncs.com/luna/zcs-20250811 \
--name ${name} \
--queue queue-h100-4n \
--gpus ${gpus} \
--cpus ${cpus} \
--memgb ${memgb} \
--shmgb ${shmgb} \
--replicas ${replicas} \
--launch-command "bash /opt/gpfs/home/chushu/codes/2506/EAT/src/eat/classification_scratch_AS20K_oms.sh ${gpus} ${lr_float}"

oms job pods ${name}
echo "##### submit job ${name} done #####\n"


echo "##### job id: $(oms job list | grep ${name}) #####\n"
echo "##### check pod command: oms job pods ${name} #####\n"
echo "##### check log command: oms pod logs <pod-name> #####\n"