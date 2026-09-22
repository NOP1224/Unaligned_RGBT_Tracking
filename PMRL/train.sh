SCRIPT=ostrack
CONFIG=pmrl_luart
LOG=PMRL_LUART
# 选择多卡
GPUS=0,1
NPROC_PER_NODE=$(echo "${GPUS}" | awk -F',' '{print NF}')

export CUDA_VISIBLE_DEVICES=${GPUS}

mkdir -p ./logs
nohup torchrun \
    --standalone \
    --nproc_per_node=${NPROC_PER_NODE} \
    lib/train/run_training.py \
    --script ${SCRIPT} \
    --config ${CONFIG} \
    --vis_gpus 0 \
    --save_dir output/${LOG} \
    > ./logs/${LOG}-train.log 2>&1 &

echo $! > ./logs/${LOG}-pid.txt
tail -f ./logs/${LOG}-train.log

