SCRIPT=ostrack
CONFIG=lasher
LOG=20260615_Refine

nohup python -u lib/train/run_training.py \
    --script ${SCRIPT} \
    --config ${CONFIG} \
    --vis_gpus 6 \
    --save_dir ./output/${LOG} \
    > ./logs/${LOG}-train.log 2>&1 &

echo $! > ./logs/${LOG}-pid.txt
tail -f ./logs/${LOG}-train.log