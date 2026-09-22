SCRIPT=ostrack
CONFIG=pmrl_lasher
DATASET=LasHeR-Unaligned
EPOCH=59
MODAL=output/PMRL
ENDFIX="_pmrl"
python RGBT_workspace/test_rgbt_mgpus_stepalign.py \
    --script_name ${SCRIPT} \
    --yaml_name ${CONFIG} \
    --dataset_name ${DATASET} \
    --threads 8 \
    --num_gpus 2 \
    --checkpoint_path ${MODAL} \
    --vis_gpus 0,1 \
    --template_update_interval 10 \
    --template_score_thr 0.4 \
    --tocu_mode new \
    --epoch ${EPOCH} \
    --end_fix ${ENDFIX}

