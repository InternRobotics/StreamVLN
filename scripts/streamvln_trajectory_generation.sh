#!/bin/bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

set -x
umask 000

TIME=$(date +%Y%m%d_%H%M)
DATASET=R2R
CONFIG_PATH=config/vln_r2r.yaml
OUTPUT_PATH=data/trajectory_data/${DATASET}/${TIME}
DATA_PATH=data/datasets/envdrop/envdrop.json.gz  # Set to None to use default dataset path

mkdir -p ${OUTPUT_PATH}
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT streamvln/streamvln_trajectory_generation.py \
    --dataset ${DATASET} \
    --config_path ${CONFIG_PATH} \
    --output_path ${OUTPUT_PATH} \
    --data_path ${DATA_PATH} \
    > ${OUTPUT_PATH}/log.log 2>&1