#!/usr/bin/env bash

# set -xe

DATA_NAME=finqa

BASE_DIR=..
DATA_DIR=${BASE_DIR}/datasets/cached_data
MODEL_SAVE_DIR=${BASE_DIR}/saved_models_${DATA_NAME}

# CUSTOMIZED
TEST_DATA=${DATA_DIR}/${DATA_NAME}/cached_test_data.json
RELOAD_MODEL_PATH=${MODEL_SAVE_DIR}/checkpoint_best_0.65.pt

PLM=roberta-large
N_LAYERS=4
DROPOUT_P=0.0
MAX_OP_LEN=7
MAX_ARGU_LEN=2
MERGE_OP=0
N_HEAD=8

E_BSZ=10
INFERENCE_RESULTS_PATH=${BASE_DIR}/test_results.json
INFERENCE_WRONG_RESULTS_PATH=${BASE_DIR}/wrong_test_results.json

if [ ! -f ${RELOAD_MODEL_PATH} ]; then
    echo "wrong reload checkpoint path: ${RELOAD_MODEL_PATH}."
    exit
fi

PATH_ARGS="--data_name ${DATA_NAME}
           --cached_test_data ${TEST_DATA}
           --reload_model_path ${RELOAD_MODEL_PATH}"

MODEL_ARGS="--plm ${PLM} \
            --n_layers ${N_LAYERS} \
            --dropout_p ${DROPOUT_P} \
            --max_op_len ${MAX_OP_LEN} \
            --max_argu_len ${MAX_ARGU_LEN} \
            --merge_op ${MERGE_OP} \
            --n_head ${N_HEAD}"

TEST_ARGS="--e_bsz ${E_BSZ} \
           --inference_results_path ${INFERENCE_RESULTS_PATH} \
           --inference_wrong_results_path ${INFERENCE_WRONG_RESULTS_PATH}"
    
python ${BASE_DIR}/test.py \
    ${PATH_ARGS} \
    ${MODEL_ARGS} \
    ${TEST_ARGS}