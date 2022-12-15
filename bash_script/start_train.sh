#!/usr/bin/env bash

# set -xe

USE_WANDB=$1
[ "${USE_WANDB}" == "" ] && USE_WANDB=0 || USE_WANDB=${USE_WANDB}

JOINED_TRAIN=False

BASE_DIR=..
DATA_DIR=${BASE_DIR}/datasets/cached_data

# CUSTOMIZED
DATA_NAME=finqa
TRAIN_DATA=${DATA_DIR}/${DATA_NAME}/cached_train_data.json
DEV_DATA=${DATA_DIR}/${DATA_NAME}/cached_test_data.json
TEST_DATA=${DATA_DIR}/${DATA_NAME}/cached_dev_data.json

# DATA_NAME=mathqa
# TRAIN_DATA=${DATA_DIR}/${DATA_NAME}/cached_train_data.json
# DEV_DATA=${DATA_DIR}/${DATA_NAME}/cached_test_data.json
# TEST_DATA=${DATA_DIR}/drop/cached_test_data.json

# DATA_NAME=drop_annotated
# TRAIN_DATA=${DATA_DIR}/${DATA_NAME}/cached_train_data.json
# DEV_DATA=${DATA_DIR}/${DATA_NAME}/cached_train_data.json
# TEST_DATA=${DATA_DIR}/${DATA_NAME}/cached_train_data.json

# DATA_NAME=drop_fewshot
# TRAIN_DATA=${DATA_DIR}/${DATA_NAME}/cached_train_data.json
# DEV_DATA=${DATA_DIR}/${DATA_NAME}/cached_dev_data.json
# TEST_DATA=${DATA_DIR}/${DATA_NAME}/cached_train_data.json

# DATA_NAME=drop_fakedata
# TRAIN_DATA=${DATA_DIR}/${DATA_NAME}/cached_train_data.json
# DEV_DATA=${DATA_DIR}/${DATA_NAME}/cached_dev_data.json
# TEST_DATA=${DATA_DIR}/${DATA_NAME}/cached_train_data.json

MODEL_SAVE_DIR=${BASE_DIR}/saved_models_${DATA_NAME}
EVAL_RESULTS_DIR=${MODEL_SAVE_DIR}

# RELOAD_MODEL_PATH=${MODEL_SAVE_DIR}/checkpoint_best_0.68.pt
# RELOAD_CONFIG_PATH=${MODEL_SAVE_DIR}/checkpoint_best_0.68.ct
# RELOAD_OPTIMIZER_PATH=${MODEL_SAVE_DIR}/checkpoint_best_0.68.op
# RELOAD_SCHEDULER_PATH=${MODEL_SAVE_DIR}/checkpoint_best_0.68.lr

PLM=roberta-large
# PLM=t5-small
N_LAYERS=4
DROPOUT_P=0.1
MAX_OP_LEN=7
MAX_ARGU_LEN=2
MERGE_OP=0
N_HEAD=8
IS_PROGRAM_AS_SEQUENCE=0

MAX_EPOCH=50
T_BSZ=10
E_BSZ=10
GRADIENT_ACCUMULATION_STEPS=1
LR=1e-5
WD=1e-5
FINE_TUNE=1
SCHEDULED_SAMPLING=0
SAMPLING_K=300

LOG_PER_UPDATES=10
SAVE_EVERY_STEPS=1000

if [ ! -d ${MODEL_SAVE_DIR} ]; then
    echo "create directory to save model: ${MODEL_SAVE_DIR}"
    mkdir ${MODEL_SAVE_DIR}
fi

PATH_ARGS="--data_name ${DATA_NAME} \
           --cached_train_data ${TRAIN_DATA} \
           --cached_dev_data ${DEV_DATA} \
           --model_save_dir ${MODEL_SAVE_DIR} \
           --eval_results_dir ${EVAL_RESULTS_DIR} 
           --cached_test_data ${TEST_DATA}"

# PATH_ARGS="--data_name ${DATA_NAME} \
#            --cached_train_data ${TRAIN_DATA} \
#            --cached_dev_data ${DEV_DATA} \
#            --model_save_dir ${MODEL_SAVE_DIR} \
#            --eval_results_dir ${EVAL_RESULTS_DIR}  \
#            --cached_test_data ${TEST_DATA} \
#            --reload_model_path ${RELOAD_MODEL_PATH} \
#            --reaload_config_path ${RELOAD_CONFIG_PATH} \
#            --reload_optimizer_path ${RELOAD_OPTIMIZER_PATH} \
#            --reload_scheduler_path ${RELOAD_SCHEDULER_PATH}"

MODEL_ARGS="--plm ${PLM} \
            --n_layers ${N_LAYERS} \
            --dropout_p ${DROPOUT_P} \
            --max_op_len ${MAX_OP_LEN} \
            --max_argu_len ${MAX_ARGU_LEN} \
            --merge_op ${MERGE_OP} \
            --n_head ${N_HEAD} \
            --is_program_as_sequence ${IS_PROGRAM_AS_SEQUENCE}"

TRAIN_ARGS="--t_bsz ${T_BSZ} \
            --e_bsz ${E_BSZ} \
            --max_epoch ${MAX_EPOCH} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --lr ${LR} \
            --weight_decay ${WD} \
            --fine_tune ${FINE_TUNE} \
            --sheduled_sampling ${SCHEDULED_SAMPLING} \
            --sampling_k ${SAMPLING_K}"

ASSIST_ARGS="--wandb ${USE_WANDB} \
             --log_per_updates ${LOG_PER_UPDATES} \
             --save_every_steps ${SAVE_EVERY_STEPS}"

if [ "${JOINED_TRAIN}" == "True" ]; then
    python ${BASE_DIR}/joined_train.py \
        ${PATH_ARGS} \
        ${MODEL_ARGS} \
        ${TRAIN_ARGS} \
        ${ASSIST_ARGS} \
        --cached_test_data ${TEST_DATA}
elif [ "${DATA_NAME}" == "drop_annotated" ]; then
    python ${BASE_DIR}/train_kfold.py \
        ${PATH_ARGS} \
        ${MODEL_ARGS} \
        ${TRAIN_ARGS} \
        ${ASSIST_ARGS}
else
    python ${BASE_DIR}/train.py \
        ${PATH_ARGS} \
        ${MODEL_ARGS} \
        ${TRAIN_ARGS} \
        ${ASSIST_ARGS}
fi