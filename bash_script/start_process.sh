#!/usr/bin/env bash

# set -xe

FORCE=$1

BASE_DIR=..
CODE_DIR=${BASE_DIR}/process_data
DATA_DIR=${BASE_DIR}/datasets

# CUSTOMIZED
DATA_NAME=finqa
TRAIN_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/train_retrieve.json
DEV_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/dev_retrieve.json
TEST_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/test_retrieve.json

# DATA_NAME=mathqa
# TRAIN_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/train.json
# DEV_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/dev.json
# TEST_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/test.json

# DATA_NAME=drop
# TRAIN_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/drop_dataset_train.json
# DEV_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/drop_dataset_dev.json
# TEST_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/drop_dataset_test.json

# DATA_NAME=svamp
# TRAIN_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/fake_svamp_train.json
# DEV_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/fake_svamp_dev.json
# TEST_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/svamp_test.json

# DATA_NAME=drop_annotated
# TRAIN_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/train_retrieve.json
# DEV_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/train_retrieve.json
# TEST_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/train_retrieve.json

# DATA_NAME=drop_fewshot
# TRAIN_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/train_retrieve.json
# DEV_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/drop_dataset_dev.json
# TEST_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/train_retrieve.json

# DATA_NAME=drop_fakedata
# TRAIN_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/train_retrieve.json
# DEV_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/drop_dataset_dev.json
# TEST_DATA=${DATA_DIR}/raw_data/${DATA_NAME}/drop_dataset_train.json

SAVE_DIR=${DATA_DIR}/cached_data/${DATA_NAME}/

PLM_TYPE=roberta-large
# PLM_TYPE=t5-small
MERGE_OP=0

IS_PROGRAM_AS_SEQUENCE=0

DATA_LIMIT=-1

if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${DEV_DATA}" ] || [ ! -f "${TEST_DATA}" ]; then
    echo "cannot find dataset from the given path"
    exit
else
    echo "find dataset"
fi

if [ ! -d "${SAVE_DIR}" ]; then
    echo "create save cache directory in: ${SAVE_DIR}"
    mkdir ${SAVE_DIR}
else
    echo "save cache directory exists"
fi

if [ "`ls -A ${SAVE_DIR}`" = "" ] || [ "${FORCE}" = "force" ]; then
    echo "start to process..."

    PATH_ARGS="--data_name ${DATA_NAME}
               --train_data ${TRAIN_DATA}
               --dev_data ${DEV_DATA}
               --test_data ${TEST_DATA}
               --save_dir ${SAVE_DIR}"
    MODEL_ARGS="--plm ${PLM_TYPE} \
                --merge_op ${MERGE_OP} \
                --is_program_as_sequence ${IS_PROGRAM_AS_SEQUENCE}"
    ASSIS_ARGS="--data_limit ${DATA_LIMIT}"

    python ${CODE_DIR}/start.py ${PATH_ARGS} ${MODEL_ARGS} ${ASSIS_ARGS}
else
    echo "cache exists in the ${SAVE_DIR}!"
    exit
fi