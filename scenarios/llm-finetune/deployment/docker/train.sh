export DATA_DIR=$PWD/../../data

export MEDQA_INPUT_PATH=$DATA_DIR/medqa/preprocessed
export MEDQUAD_INPUT_PATH=$DATA_DIR/medquad/preprocessed
export CHATDOCTOR_INPUT_PATH=$DATA_DIR/chatdoctor/preprocessed
export MODEL_INPUT_PATH=$DATA_DIR/modeller/model
export MODEL_OUTPUT_PATH=/tmp/output
mkdir -p $MODEL_OUTPUT_PATH
export CONFIGURATION_PATH=/tmp
cp $PWD/../../config/pipeline_config.json /tmp/pipeline_config.json
docker compose -f docker-compose-train.yml up --remove-orphans
