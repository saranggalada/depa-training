#!/bin/bash

# Build pytrain
pushd src/train
python3 setup.py bdist_wheel
popd

# Build training container (CPU — classic pytrain)
docker build -f ci/Dockerfile.train src -t depa-training:latest

# Build LLM fine-tuning container (GPU — pytrain + LLM frameworks)
# docker build -f ci/Dockerfile.llm-train src -t depa-llm-finetune:latest

# Build LLM fine-tuning container (CPU — for local sanity testing)
docker build -f ci/Dockerfile.llm-train-cpu src -t depa-llm-finetune-cpu:latest

# Build encrypted filesystem sidecar
pushd external/confidential-sidecar-containers
./buildall.sh
popd

pushd external/contract-ledger/pyscitt
python3 setup.py bdist_wheel
popd

docker build -f ci/Dockerfile.encfs . -t depa-training-encfs