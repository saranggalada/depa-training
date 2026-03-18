#!/bin/bash

docker build -f ci/Dockerfile.modelsave src -t llm-finetune-modelsave:latest
docker build -f ci/Dockerfile.market-data src -t preprocess-market-data:latest
docker build -f ci/Dockerfile.fintech src -t preprocess-fintech:latest
docker build -f ci/Dockerfile.bank src -t preprocess-bank:latest