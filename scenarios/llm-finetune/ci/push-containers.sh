#!/bin/bash

containers=("llm-finetune-modelsave:latest" "preprocess-market-data:latest" "preprocess-fintech:latest" "preprocess-bank:latest")
for container in "${containers[@]}"
do
  docker tag $container $CONTAINER_REGISTRY"/"$container
  docker push $CONTAINER_REGISTRY"/"$container
done
