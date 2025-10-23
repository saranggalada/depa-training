#!/bin/bash
docker tag preprocess-genomicslab:latest $CONTAINER_REGISTRY/preprocess-genomicslab:latest
docker push $CONTAINER_REGISTRY/preprocess-genomicslab:latest
docker tag preprocess-pharmaceuticalcompany:latest $CONTAINER_REGISTRY/preprocess-pharmaceuticalcompany:latest
docker push $CONTAINER_REGISTRY/preprocess-pharmaceuticalcompany:latest
docker tag preprocess-computationalbiologylab:latest $CONTAINER_REGISTRY/preprocess-computationalbiologylab:latest
docker push $CONTAINER_REGISTRY/preprocess-computationalbiologylab:latest
docker tag preprocess-cancerinstitute:latest $CONTAINER_REGISTRY/preprocess-cancerinstitute:latest
docker push $CONTAINER_REGISTRY/preprocess-cancerinstitute:latest
