#!/bin/bash
docker build -f ci/Dockerfile.genomicslab src -t preprocess-genomics-lab:latest
docker build -f ci/Dockerfile.pharmaceuticalcompany src -t preprocess-pharmaceutical-company:latest
docker build -f ci/Dockerfile.computationalbiologylab src -t preprocess-computational-biology-lab:latest
docker build -f ci/Dockerfile.cancerinstitute src -t preprocess-cancer-institute:latest

# # DEBUG: feature engineering
# docker build -f ci/Dockerfile.datacollab src -t oncology-feature-engineering:latest
