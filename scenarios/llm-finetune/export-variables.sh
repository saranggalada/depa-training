#!/bin/bash

# Azure resource variables for the LLM fine-tuning scenario.

declare -x SCENARIO=llm-finetune
declare -x REPO_ROOT="$(git rev-parse --show-toplevel)"
declare -x CONTAINER_REGISTRY=ispirt.azurecr.io
declare -x AZURE_LOCATION=<azure-location>
declare -x AZURE_SUBSCRIPTION_ID=
declare -x AZURE_RESOURCE_GROUP=
declare -x AZURE_KEYVAULT_ENDPOINT=
declare -x AZURE_STORAGE_ACCOUNT_NAME=

declare -x AZURE_MARKET_DATA_CONTAINER=marketdatacontainer
declare -x AZURE_FINTECH_CONTAINER=fintechcontainer
declare -x AZURE_BANK_CONTAINER=bankcontainer
declare -x AZURE_MODEL_CONTAINER=modelcontainer
declare -x AZURE_OUTPUT_CONTAINER=outputcontainer

declare -x CONTRACT_SERVICE_URL=https://<contract-service-url>:8000
declare -x TOOLS_HOME=$REPO_ROOT/external/confidential-sidecar-containers/tools

export SCENARIO REPO_ROOT CONTAINER_REGISTRY
export AZURE_LOCATION AZURE_SUBSCRIPTION_ID AZURE_RESOURCE_GROUP
export AZURE_KEYVAULT_ENDPOINT AZURE_STORAGE_ACCOUNT_NAME
export AZURE_MARKET_DATA_CONTAINER AZURE_FINTECH_CONTAINER AZURE_BANK_CONTAINER
export AZURE_MODEL_CONTAINER AZURE_OUTPUT_CONTAINER
export CONTRACT_SERVICE_URL TOOLS_HOME
