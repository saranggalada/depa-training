#!/bin/bash

echo "Checking if resource group $AZURE_RESOURCE_GROUP exists..."
RG_EXISTS=$(az group exists --name $AZURE_RESOURCE_GROUP)

if [ "$RG_EXISTS" == "false" ]; then
  echo "Resource group $AZURE_RESOURCE_GROUP does not exist. Creating it now..."
  az group create --name $AZURE_RESOURCE_GROUP --location $AZURE_LOCATION
else
  echo "Resource group $AZURE_RESOURCE_GROUP already exists. Skipping creation."
fi

echo "Check if storage account $AZURE_STORAGE_ACCOUNT_NAME exists..."
STORAGE_ACCOUNT_EXISTS=$(az storage account check-name --name $AZURE_STORAGE_ACCOUNT_NAME --query "nameAvailable" --output tsv)

if [ "$STORAGE_ACCOUNT_EXISTS" == "true" ]; then
  echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME does not exist. Creating it now..."
  az storage account create --resource-group $AZURE_RESOURCE_GROUP --name $AZURE_STORAGE_ACCOUNT_NAME
else
  STORAGE_ACCOUNT_EXIST_IN_RG=$(az storage account show --name $AZURE_STORAGE_ACCOUNT_NAME --resource-group $AZURE_RESOURCE_GROUP --query "name" -o tsv 2>/dev/null)
  if [ -z "$STORAGE_ACCOUNT_EXIST_IN_RG" ]; then
    echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME is already reserved and does not exist in the $AZURE_RESOURCE_GROUP resource group. Please select a different name."
    exit 1
  fi
  echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME already exists in the $AZURE_RESOURCE_GROUP resource group. Skipping creation."
fi

ACCOUNT_KEY=$(az storage account keys list --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)

for CONTAINER_NAME in "$AZURE_MARKET_DATA_CONTAINER" "$AZURE_FINTECH_CONTAINER" "$AZURE_BANK_CONTAINER" "$AZURE_MODEL_CONTAINER" "$AZURE_OUTPUT_CONTAINER"; do
  CONTAINER_EXISTS=$(az storage container exists --name $CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)
  if [ "$CONTAINER_EXISTS" == "false" ]; then
    echo "Container $CONTAINER_NAME does not exist. Creating it now..."
    az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $CONTAINER_NAME --account-key $ACCOUNT_KEY
  else
    echo "Container $CONTAINER_NAME already exists."
  fi
done
