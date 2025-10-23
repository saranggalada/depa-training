#!/bin/bash

echo "Checking if resource group $AZURE_RESOURCE_GROUP exists..."
RG_EXISTS=$(az group exists --name $AZURE_RESOURCE_GROUP)

if [ "$RG_EXISTS" == "false" ]; then
  echo "Resource group $AZURE_RESOURCE_GROUP does not exist. Creating it now..."
  # Create the resource group
  az group create --name $AZURE_RESOURCE_GROUP --location $AZURE_LOCATION
else
  echo "Resource group $AZURE_RESOURCE_GROUP already exists. Skipping creation."
fi

echo "Check if storage account $STORAGE_ACCOUNT_NAME exists..."
STORAGE_ACCOUNT_EXISTS=$(az storage account check-name --name $AZURE_STORAGE_ACCOUNT_NAME --query "nameAvailable" --output tsv)

if [ "$STORAGE_ACCOUNT_EXISTS" == "true" ]; then
  echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME does not exist. Creating it now..."
  az storage account create  --resource-group $AZURE_RESOURCE_GROUP  --name $AZURE_STORAGE_ACCOUNT_NAME
else
  echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME already exists. Skipping creation."
fi

# Get the storage account key
ACCOUNT_KEY=$(az storage account keys list --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)


# Check if the GENOMICS_LAB container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_GENOMICS_LAB_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
    echo "Container $AZURE_GENOMICS_LAB_CONTAINER_NAME does not exist. Creating it now..."
    az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_GENOMICS_LAB_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the PHARMACEUTICAL_COMPANY container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_PHARMACEUTICAL_COMPANY_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
    echo "Container $AZURE_PHARMACEUTICAL_COMPANY_CONTAINER_NAME does not exist. Creating it now..."
    az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_PHARMACEUTICAL_COMPANY_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the COMPUTATIONAL_BIOLOGY_LAB container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_COMPUTATIONAL_BIOLOGY_LAB_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
    echo "Container $AZURE_COMPUTATIONAL_BIOLOGY_LAB_CONTAINER_NAME does not exist. Creating it now..."
    az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_COMPUTATIONAL_BIOLOGY_LAB_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the CANCER_INSTITUTE container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_CANCER_INSTITUTE_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
    echo "Container $AZURE_CANCER_INSTITUTE_CONTAINER_NAME does not exist. Creating it now..."
    az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_CANCER_INSTITUTE_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the OUTPUT container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_OUTPUT_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
  echo "Container $AZURE_OUTPUT_CONTAINER_NAME does not exist. Creating it now..."
  az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_OUTPUT_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi
