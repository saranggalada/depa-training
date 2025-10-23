#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

ACCOUNT_KEY=$(az storage account keys list --account-name $AZURE_STORAGE_ACCOUNT_NAME --only-show-errors | jq -r .[0].value)

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_GENOMICS_LAB_CONTAINER_NAME \
  --file $DATADIR/genomics_lab.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_PHARMACEUTICAL_COMPANY_CONTAINER_NAME \
  --file $DATADIR/pharmaceutical_company.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_COMPUTATIONAL_BIOLOGY_LAB_CONTAINER_NAME \
  --file $DATADIR/computational_biology_lab.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_CANCER_INSTITUTE_CONTAINER_NAME \
  --file $DATADIR/cancer_institute.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_OUTPUT_CONTAINER_NAME \
  --file $MODELDIR/output.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY
