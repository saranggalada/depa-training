#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

./generatefs.sh -d $DATADIR/market_data/preprocessed -k $DATADIR/market_data_key.bin -i $DATADIR/market_data.img
./generatefs.sh -d $DATADIR/fintech/preprocessed -k $DATADIR/fintech_key.bin -i $DATADIR/fintech.img
./generatefs.sh -d $DATADIR/bank/preprocessed -k $DATADIR/bank_key.bin -i $DATADIR/bank.img
./generatefs.sh -d $MODELDIR/models -k $MODELDIR/model_key.bin -i $MODELDIR/model.img

sudo rm -rf $MODELDIR/output
mkdir -p $MODELDIR/output
./generatefs.sh -d $MODELDIR/output -k $MODELDIR/output_key.bin -i $MODELDIR/output.img
