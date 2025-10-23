#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

./generatefs.sh -d $DATADIR/genomics_lab/preprocessed -k $DATADIR/genomics_lab_key.bin -i $DATADIR/genomics_lab.img
./generatefs.sh -d $DATADIR/pharmaceutical_company/preprocessed -k $DATADIR/pharmaceutical_company_key.bin -i $DATADIR/pharmaceutical_company.img
./generatefs.sh -d $DATADIR/computational_biology_lab/preprocessed -k $DATADIR/computational_biology_lab_key.bin -i $DATADIR/computational_biology_lab.img
./generatefs.sh -d $DATADIR/cancer_institute/preprocessed -k $DATADIR/cancer_institute_key.bin -i $DATADIR/cancer_institute.img

sudo rm -rf $MODELDIR/output
mkdir -p $MODELDIR/output
./generatefs.sh -d $MODELDIR/output -k $MODELDIR/output_key.bin -i $MODELDIR/output.img
