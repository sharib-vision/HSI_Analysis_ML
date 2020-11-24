#!/bin/bash

BASE_FOLDER=`pwd`
echo $BASE_FOLDER
ROOT_FOLDER=$BASE_FOLDER
source activate myenv

# for 3 class training use 3 else use 2
nClasses=3

dataType=0 # selection 0 , 1, 2
# [noExclusion, withExclusion, balanced] ----> [0, 1, 2]

if (( $nClasses == '3' ))
then 
##========================= 3 class category training ===============================================##
# TO run 3 class classification using CNN on "No exclusion" dataset

    if (( $dataType == 0 )) 
    then
    echo '----> noExclusion data for training...'
    python $BASE_FOLDER/train.py --train_data $BASE_FOLDER/data/noExclusion_train_data.csv --train_label $BASE_FOLDER/data/noExclusion_train_label.csv \
    --num_labels 3 --batch_size 16 --val_batch_size 8 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0

    elif (( $dataType == 1 )) 
    then
    echo '----> withExclusion data for training...'
    # TO run 3 class classification using CNN on "With exclusion" dataset
    python $BASE_FOLDER/train.py --train_data $BASE_FOLDER/data/withExclusion_train_data.csv --train_label $BASE_FOLDER/data/withExclusion_train_label.csv \
    --num_labels 3 --batch_size 16 --val_batch_size 8 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0

    else
     echo '----> balanced data for training...'
    # TO run 3 class classification using CNN on "balanced" dataset
    python $BASE_FOLDER/train.py --train_data $BASE_FOLDER/data/balanced_train_data.csv --train_label $BASE_FOLDER/data/balanced_train_label.csv \
    --num_labels 3 --batch_size 16 --val_batch_size 8 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0
    fi

fi 

##========================= 2 class category training ===============================================##
if (( $nClasses == 2 ))
then 

    echo 'Not implemented!'

fi