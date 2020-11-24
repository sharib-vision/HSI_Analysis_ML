#!/bin/bash
#@author: sharib

BASE_FOLDER=`pwd`
echo $BASE_FOLDER
ROOT_FOLDER=$BASE_FOLDER
source activate myenv

# for 3 class training use 3 else use 2
nClasses=2

dataType=0 # selection 0 , 1, 2
# [noExclusion, withExclusion, balanced] ----> [0, 1, 2]

if (( $nClasses == '2' ))
then 
##========================= 3 class category training ===============================================##
# TO run 2 class classification using CNN on "No exclusion" dataset

# comment the block if you want to run just one!!!

    if (( $dataType == 0 )) 
    then
    echo '----> noExclusion data for training for class 1 and 2...'
    python $BASE_FOLDER/train_2class.py --train_data $BASE_FOLDER/data/noExclusion_train_data_1_2.csv --train_label $BASE_FOLDER/data/noExclusion_train_label_1_2.csv \
    --num_labels $nClasses --batch_size 16 --val_batch_size 4 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0 --classSplit '1,2'


    echo '----> noExclusion data for training for class 1 and 3...'
    python $BASE_FOLDER/train_2class.py --train_data $BASE_FOLDER/data/noExclusion_train_data_1_3.csv --train_label $BASE_FOLDER/data/noExclusion_train_label_1_3.csv \
    --num_labels $nClasses --batch_size 16 --val_batch_size 4 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0 --classSplit '1,3'


    echo '----> noExclusion data for training for class 2 and 3...'
    python $BASE_FOLDER/train_2class.py --train_data $BASE_FOLDER/data/noExclusion_train_data_2_3.csv --train_label $BASE_FOLDER/data/noExclusion_train_label_2_3.csv \
    --num_labels $nClasses --batch_size 16 --val_batch_size 4 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0 --classSplit '2,3'


    elif (( $dataType == 1 )) 
    then
    echo '----> withExclusion data for training for class 1 and 2...'
    python $BASE_FOLDER/train.py --train_data $BASE_FOLDER/data/withExclusion_train_data_1_2.csv --train_label $BASE_FOLDER/data/withExclusion_train_labela_1_2.csv \
    --num_labels 3 --batch_size 16 --val_batch_size 4 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0 --classSplit '1,2'

    echo '----> withExclusion data for training for class 1 and 3...'
    python $BASE_FOLDER/train.py --train_data $BASE_FOLDER/data/withExclusion_train_data_1_3.csv --train_label $BASE_FOLDER/data/withExclusion_train_labela_1_3.csv \
    --num_labels 3 --batch_size 16 --val_batch_size 4 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0 --classSplit '1,3'

    echo '----> withExclusion data for training for class 2 and 3...'
    python $BASE_FOLDER/train.py --train_data $BASE_FOLDER/data/withExclusion_train_data_2_3.csv --train_label $BASE_FOLDER/data/withExclusion_train_labela_2_3.csv \
    --num_labels 3 --batch_size 16 --val_batch_size 4 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0 --classSplit '2,3'

    else
     echo '----> balanced data for training for class 1 and 2...'
    python $BASE_FOLDER/train.py --train_data $BASE_FOLDER/data/balanced_train_data_1_2.csv --train_label $BASE_FOLDER/data/balanced_train_label_1_2.csv \
    --num_labels 3 --batch_size 16 --val_batch_size 4 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0 --classSplit '1,2'

    echo '----> balanced data for training for class 1 and 3...'

    python $BASE_FOLDER/train.py --train_data $BASE_FOLDER/data/balanced_train_data_1_3.csv --train_label $BASE_FOLDER/data/balanced_train_label_1_3.csv \
    --num_labels 3 --batch_size 16 --val_batch_size 4 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0 --classSplit '1,3'

    echo '----> balanced data for training for class 2 and 3...'
    python $BASE_FOLDER/train.py --train_data $BASE_FOLDER/data/balanced_train_data_2_3.csv --train_label $BASE_FOLDER/data/balanced_train_label_2_3.csv \
    --num_labels 3 --batch_size 16 --val_batch_size 4 --trainsplit 0.20 --outputFolder_ckpt $BASE_FOLDER/ckptDir --gpu_id 0 --classSplit  '2,3'

    fi

fi 

##========================= 3 class category training ===============================================##
if (( $nClasses == 3 ))
then 

    echo 'Not implemented! See script_train.sh'

fi