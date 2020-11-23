#!/bin/bash

BASE_FOLDER=`pwd`
echo $BASE_FOLDER
ROOT_FOLDER=$BASE_FOLDER

source activate myenv

# TO run 3 class classification using CNN
nClasses=3
dataType='noExclusion' 
# noExclusion, withExclusion, balanced
echo '----> You are using the data ' $dataType 'for testing...'

if (( $nClasses == '3' ))
then 
##========================= 3 class category testing ===============================================##
# TO run 3 class classification using CNN on "No exclusion" dataset

    python $BASE_FOLDER/test.py --test_data $BASE_FOLDER/data/$dataType'_test_data.csv' --test_label $BASE_FOLDER/data/$dataType'_test_label.csv' --fileType $dataType \
    --num_labels 3 --ckptFolderAndFile $BASE_FOLDER/ckptDir/$dataType'_train_data_sept_nclasses_3_v20.pth' --classSplit '1,2,3' --gpu_id 0

fi 

##========================= 2 class category training ===============================================##
if (( $nClasses == 2 ))
then 

    echo 'Not implemented!'

fi