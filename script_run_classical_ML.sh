#!/bin/bash

BASE_FOLDER=`pwd`
echo $BASE_FOLDER
ROOT_FOLDER=$BASE_FOLDER
source activate myenv

# for 3 class training use 3 else use 2
nClasses=2
dataType='noExclusion' 
# [noExclusion, withExclusion, balanced] ----> [0, 1, 2]

if (( $nClasses == '3' ))
then 
##========================= Classical ML for 3 class category training ===============================================##
    echo '---->' $dataType 'data for training...'
    python $BASE_FOLDER/knn_PCA_LDA_SVM.py --train_data $BASE_FOLDER/data/$dataType'_train_data.csv' --train_label $BASE_FOLDER/data/$dataType'_train_label.csv' \
    --test_data $BASE_FOLDER/data/$dataType'_test_data.csv' --test_label $BASE_FOLDER/data/$dataType'_test_label.csv' --num_labels 3  --classSplit '1,2,3'

fi 

##========================= Classical ML 2 class category training ===============================================##
if (( $nClasses == 2 ))
then 

    echo '---->' $dataType 'data for training...'
    classPairSrting='1,2'
    classA='1'
    classB='2'
    python $BASE_FOLDER/knn_PCA_LDA_SVM.py --train_data $BASE_FOLDER/data/$dataType'_train_data_'$classA'_'$classB'.csv' --train_label $BASE_FOLDER/data/$dataType'_train_label_'$classA'_'$classB'.csv' \
    --test_data $BASE_FOLDER/data/$dataType'_test_data_'$classA'_'$classB'.csv' --test_label $BASE_FOLDER/data/$dataType'_test_label_'$classA'_'$classB'.csv' \
     --num_labels 2  --classSplit $classPairSrting

    classPairSrting='1,3'
    classA='1'
    classB='3'
    python $BASE_FOLDER/knn_PCA_LDA_SVM.py --train_data $BASE_FOLDER/data/$dataType'_train_data_'$classA'_'$classB'.csv' --train_label $BASE_FOLDER/data/$dataType'_train_label_'$classA'_'$classB'.csv' \
    --test_data $BASE_FOLDER/data/$dataType'_test_data_'$classA'_'$classB'.csv' --test_label $BASE_FOLDER/data/$dataType'_test_label_'$classA'_'$classB'.csv' \
     --num_labels 2  --classSplit $classPairSrting


    classPairSrting='2,3'
    classA='2'
    classB='3'
    python $BASE_FOLDER/knn_PCA_LDA_SVM.py --train_data $BASE_FOLDER/data/$dataType'_train_data_'$classA'_'$classB'.csv' --train_label $BASE_FOLDER/data/$dataType'_train_label_'$classA'_'$classB'.csv' \
    --test_data $BASE_FOLDER/data/$dataType'_test_data_'$classA'_'$classB'.csv' --test_label $BASE_FOLDER/data/$dataType'_test_label_'$classA'_'$classB'.csv' \
     --num_labels 2  --classSplit $classPairSrting


fi