#!/bin/bash
# @author: sharib

BASE_FOLDER=`pwd`
echo $BASE_FOLDER
ROOT_FOLDER=$BASE_FOLDER

source activate myenv

# TO run 3 class classification using CNN
nClasses=2 # change to 2 if two class and 3 for 3 class classification
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
classPairSrting='1,2'
classA='1'
classB='2'
if (( $nClasses == 2 ))
then 

    classPairSrting='1,2'
    classA='1'
    classB='2'
    python $BASE_FOLDER/test_2class.py --test_data $BASE_FOLDER/data/$dataType'_test_data_'$classA'_'$classB'.csv' --test_label $BASE_FOLDER/data/$dataType'_test_label_'$classA'_'$classB'.csv' --fileType $dataType \
    --num_labels 2 --ckptFolderAndFile $BASE_FOLDER/ckptDir/$dataType'_sept_nclasses_2_v20_'$classA'_'$classB'.pth' --classSplit $classPairSrting --gpu_id 0




    classPairSrting='1,3'
    classA='1'
    classB='3'
    python $BASE_FOLDER/test_2class.py --test_data $BASE_FOLDER/data/$dataType'_test_data_'$classA'_'$classB'.csv' --test_label $BASE_FOLDER/data/$dataType'_test_label_'$classA'_'$classB'.csv' --fileType $dataType \
    --num_labels 2 --ckptFolderAndFile $BASE_FOLDER/ckptDir/$dataType'_sept_nclasses_2_v20_'$classA'_'$classB'.pth' --classSplit $classPairSrting --gpu_id 0



    classPairSrting='2,3'
    classA='2'
    classB='3'
    python $BASE_FOLDER/test_2class.py --test_data $BASE_FOLDER/data/$dataType'_test_data_'$classA'_'$classB'.csv' --test_label $BASE_FOLDER/data/$dataType'_test_label_'$classA'_'$classB'.csv' --fileType $dataType \
    --num_labels 2 --ckptFolderAndFile $BASE_FOLDER/ckptDir/$dataType'_sept_nclasses_2_v20_'$classA'_'$classB'.pth' --classSplit $classPairSrting --gpu_id 0

fi