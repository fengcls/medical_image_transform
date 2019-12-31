#!/bin/bash
# in the directory BraTS
mkdir MICCAI_BraTS_2019_Data_Training_by_modality
mkdir MICCAI_BraTS_2019_Data_Training_by_modality/t1
mkdir MICCAI_BraTS_2019_Data_Training_by_modality/t1ce
mkdir MICCAI_BraTS_2019_Data_Training_by_modality/t2
mkdir MICCAI_BraTS_2019_Data_Training_by_modality/flair
mkdir MICCAI_BraTS_2019_Data_Validation_by_modality
mkdir MICCAI_BraTS_2019_Data_Validation_by_modality/t1
mkdir MICCAI_BraTS_2019_Data_Validation_by_modality/t1ce
mkdir MICCAI_BraTS_2019_Data_Validation_by_modality/t2
mkdir MICCAI_BraTS_2019_Data_Validation_by_modality/flair

cp MICCAI_BraTS_2019_Data_Validation/*/*t1.nii.gz MICCAI_BraTS_2019_Data_Validation_by_modality/t1/
cp MICCAI_BraTS_2019_Data_Validation/*/*t1ce.nii.gz MICCAI_BraTS_2019_Data_Validation_by_modality/t1ce/
cp MICCAI_BraTS_2019_Data_Validation/*/*flair.nii.gz MICCAI_BraTS_2019_Data_Validation_by_modality/flair/
cp MICCAI_BraTS_2019_Data_Validation/*/*t2.nii.gz MICCAI_BraTS_2019_Data_Validation_by_modality/t2/

for f in MICCAI_BraTS_2019_Data_Training/LGG/*/*t1.nii.gz;do cp $f MICCAI_BraTS_2019_Data_Training_by_modality/t1/LGG_${f##*/};done
for f in MICCAI_BraTS_2019_Data_Training/LGG/*/*t1ce.nii.gz;do cp $f MICCAI_BraTS_2019_Data_Training_by_modality/t1ce/LGG_${f##*/};done
for f in MICCAI_BraTS_2019_Data_Training/LGG/*/*flair.nii.gz;do cp $f MICCAI_BraTS_2019_Data_Training_by_modality/flair/LGG_${f##*/};done
for f in MICCAI_BraTS_2019_Data_Training/LGG/*/*t2.nii.gz;do cp $f MICCAI_BraTS_2019_Data_Training_by_modality/t2/LGG_${f##*/};done
                                                                                                                                           
for f in MICCAI_BraTS_2019_Data_Training/HGG/*/*t1.nii.gz;do cp $f MICCAI_BraTS_2019_Data_Training_by_modality/t1/HGG_${f##*/};done
for f in MICCAI_BraTS_2019_Data_Training/HGG/*/*t1ce.nii.gz;do cp $f MICCAI_BraTS_2019_Data_Training_by_modality/t1ce/HGG_${f##*/};done
for f in MICCAI_BraTS_2019_Data_Training/HGG/*/*flair.nii.gz;do cp $f MICCAI_BraTS_2019_Data_Training_by_modality/flair/HGG_${f##*/};done
for f in MICCAI_BraTS_2019_Data_Training/HGG/*/*t2.nii.gz;do cp $f MICCAI_BraTS_2019_Data_Training_by_modality/t2/HGG_${f##*/};done
