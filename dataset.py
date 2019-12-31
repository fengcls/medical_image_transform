import os
import glob
import csv

import numpy as np
import torch
import torch.utils.data

import nibabel as nib

from scipy.io import loadmat

import pickle

def transform_img(f, opt):
    """
    load nifti image and do necessary transform

    :param f: nifti file
    """
    data = np.array(nib.load(f).get_data())
    if not opt.no_normalize:
        data = data/np.percentile(data[data>0], opt.normalize_percentile)
    data = np.transpose(data, (2, 0, 1))
    # special for 384 384 52
    # data = np.concatenate((np.zeros((1, data.shape[1], data.shape[2])), data, np.zeros((1, data.shape[1], data.shape[2]))), axis = 0)

    padb = int((np.ceil(data.shape[0] / 32) * 32 - data.shape[0]) / 2)
    pada = int(np.ceil(data.shape[0] / 32) * 32 - data.shape[0]) - padb
    pad_list = [(padb, pada)]

    for s in data.shape[1:]:
        padb = int((np.ceil(s/32)*32 - s)/2)
        pada = int(np.ceil(s/32)*32 - s) - padb
        pad_list.append((padb, pada))

    pad_tuple = tuple(pad_list)
    data = np.pad(data, pad_tuple, 'constant', constant_values=0)

    return np.expand_dims(data, axis=0).astype('float32')


class image_data(torch.utils.data.Dataset):
    def __init__(self, dirA, dirB, fold_list=[], opt):
        if not opt.study in ["adni-t1-flair","ppp"]:
            raise ValueError("Invalid study group")

        self.dirA = dirA
        self.dirB = dirB
        self.opt = opt

        try:
            self.filesA, self.filesB = self.load_data(fold_list)
        except FileNotFoundError:
            print("CSV information file not found")
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

    def load_data(self, fold_list):
        """
        Load labels from csv file info and files 

        :return labels: dict{patient_id : label}
        :return files: mgh files 
        """
        # load from a pre-defined file
        if len(fold_list) != 0:
            with open(os.path.join(self.dirA, "info.csv"), 'rt') as f:
                reader = csv.reader(f)
                filesA, filesB = [], []
                # rid, fold
                for idx, row in enumerate([x for x in reader][1:]): # excluding the title line
                    if int(row[2]) in fold_list: # in the folds
                        tmpdirA = sorted(glob.glob(self.dirA+'/*{0:04d}*.nii.gz'.format(int(row[0]))))
                        if len(tmpdirA)==0:
                            print(self.dirA+'/*{0:04d}*.nii.gz'.format(int(row[0])))
                            raise ValueError('Dataset not found in A')
                        filesA.append(tmpdirA[0])

                        tmpdirB = sorted(glob.glob(self.dirB + '/*{0:04d}*.nii.gz'.format(int(row[0]))))
                        if len(tmpdirB) == 0:
                            print(self.dirB + '/*{0:04d}*.nii.gz'.format(int(row[0])))
                            raise ValueError('Dataset not found in B')
                        filesB.append(tmpdirB[0])
        else:
            filesA = glob.glob(self.dirA + '/*.nii.gz')
            if len(filesA) == 0:
                print(self.dirA + '/*.nii.gz')
                raise ValueError('Dataset not found in A')

            filesB = glob.glob(self.dirB + '/*.nii.gz')
            if len(filesB) == 0:
                print(self.dirB + '/*.nii.gz')
                raise ValueError('Dataset not found in B')

        return filesA, filesB


    def __getitem__(self, index):
        imgA_path = self.filesA[index]
        imgB_path = self.filesB[index]
        imgA = transform_img(imgA_path, self.opt)
        imgB = transform_img(imgB_path, self.opt)

        return imgA, imgB, imgA_path, imgB_path

    def __len__(self):
        return len(self.filesA)


    def _check_exists(self):
        return len(self.filesA) > 0 and len(self.filesB) > 0
    