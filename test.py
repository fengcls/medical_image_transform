import os, sys, h5py, cv2, torch
from options.test_options import TestOptions
from models import create_model
from dataset import image_data
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import datetime
import pickle
from skimage.measure import compare_ssim as ssim

import nibabel as nib

if __name__ == '__main__':
    testoptions = TestOptions()
    opt = testoptions.parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.input_nc, opt.output_nc = 1, 1

    testoptions.print_options(opt)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    ## directory
    projroot = os.path.join(os.getcwd()) 
    modelroot = os.path.join(projroot, 'checkpoints')
    dataroot = os.path.join(projroot,'data')
    opt.name += '_' + '{}_{}'.format(opt.study, opt.model)
    modelfolder = os.path.join(modelroot, opt.name)

    weights_name = opt.weights_name # 'weights.pth'
    weightspath = os.path.join(modelfolder, weights_name)
    save_folder = os.path.join(projroot,'result',opt.name + '_pred')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
 
    ## ground truth
    torch.manual_seed(opt.seed)

    model = create_model(opt)
    model.setup(opt)

    dataset = image_data(opt.dirA, opt.dirB, [], study=opt.study)
    data_loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers,
                                               pin_memory=True, drop_last=False)

    im_all, err_all = None, None
    mse_all = []
    for batch_idx, (imgA, imgB, imgA_path, imgB_path) in tqdm(enumerate(data_loader)):
        data = {}
        data['A']=imgA
        data['B']=imgB
        data['A_paths']= imgA_path
        data['B_paths']= imgB_path

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        for label, im_data in visuals.items():
            if label == 'fake_B':
                im = im_data.cpu().float().numpy()
                imgB = imgB.float().numpy()
                err = np.expand_dims(np.squeeze(np.square(im - imgB)),axis=0)
                imgB_path = imgB_path[0]
                nib_img = nib.load(imgB_path)
                nib_img_new = nib.Nifti1Image(np.squeeze(im), nib_img.affine)
                nib_img_pred_name = os.path.join(save_folder, imgB_path.split('/')[-1].replace('.nii.gz','_pred.nii.gz'))
                if os.path.exists(nib_img_pred_name):
                    nib.save(nib_img_new, nib_img_pred_name)
                nib_img_new = nib.Nifti1Image(np.squeeze(err), nib_img.affine)
                nib_img_err_name = os.path.join(save_folder, imgB_path.split('/')[-1].replace('.nii.gz','_err.nii.gz'))
                if os.path.exists(nib_img_err_name):
                    nib.save(nib_img_new, nib_img_err_name)
                nib_img_new = nib.Nifti1Image(np.squeeze(err)/np.squeeze(imgB), nib_img.affine)
                nib_img_err_ratio_name = os.path.join(save_folder, imgB_path.split('/')[-1].replace('.nii.gz','_err_ratio.nii.gz'))
                if os.path.exists(nib_img_err_name):
                    nib.save(nib_img_new, nib_img_err_ratio_name)

                mse = np.mean(err.flatten())
                mse_all.append(mse)
    


