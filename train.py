import time
from options.train_options import TrainOptions
from dataset import image_data
from models import create_model

import shutil
import os
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    trainoptions = TrainOptions()
    opt = trainoptions.parse()

    # setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    projroot = os.path.join(os.getcwd(), opt.study)
    modelroot = os.path.join(projroot, 'save_models')
    dataroot = os.path.join(projroot, 'data')

    modelsubfolder = '{}_{}'.format(opt.study, opt.model)
    modelfolder = os.path.join(modelroot, modelsubfolder)
    opt.name = opt.name + '_' + modelsubfolder

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    opt.input_nc, opt.output_nc = 1, 1

    trainoptions.print_options(opt)

    model = create_model(opt)
    shutil.copy2(__file__, os.path.join(model.save_dir, "script.py"))

    model.setup(opt)
    model.train()

    total_steps = 0

    if opt.train_folds:
        train_set = image_data(opt.dirA, opt.dirB, opt.train_folds, study=opt.study)
        val_set = image_data(opt.dirA, opt.dirB, [11 - opt.test_fold], study=opt.study)
        test_set = image_data(opt.dirA, opt.dirB, [opt.test_fold], study=opt.study)
    else:
        dataset = image_data(opt.dirA, opt.dirB, [], study=opt.study)
        val_len, test_len = np.ceil(opt.val_ratio * len(dataset)), np.ceil(opt.test_ratio * len(dataset))
        split_lengths = [int(len(dataset) - val_len - test_len), int(val_len), int(test_len)]
        train_set, val_set, test_set = random_split(dataset,split_lengths)
        print('train, val, test:', split_lengths)

    train_loader = DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers,
                                               pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_set, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers,
                                             pin_memory=True, drop_last=False)

    test_loader = DataLoader(test_set, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers,
                                              pin_memory=True, drop_last=False)

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    imgA, imgB = next(iter(test_loader))
    print(imgA.shape)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for batch_idx, (imgA, imgB, imgA_path, imgB_path) in tqdm(enumerate(train_loader)):
            data = {}
            data['A']=imgA
            data['B']=imgB
            data['A_paths']=imgA_path
            data['B_paths']=imgB_path

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters(opt.using_gan)

            torch.cuda.empty_cache()

        if epoch % opt.print_freq == 0:
            losses = model.get_current_losses()
            t = (time.time() - epoch_start_time)

            message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, epoch_iter, t)
            for k, v in losses.items():
                message += '%s: %.3f ' % (k, v)
            print(message)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
