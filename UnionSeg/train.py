# Code adapted from FOUND: https://github.com/valeoai/FOUND/tree/main
# Modified by the author of UnionCut
from DataLoader import UnionSegDataset
import torch.utils.data as Data
from model import UnionSeg
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CRF'))
from crf import densecrf
import torch.nn.functional as F
import argparse


def setup_seed(seed=3407):
    import random
    import os
    import numpy as np
    random.seed(seed)  # random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed)  # random seed for hash
    np.random.seed(seed)  # numpy random seed
    torch.manual_seed(seed)  # torch random seed for cpu
    torch.cuda.manual_seed(seed)  # torch random seed for gpu
    torch.cuda.manual_seed_all(seed)  # random seed for multi-GPU
    torch.backends.cudnn.deterministic = True  # deterministic selection
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser('UnionSeg training script')
    # default arguments
    parser.add_argument('--img-folder', type=str, default="/home/wzl/DataSet/", help='parent folder path of the training dataset')
    parser.add_argument('--h5-path', type=str, default="/media/wzl/T7/PhysicalImageAugmentation/PseudoMaskGenration/DUTS/h5/TokenGraphCutN1_tao0.2_DUTS_Train_ViT_S_16_flexible_8_fixed.h5", help='path of the h5 file saving the output of UnionCut')
    args = parser.parse_args()

    # set hyper parameters
    seed = 0
    max_iter = 500
    nb_epochs = 3
    batch_size = 50
    lr0 = 5e-2
    step_lr_size = 50
    step_lr_gamma = 0.95
    w_bs_loss = 1.5
    w_self_loss = 4.0
    stop_bkg_loss = 100
    save_model_freq = 100

    # setup random seed
    setup_seed(seed)

    # initialize model
    net = UnionSeg()
    sigmoid = nn.Sigmoid()

    # initialize training devices
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.decoder.train()
    net.decoder.to(train_device)

    # initialize dataset
    dataset = UnionSegDataset(args.img_folder,
                              args.h5_path)
    loader = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    # initialize loss function
    loss_func = nn.BCEWithLogitsLoss()

    # initialize optimizer
    optimizer = torch.optim.AdamW(
        net.decoder.parameters(),
        lr=lr0
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_lr_size,
        gamma=step_lr_gamma,
    )
    # train
    n_iter = 0
    for epoch in range(nb_epochs):
        for step, (img_ts, img_inits, mask_gts, img_paths) in enumerate(loader):
            # move the input to traning device
            torch_images = img_ts.to(train_device)
            # Forward steps
            preds, _, shape_f, att = net(torch_images)
            # Compute mask detection
            preds_mask = (sigmoid(preds.detach()) > 0.5).float()
            # use crf to smooth the mask
            crf_pred_masks = []
            for i in range(batch_size):
                # decode the initial image
                img_init = np.transpose(img_inits[i].cpu().numpy(), (1, 2, 0))
                img_init = (img_init * 255).astype(np.uint8)
                img_init = cv2.cvtColor(img_init, cv2.COLOR_RGB2BGR)
                # apply crf
                crf_mask = densecrf(img_init, preds_mask[i, 0].cpu().detach().numpy().astype(np.uint8)).astype(np.uint8)
                crf_mask = cv2.resize(crf_mask, (224, 224), cv2.INTER_NEAREST)
                # calculate the IOU between the UnionSeg prediction and UnionCut output
                UnionCut_mask = mask_gts[i].numpy()
                intersection = crf_mask * UnionCut_mask
                union = crf_mask + UnionCut_mask - intersection
                IoU = np.sum(intersection) / np.sum(union)
                # use the prediction if IoU > 0.5
                if IoU >= 0.5:
                    crf_pred_masks.append(cv2.resize(crf_mask, (28, 28), cv2.INTER_NEAREST)[None, :, :])
                else:
                    crf_pred_masks.append(cv2.resize(UnionCut_mask, (28, 28), cv2.INTER_NEAREST)[None, :, :])
            # make the label of UnionSeg prediction
            preds_mask = torch.from_numpy(np.stack(crf_pred_masks)).type(torch.LongTensor).to(train_device)

            # Compute loss
            flat_preds = preds.permute(0, 2, 3, 1).reshape(-1, 1)
            preds_mask = preds_mask.permute(0, 2, 3, 1).reshape(-1, 1)
            loss = w_bs_loss * loss_func(flat_preds, preds_mask.float())

            # Apply bkg loss
            if n_iter < stop_bkg_loss:
                # pseudo_mask vs preds [loss]
                flat_labels = F.interpolate(mask_gts.unsqueeze(1), (28, 28)).permute(0, 2, 3, 1).reshape(-1, 1).to(train_device)
                loss += loss_func(flat_preds, flat_labels.float())
            # Add regularization when bkg loss stopped
            else:
                self_loss = loss_func(flat_preds, preds_mask.float())
                self_loss = w_self_loss * self_loss
                loss += self_loss
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print progress
            print('epoch:', epoch, '|step:', step, '|train loss:%.4f' % loss)

            # Save model
            if n_iter % save_model_freq == 0 and n_iter > 0:
                net.decoder_save_weights("./module/", n_iter)

            n_iter += 1