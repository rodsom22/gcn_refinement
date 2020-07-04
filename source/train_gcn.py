from __future__ import division
from __future__ import print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from gcn.utils import load_data, accuracy, reconstruct_from_n6, map_voxel_nodes
from gcn.models import GCN as GCN
import path_config as dirs
import utilities.nparrays as arrtools
import dlm.fcn_tools as tools
from utilities.misc import npy_to_nifti
import global_param as mpar
import cv2


# Using the default parameters is equivalent to the cross entropy loss
def focal_loss(p, y, alpha=0.5, gamma=0.0):
    eps = 1.0e-15
    p0 = torch.ones_like(p) - p
    y0 = torch.ones_like(y) - y

    loss = -1.0*alpha
    loss *= y
    pw = (p0.pow(gamma))
    loss *= pw
    loss *= torch.log(p + eps)

    loss2 = -1.0*(1.0 - alpha)
    loss2 *= y0
    loss2 *= (p ** gamma)
    loss2 *= torch.log(p0 + eps)

    loss += loss2
    return torch.mean(loss)


def train(model, optimizer, epoch, adj, features, labels, idx_train, idx_val, valid=False):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
#    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train = focal_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if valid:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
#        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        loss_val = focal_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


def mean_vol_dsc(vol, gt_vol):
    val_acc = 0.0
    num_slices = vol.shape[2]
    for i in range(num_slices):
        mat = vol[:, :, i]
        gt_slice = gt_vol[:, :, i]
        acc = tools.dice_score_np(mat, gt_slice)
        val_acc += acc

    val_acc = val_acc / float(num_slices)
    return val_acc


def vol_dsc(vol, gt_vol):
    eps = 1e-9
    ab = np.sum(vol * gt_vol)
    a = np.sum(vol)
    b = np.sum(gt_vol)
    dsc = (2 * ab + eps) / (a + b + eps)
    return dsc


def gcn_inference(sample_name, model, adj, features, y_test, idx_test, get_probs=False):
    roi_limits = np.load(dirs.ROI_LIMITS)
    segmentation = np.load(dirs.ROI_PREDICTION)
    gt = np.load(dirs.WORKING_DIR + sample_name[1])
    gt[gt != 0] = 1

    roi_vol = np.load(dirs.ROI_VOLUME)
    vol = np.load(dirs.WORKING_DIR + sample_name[0])
    valid_nodes = np.load(dirs.DILATION_PATH)
    model.eval()
    output = model(features, adj)
#    loss_test = F.nll_loss(output[idx_test], y_test[idx_test])S
    loss_test = focal_loss(output[idx_test], y_test[idx_test])
    acc_test = accuracy(output[idx_test], y_test[idx_test])

    voxel_node, node_voxel = map_voxel_nodes(roi_vol.shape, valid_nodes.astype(np.bool))
    if get_probs:
        graph_predictions = output.cpu().detach().numpy().astype(np.float32)
        graph_predictions = reconstruct_from_n6(graph_predictions, node_voxel, roi_vol.shape, dtype=np.float)
        gp_expanded = np.zeros(vol.shape, dtype=np.float)
        gp_expanded[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]] \
            = graph_predictions
        return gp_expanded
    else:
        graph_predictions = (output > mpar.gcn_th).cpu().numpy().astype(np.float32)

    graph_predictions = reconstruct_from_n6(graph_predictions, node_voxel, roi_vol.shape)  # recovering the volume shape

    refined = graph_predictions

    # recovering sizes
    segmentation_expanded = np.zeros(vol.shape, dtype=np.float)
    segmentation_expanded[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]] \
        = segmentation

    refined_expanded = np.zeros(vol.shape, dtype=np.float)
    refined_expanded[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]] = refined

    cnn_slice_dsc = mean_vol_dsc(segmentation_expanded, gt)
    gcn_slice_dsc = mean_vol_dsc(refined_expanded, gt)

    cnn_vol_dsc = vol_dsc(segmentation_expanded, gt)
    gcn_vol_dsc = vol_dsc(refined_expanded, gt)

    np.save(dirs.GRAPH_PREDICTION, refined)
    npy_to_nifti(refined_expanded, dirs.NIFTI_GRAPH_SEG)

    info = {
        "cnn_slice_dsc": cnn_slice_dsc,
        "gcn_slice_dsc": gcn_slice_dsc,
        "cnn_vol_dsc": cnn_vol_dsc,
        "gcn_vol_dsc": gcn_vol_dsc
    }

    return info


def main(sample_name, epochs=200, get_probs=False):
    # Training settings
    valid = False
    no_cuda = False
    seed = 42
    lr = 1e-2
    weight_decay = 1e-5
    hidden = 32
    dropout = 0.5

    cuda = not no_cuda and torch.cuda.is_available()

    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # Load data
    adj, features, labels, y_test, idx_train, idx_val, idx_test = load_data()

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=1,
                dropout=dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    if cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        y_test = y_test.cuda()

#   Training model
    torch.set_grad_enabled(True)
    t_total = time.time()
    model.eval()
    print("------- Training GCN")
    for epoch in range(epochs):
        if epoch == epochs - 1:
            valid = True
        train(model, optimizer, epoch, adj, features, labels, idx_train, idx_val, valid)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    info = gcn_inference(sample_name, model, adj, features, y_test, idx_test, get_probs=get_probs)
    return info
