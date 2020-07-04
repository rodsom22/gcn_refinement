import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_gaussian, create_pairwise_bilateral
import dlm.fcn_tools as tools

import path_config as dirs
from utilities.misc import npy_to_nifti


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


def get_unary2d(probabilities, num_class):
    y, x = probabilities.shape
    U = np.ndarray(shape=[num_class, y, x], dtype=np.float)
    U[0, :, :] = 1 - probabilities
    U[1, :, :] = probabilities
    return -np.log(U.reshape((num_class, -1)))


def crf_refine(name, it=5):
    vol = np.load(dirs.WORKING_DIR + name[0])
    vol_crop = np.load(dirs.ROI_VOLUME)
    segmentation = np.load(dirs.ROI_PREDICTION)  # CNN Prediction
    gt = np.load(dirs.WORKING_DIR + name[1])
    gt[gt != 0] = 1

    roi_limits = np.load(dirs.ROI_LIMITS)
    cnn_prediction = np.load(dirs.ROI_PREDICTION)

    y, x, z = vol_crop.shape

    U = np.ndarray(shape=[2, y, x, z], dtype=np.float32)
    U[0, :] = 1 - cnn_prediction
    U[1, :] = cnn_prediction

    d = dcrf.DenseCRF(x*y*z, 2)  # npoints, nlabels

    U = U.reshape((2, -1))
    d.setUnaryEnergy(-np.log(U + 1.0e-15))

    pairwise_gauss = create_pairwise_gaussian((y, x, z), (y, x, z))
    d.addPairwiseEnergy(pairwise_gauss, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    pairwise_bilat = create_pairwise_bilateral(sdims=(10, 10, 10), schan=(1,), img=vol_crop)
    d.addPairwiseEnergy(pairwise_bilat, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(it)
    map_res = np.argmax(Q, axis=0)

    graph_predictions = map_res.reshape((y, x, z))
    refined = graph_predictions

    # recovering sizes
    segmentation_expanded = np.zeros(vol.shape, dtype=np.float)
    segmentation_expanded[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]] \
        = segmentation

    refined_expanded = np.zeros(vol.shape, dtype=np.float)
    refined_expanded[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]] = refined

    cnn_slice_dsc = mean_vol_dsc(segmentation_expanded, gt)
    crf_slice_dsc = mean_vol_dsc(refined_expanded, gt)

    cnn_vol_dsc = vol_dsc(segmentation_expanded, gt)
    crf_vol_dsc = vol_dsc(refined_expanded, gt)

    np.save(dirs.CRF_PREDICTION, refined)
    npy_to_nifti(refined_expanded, dirs.NIFTI_CRF_SEG)

    info = {
        "cnn_slice_dsc": cnn_slice_dsc,
        "crf_slice_dsc": crf_slice_dsc,
        "cnn_vol_dsc": cnn_vol_dsc,
        "crf_vol_dsc": crf_vol_dsc
    }

    return info


def main(name, it=10):
    info = crf_refine(name, it)
    return info
