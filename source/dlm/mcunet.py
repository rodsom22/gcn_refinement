import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage as ndimage
import path_config as dirs
import datetime
from utilities.misc import npy_to_nifti
import dlm.fcn_tools as tools


# ---------------------- Uncertainty ------------------------------------------
def expand_ft_to_size(x, y):

    expanded = np.zeros((x.shape[0], y.shape[0], y.shape[1]))
    xoff = (y.shape[1] - x.shape[2]) // 2
    yoff = (y.shape[0] - x.shape[1]) // 2
    xs = x.shape[2]
    ys = x.shape[1]
    expanded[:, yoff:yoff + ys, xoff:xoff + xs] = x
    return expanded


def assemble_fts_save(fts_arr, save_dir=None):
    z = len(fts_arr)
    d, y, x = fts_arr[0].shape
    print(fts_arr[0].shape)
    print(z)
    np_vol = np.empty(shape=(y, x, z, d), dtype=float)
    for i in range(z):
        for j in range(d):
            np_vol[:, :, i, j] = fts_arr[i][j, :, :]
    if save_dir is not None:
        np.save(save_dir, np_vol)
    return np_vol


def assemble_vol_save(vol, save_dir=None):
    z = len(vol)
    y, x = vol[0].shape
    np_vol = np.empty(shape=(y, x, z), dtype=float)
    for i in range(z):
        np_vol[:, :, i] = vol[i]

    if save_dir is not None:
        np.save(save_dir, np_vol)
    return np_vol


def voxel_selection(entropy_th, ref_shape, save_prefix=""):
    entropy_vol = np.load(dirs.ENT_PATH)
    probability_vol = np.load(dirs.E_PATH)
    roi = np.load(dirs.ROI_LIMITS)

    print("Selecting Voxels for Graph ROI...")
    print("Entropy shape: {}".format(entropy_vol.shape))
    print("Probability shape: {}".format(probability_vol.shape))
    print("Entropy th is {}".format(entropy_th))
    bin_entropy = (entropy_vol > entropy_th).astype(np.uint8)
    bin_prob = (probability_vol > 0.5).astype(np.uint8)
    kernel = np.ones(shape=(5, 7, 7), dtype=np.bool)
    dilated = ndimage.binary_dilation(bin_entropy, structure=kernel).astype(np.uint8)
    print("Input nodes: {} reduced to {}".format(probability_vol.shape[0]*probability_vol.shape[1] *
                                                 probability_vol.shape[2], np.sum(dilated)))

    dilated = ((dilated + bin_prob) > 0).astype(np.int)

    expanded_bin_entropy = np.zeros(shape=ref_shape)
    expanded_bin_entropy[roi[0]:roi[3], roi[1]:roi[4], roi[2]:roi[5]] = bin_entropy
    np.save(dirs.DILATION_PATH, dilated)
    np.save(dirs.BIN_ENTROPY, bin_entropy)
    npy_to_nifti(expanded_bin_entropy, dirs.NIFTI_BIN_ENTROPY + save_prefix)
    print("done!")


def montecarlo_dropout(model, val_dataflow, device, num_samples, roi, save_prefix=""):
    date = datetime.datetime.now()
    print("Starting Montecarlo Dropout: {}".format(date))
    print("Running for {} samples".format(num_samples))
    torch.set_grad_enabled(False)
    sigm = nn.Sigmoid()
    probability_vol = []
    entropy_vol = []
    slice_counter = 0
    for data_np, _ in val_dataflow:
        if roi is not None and not (roi[2] <= slice_counter < roi[5]):
            probability_vol.append(np.zeros(data_np[0, 0, :, :].shape, dtype=np.float))
            entropy_vol.append(np.zeros(data_np[0, 0, :, :].shape, dtype=np.float))
            slice_counter += 1
            continue

        data_tensor = torch.from_numpy(data_np).float().to(device)
        mc_outs = []
        model.train()
        for t in range(num_samples):
            logits, _ = model(data_tensor)
            probabilities_tensor = sigm(logits)

#           ------------- Visualization of results -------------------------------
            probabilities = probabilities_tensor.cpu().numpy()
            mc_outs.append(probabilities[0])

        mc_outs = np.array(mc_outs)
        mc_mat = np.sum(mc_outs, axis=0, dtype=np.float) / float(num_samples)
        probability_vol.append(tools.expand_to_size_np(mc_mat[0], data_np[0, 0, :, :]))
        entropy_mat = -mc_mat * np.log2(mc_mat + 1.0e-15) - (1.0 - mc_mat) * np.log2(1.0 - mc_mat + 1.0e-15)
        entropy_vol.append(tools.expand_to_size_np(entropy_mat[0], data_np[0, 0, :, :]))
        slice_counter += 1

    probability_vol = assemble_vol_save(probability_vol)
    entropy_vol = assemble_vol_save(entropy_vol)
    entropy_vol[entropy_vol < 0] = 0.0

    # Saving the expectation and entropy as NIFTI
    npy_to_nifti(probability_vol * 255, dirs.NIFTI_E + save_prefix)
    max_ent = np.max(entropy_vol)
    npy_to_nifti((entropy_vol * (255 // max_ent)).astype(np.uint8), dirs.NIFTI_ENT + save_prefix)

    ret_vol = (probability_vol > 0.5).astype(np.int)

    #  Getting the expectation and entropy in the ROI, for internal use.
    if roi is not None:
        probability_vol = probability_vol[roi[0]:roi[3], roi[1]:roi[4], roi[2]:roi[5]]
        entropy_vol = entropy_vol[roi[0]:roi[3], roi[1]:roi[4], roi[2]:roi[5]]

    np.save(dirs.E_PATH, probability_vol)
    np.save(dirs.ENT_PATH, entropy_vol)
    total_time = datetime.datetime.now() - date
    date = datetime.datetime.now()
    print("--------- DONE Montecarlo Dropout --------------")
    return ret_vol.shape
