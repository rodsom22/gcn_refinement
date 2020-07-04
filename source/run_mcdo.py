import torch
import numpy as np
from dlm import mcunet
from utilities.misc import npy_to_nifti
import path_config as dirs
import utilities.nparrays as arrtools
import unet_config as models
import dlm.fcn_tools as tools
from loaders.tploaders import get_pancreas_generator


def mean_vol_dsc(vol, gt_vol):
    val_acc = 0.0
    num_slices = vol.shape[2]
    for i in range(num_slices):
        mat = vol[:, :, i]
        gt_slice = gt_vol[:, :, i]
        acc = tools.dice_score_np(mat, gt_slice)
        val_acc += acc

    val_acc = val_acc / float(num_slices)
    print("Validation: acc: {:.4f}".format(val_acc))


def vol_dsc(vol, gt_vol):
    eps = 1e-9
    ab = np.sum(vol * gt_vol)
    a = np.sum(vol)
    b = np.sum(gt_vol)
    dsc = (2 * ab + eps) / (a + b + eps)
    print("Volume validation: acc: {:.4f}".format(dsc))
    return dsc


# The MCDO process, to get the initial prediction and uncertainty analysis.
def mcdo_process(sample_name, model_specs, mc_samples, save_prefix=""):
    get_organ_generator = get_pancreas_generator

    input_data = get_organ_generator(sample_name, volumes_path=dirs.WORKING_DIR, references_path=dirs.WORKING_DIR)
    vol = np.load(dirs.WORKING_DIR + sample_name[0])
    reference = np.load(dirs.WORKING_DIR + sample_name[1])

    npy_to_nifti(vol, dirs.WORKING_DIR + save_prefix + "volume.nii.gz")  # Saving the input as a nifti file
    npy_to_nifti(reference, dirs.WORKING_DIR + save_prefix + "refernce.nii.gz")  # Saving the reference as a nifti file

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("found cuda device")
    print("Running on {}".format(device))
    model = model_specs.model
    model = tools.load_chkpnt(model, model_specs.chkp_dir)
    model.to(device)
    seg_vol = tools.segment_vol(model=model, logits_to_predictions=model_specs.logits_to_predictions,
                                dataflow=input_data, device=device)
    reduced_vol = arrtools.largest_connected_component3d(vol=seg_vol)
    npy_to_nifti(reduced_vol, dirs.NIFTI_PREDICTION + save_prefix)  # Saving CNN prediction as a nifti file

    roi_limits = arrtools.bounding_cube(reduced_vol)
    np.save(dirs.ROI_LIMITS, np.asarray(roi_limits))  # saving a square roi containing the organ.
    # Getting the components that are inside the ROI
    roi_prediction = reduced_vol[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]

    roi_vol = vol[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]
    roi_ref = reference[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]
    np.save(dirs.ROI_PREDICTION, roi_prediction)
    np.save(dirs.ROI_VOLUME, roi_vol)
    np.save(dirs.ROI_REFERENCE, roi_ref)
    input_data = get_organ_generator(sample_name, volumes_path=dirs.WORKING_DIR, references_path=dirs.WORKING_DIR)
    # Once we have the original prediction, we perform MCDO
    ref_shape = mcunet.montecarlo_dropout(model, input_data, device, mc_samples, roi_limits, save_prefix=save_prefix)
    return ref_shape


#  Generates a ROI based on the shape of entropy and CNN prediction
def generate_graph_roi(entropy_th, ref_shape, save_prefix=""):
    mcunet.voxel_selection(entropy_th, ref_shape, save_prefix=save_prefix)


def main(sample_name, mc_samples, dropout_rate, save_prefix=""):
    model_specs = models.ModelUnetAxial1Montecarlo(dropout_rate=dropout_rate)
    ref_shape = mcdo_process(sample_name=sample_name, model_specs=model_specs, mc_samples=mc_samples,
                             save_prefix=save_prefix)
    return ref_shape
