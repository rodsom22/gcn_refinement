# pylint: skip-file
import csv
import cv2 as cv
import numpy as np
import nibabel as nb
# import utilities.files as ftools


def nifti_to_numpy(filename):
    pass


def npy_norm_to_nifti(npy_vol, filename):
    byte_vol = np.round((255*npy_vol)).astype(np.uint8)
    nifti = nb.Nifti1Image(byte_vol, None)
    nb.save(nifti, filename)


def npy_to_nifti(npy_vol, filename):
    byte_vol = np.round(npy_vol).astype(np.uint8)
    nifti = nb.Nifti1Image(byte_vol, None)
    nb.save(nifti, filename)
