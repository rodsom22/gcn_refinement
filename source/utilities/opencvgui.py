# pylint: skip-file
"""
Utilities for displaying and interacting with images using opencv
"""
import cv2 as cv
import numpy as np

def two_image_view(img1, img2):
    """
    Combines img1 and img2 into a single image.
    Parameters
    ----------
    img1, img2: cv.mat
        Images to be combined. uint8 single or rgb channel.
    Returns
    -------
    cv.mat
        The combined image in format [img1 | img2].
    """
    if len(img1.shape) < 3:
        img1 = cv.cvtColor(img1 ,cv.COLOR_GRAY2BGR)
    if len(img2.shape) < 3:
        img2 = cv.cvtColor(img2 ,cv.COLOR_GRAY2BGR)
    r1 = img1.shape[0]
    c1 = img1.shape[1]
    r2 = img2.shape[0]
    c2 = img2.shape[1]
    fr = r1
    fc = c1 + c2 + 3
    if r1 < r2:
        fr = r2    
    view = np.zeros((fr, fc, 3), dtype = np.uint8)    
    view[0:r1, 0:c1, :] = img1
    view[0:r2, c1 + 3:fc, :] = img2
    return view
    

def three_image_view(img1, img2, img3):
    """
    Combines img1, img2, and img3 into a single image.
    Parameters
    ----------
    img1, img2, img3: cv.mat
        Images to be combined. uint8 single or rgb channel.
    Returns
    -------
    cv.mat
        The combined image in format [img1 | img2 | img3].
    """
    if len(img1.shape) < 3:
        img1 = cv.cvtColor(img1 ,cv.COLOR_GRAY2BGR)
    if len(img2.shape) < 3:
        img2 = cv.cvtColor(img2 ,cv.COLOR_GRAY2BGR)
    if len(img3.shape) < 3:
        img3 = cv.cvtColor(img3, cv.COLOR_GRAY2BGR)
    r1 = img1.shape[0]
    c1 = img1.shape[1]
    r2 = img2.shape[0]
    c2 = img2.shape[1]
    r3 = img3.shape[0]
    c3 = img3.shape[1]
    fr = r1
    fc = c1 + c2 + c3 + 6
    if r1 < r2:
        fr = r2
    if r2 < r3:
        fr = r3

    view = np.zeros((fr, fc, 3), dtype=np.uint8)
    view[0:r1, 0:c1, :] = img1
    view[0:r2, c1+3:c1+3+c2, :] = img2
    view[0:r3, c1+6+c2:fc, :] = img3
    return view


def overlap_images(base, layer, color=[255, 255, 255], transparent=False, layer_front=False):
    imres_u = np.zeros((base.shape[0], base.shape[1], 3), dtype=np.uint8)
    cv.cvtColor(src=base.astype(np.uint8), code=cv.COLOR_GRAY2BGR, dst=imres_u)

    union = base*layer
    if not transparent:
        imres_u[union != 0] = color
    else:
        if not layer_front:
            imres_u[union != 0, 0] = np.round(base[union != 0] / 255 * color[0])
            imres_u[union != 0, 1] = np.round(base[union != 0] / 255 * color[1])
            imres_u[union != 0, 2] = np.round(base[union != 0] / 255 * color[2])
        else:
            imres_u[union != 0, 0] = np.round(layer[union != 0] * color[0])
            imres_u[union != 0, 1] = np.round(layer[union != 0] * color[1])
            imres_u[union != 0, 2] = np.round(layer[union != 0] * color[2])

    return imres_u