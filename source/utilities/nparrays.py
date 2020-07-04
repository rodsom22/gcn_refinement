import numpy as np
from scipy.ndimage import label


def extend1_before(d):
    return np.expand_dims(d, axis=0)


def extend2_before(d):
    return np.expand_dims(extend1_before(d), axis=0)


def largest_connected_component3d(vol):
    ret_vol = np.copy(vol)
    s = np.ones(shape=(3, 3, 3))
    labels, num_ft = label(vol, structure=s)
    max_connected = 0
    max_label = 0
    for i in range(1, num_ft + 1):
        num = np.sum(labels == i)
        if num > max_connected:
            max_connected = num
            max_label = i

    ret_vol[labels != max_label] = 0
    return ret_vol


def bounding_cube(vol, offset=0):
    a = np.where(vol != 0)
    bbox = np.min(a[0]), np.min(a[1]), np.min(a[2]) - offset, \
        np.max(a[0]) + 1, np.max(a[1]) + 1, np.max(a[2]) + 1 + offset
    return bbox


def expand_to_size_3d(x, shape):
    expanded = np.zeros(shape)
    xoff = (shape[1] - x.shape[1]) // 2
    yoff = (shape[0] - x.shape[0]) // 2
    zoff = (shape[2] - x.shape[2]) // 2
    xs = x.shape[1]
    ys = x.shape[0]
    zs = x.shape[2]
    expanded[yoff:yoff + ys, xoff:xoff + xs, zoff:zoff + zs] = x
    return expanded
