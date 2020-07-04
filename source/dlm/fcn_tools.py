import torch
import cv2
import numpy as np
import utilities.opencvgui as gui
import global_param as mpar


def dice_score_tensor(reference, predictions):
    """Computes the dice score for the data and prediction. Tensor shapes are ``[batch, 1, rows, cols]``. The
    dice score is computed along the batch and is averaged at the end.

    Parameters
    ----------
    reference : torch.Tensor
        The reference data, e.g. a ground truth binary image. Shape is `[batch, 1, rows, cols]`.
    predictions : torch.Tensor
        Segmentation predicted for a particular model, with the same shape as input.

    Returns
    -------
    tensor
        A single valued tensor with the average dice score across the elements of the batch.
    """
    eps = 1.
    ab = torch.sum(reference * predictions, dim=(1, 2, 3))
    a = torch.sum(reference, dim=(1, 2, 3))
    b = torch.sum(predictions, dim=(1, 2, 3))
    dsc = (2 * ab + eps) / (a + b + eps)
    dsc = torch.mean(dsc)
    return dsc


def dice_score_np(data, predictions):
    """Computes the dice score between the 2D arrays data and prediction. The shapes for both inputs are
    ``[rows, cols]``

    Parameters
    ----------
    data : numpy.ndarray
        The reference data, e.g. a ground truth binary image. Shape is `[rows, cols]`.
    predictions : numpy.ndarray
        Predicted segmentation from a particular model, with the same shape as input.

    Returns
    -------
    float
        The dice score between the inputs.
    """
    eps = 1e-9
    ab = np.sum(data * predictions)
    a = np.sum(data)
    b = np.sum(predictions)
    dsc = (2 * ab + eps) / (a + b + eps)
    return dsc


def expand_to_size_np(x, y):
    """Expand the 2D array ``x`` to the size of the 2D array ``y`` by centering x in the new 2D array and filling with
    zero the remaining space.

    Parameters
    ----------
    x : numpy.ndarray
        The source 2D array.
    y : numpy.ndarray
        The reference 2D array. The dimensions of y must be bigger or equal than the dimensions of x.

    Returns
    -------
    numpy.ndarray
        An array with the same shape of y, the information of x center-aligned and filled with zero in the remaining
        space.

    """
    expanded = np.zeros(y.shape)
    x_off = (y.shape[1] - x.shape[1]) // 2
    y_off = (y.shape[0] - x.shape[0]) // 2
    xs = x.shape[1]
    ys = x.shape[0]
    expanded[y_off:y_off + ys, x_off:x_off + xs] = x
    return expanded


def view_bin_seg_results_np(image, reference, prediction, wait_time=0, window_name="Results"):
    """Shows the image, reference and segmentation using opencv based windows.

    Parameters
    ----------
    image : numpy.ndarray
        A numpy array with the input image.
    reference : numpy.ndarray
        A numpy array with the binary ground truth.
    prediction : numpy.ndarray
        A numpy array with the predicted binary segmentation.
    wait_time : int
        The time that the image will be visible.
    window_name : str
        A name for the showing window.

    Returns
    -------
    None
    """
    segmentation = expand_to_size_np(prediction, reference)
    im_res = np.zeros((reference.shape[0], reference.shape[1], 3), dtype=np.uint8)
    im_res[segmentation == 1] = [0, 0, 255]
    im_res[reference == 1] = [255, 255, 255]
    im_res[(reference * segmentation) == 1] = [0, 255, 0]
    im_show = np.array(image)
    cv2.normalize(src=image, dst=im_show, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    results = gui.three_image_view(im_show.astype(np.uint8), reference * 255, im_res)
    cv2.imshow(window_name, results)
    cv2.waitKey(wait_time)


# region Dice Score training
def crop_tensor_to_size_reference(x1, x2):
    """A center-based crop of x1 (``shaped [batch, c, rows1, cols1]``) to the x2's (shaped ``[batch, c, rows2, cols2]``)
    size. It is assumed that rows1/cos1 >= rows2/cos2.

    Parameters
    ----------
    x1 : torch.Tensor
        The tensor that will be cropped.
    x2 : torch.Tensor
        The reference tensor. The batch size and number of channels must be the same as `x1`.

    Returns
    -------
    torch.Tensor
        A tensor with the content of x1 cropped to the rows and cols of x2.
    """
    x_off = (x1.size()[3] - x2.size()[3]) // 2
    y_off = (x1.size()[2] - x2.size()[2]) // 2
    xs = x2.size()[3]
    ys = x2.size()[2]
    x = x1[:, :, y_off:y_off + ys, x_off:x_off + xs]
    return x


def dsc_logits_to_predictions(logits):
    """Transforms a [batch, 1, rows, cols] tensor of logits to a [batch, 1, rows, cols] tensor of binary predictions.

    Parameters
    ----------
    logits : torch.Tensor
        Logits obtained by the model
    Returns
    -------
    torch.Tensor
        A tensor of predictions that can be compared with a reference.
    """
    probabilities = torch.sigmoid(logits)
    predictions = probabilities > mpar.cnn_th
    return predictions.int()


def load_chkpnt(model, chkp_name):
    print("Restoring from " + chkp_name)
    state = torch.load(chkp_name)
    model.load_state_dict(state["state_dict"])
    return model


def segment_vol(model, logits_to_predictions, dataflow, device):
    torch.set_grad_enabled(False)
    seg_vol = []
    model.eval()
    for data_np, _ in dataflow:
        data_tensor = torch.from_numpy(data_np).float().to(device)
        logits, _ = model(data_tensor)
        predictions = logits_to_predictions(logits=logits)
        segmentation = predictions.cpu().numpy()
        segmentation = segmentation[0][0].astype(np.uint8)
#        print(segmentation.shape)
        expanded_segmentation = expand_to_size_np(segmentation, data_np[0, 0, :, :])
        seg_vol.append(expanded_segmentation)

    def assemble_vol(vol):
        z = len(vol)
        y, x = vol[0].shape
        np_vol = np.empty(shape=(y, x, z), dtype=float)
        for i in range(z):
            np_vol[:, :, i] = vol[i]
        return np_vol

    ret_vol = assemble_vol(seg_vol)
    return ret_vol


def segment_vol_logits(model, dataflow, device):
    torch.set_grad_enabled(False)
    seg_vol = []
    model.eval()
    for data_np, _ in dataflow:
        data_tensor = torch.from_numpy(data_np).float().to(device)
        logits, _ = model(data_tensor)
        logits = torch.sigmoid(logits)
        segmentation = logits.cpu().numpy()
        segmentation = segmentation[0][0].astype(np.float32)
        expanded_segmentation = expand_to_size_np(segmentation, data_np[0, 0, :, :])
        seg_vol.append(expanded_segmentation)

    def assemble_vol(vol):
        z = len(vol)
        y, x = vol[0].shape
        np_vol = np.empty(shape=(y, x, z), dtype=float)
        for i in range(z):
            np_vol[:, :, i] = vol[i]
        return np_vol

    ret_vol = assemble_vol(seg_vol)
    return ret_vol