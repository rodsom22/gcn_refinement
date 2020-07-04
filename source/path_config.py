'''
Paths and configuration for module.
'''


MODEL_DIR = "./models/"
WORKING_DIR = "./models/"

WEIGHTS = MODEL_DIR + "checkpoint.pancreas.dsc.pth.tar"

ROI_LIMITS = WORKING_DIR + "limits.npy"  # The roi where the graph is constructed
ROI_PREDICTION = WORKING_DIR + "roi_prediction.npy"
ROI_VOLUME = WORKING_DIR + "roi_volume.npy"
ROI_REFERENCE = WORKING_DIR + "roi_reference.npy"
E_PATH = WORKING_DIR + "roi_expectation.npy"
ENT_PATH = WORKING_DIR + "roi_entropy.npy"
DILATION_PATH = WORKING_DIR + "graph_roi.npy"
BIN_ENTROPY = WORKING_DIR + "bin_ent.npy"

GRAPH_PATH = WORKING_DIR + "graph.npy"  # graph adjacency mat
WEIGHTS_PATH = WORKING_DIR + "graph_weights.npy"  # Edge weights
FEATURES_PATH = WORKING_DIR + "graph_node_features.npy"  # Node features
LABELS_PATH = WORKING_DIR + "graph_ground_truth.npy"  # Obtained from the CNN prediction!!!
Y_TEST_PATH = WORKING_DIR + "reference_graph.npy"  # The graph obtained from the volume ground truth
MASK_PATH = WORKING_DIR + "unc_mask.npy"  # Uncertain (unlabeled) nodes
GRAPH_PREDICTION = WORKING_DIR + "graph_prediction.npy"
CRF_PREDICTION = WORKING_DIR + "crf_prediction.npy"

NIFTI_PREDICTION = WORKING_DIR + "cnn_prediction.nii.gz"
NIFTI_E = WORKING_DIR + "cnn_expectation..nii.gz"
NIFTI_ENT = WORKING_DIR + "cnn_entropy.nii.gz"
NIFTI_BIN_ENTROPY = WORKING_DIR + "binary_cnn_entropy.nii.gz"
NIFTI_GRAPH_SEG = WORKING_DIR + "gcn_prediction.nii.gz"
NIFTI_CRF_SEG = WORKING_DIR + "crf_prediction.nii.gz"
