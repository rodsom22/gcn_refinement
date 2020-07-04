import numpy as np
import cv2
import gcn.utils as gut
import gcn.weighting as wfs
import gcn.connectivity as cfs
import path_config as dirs


def generate_components(connect_funct, weight):
    vol_path = dirs.ROI_VOLUME
    ref_path = dirs.ROI_REFERENCE
    seg_path = dirs.ROI_PREDICTION
    var_path = dirs.BIN_ENTROPY
    valid_path = dirs.DILATION_PATH

    probability = np.load(dirs.E_PATH)
    entropy = np.load(dirs.ENT_PATH)
    seg_vol = np.load(seg_path)
    var_vol = np.load(var_path)
    reference = np.load(ref_path)
    vol = np.load(vol_path)
    # ----- Normalizing volume
    num_vox = float(vol.shape[0] * vol.shape[1] * vol.shape[2])
    vmu = vol.astype(np.float32).sum() / num_vox
    vvar = np.sum((vol.astype(np.float32) - vmu) ** 2) / num_vox

    fts = np.array(vol, dtype=np.float32)
    fts = (fts - vmu) / vvar
    fts = np.expand_dims(fts, 3)

    fts = gut.add_feature(probability, fts)
    fts = gut.add_feature(entropy, fts)

    valid_nodes = np.load(valid_path)
    voxel_node, node_voxel = gut.map_voxel_nodes(vol.shape, valid_nodes.astype(np.bool))
    ft_graph = gut.graph_fts(fts, node_voxel)  # convert the feature vol to a graph representation

    args = {
        "volume": (vol.astype(np.float32) - vmu) / vvar,
        "prediction": seg_vol,
        "probability": probability,
        "uncertainty": var_vol,
        "entropy_map": entropy,
        "features": fts
    }

    graph, weights, lb, N = cfs.get_connect_func(connect_funct)(ref=seg_vol, voxel_node=voxel_node,
                                                                node_voxel=node_voxel, working_nodes=valid_nodes,
                                                                k_random=16, weighting=wfs.get_weighting_func(weight),
                                                                args=args)
    mask = gut.generate_mask(var_vol, node_voxel)  # Uncertainty mask
    # Volume ground truth are represented as nodes in the graph (reference graph)
    ref_lb = gut.reference_to_graph(reference, node_voxel)

    np.save(dirs.GRAPH_PATH, graph)
    np.save(dirs.WEIGHTS_PATH, weights)
    np.save(dirs.FEATURES_PATH, ft_graph)
    np.save(dirs.LABELS_PATH, lb)  # Node labels from CNN prediction
    np.save(dirs.Y_TEST_PATH, ref_lb) # This is the real ground truth (from the ground truth volume)
    np.save(dirs.MASK_PATH, mask)  # Elements that should not be part of training (uncertain points)

    print("Final shapes")
    print("Graph shape: {}".format(graph.shape))
    print("Weight shape: {}".format(weights.shape))
    print("Ft shape: {}".format(ft_graph.shape))
    print("Train labels shape: {}".format(lb.shape))
    print("Ref labels shape: {}".format(ref_lb.shape))
    print("Mask (uncertain) shape: {}".format(mask.shape))
    print("Num nodes: {}".format(np.sum(valid_nodes)))
    print("Num of uncertain nodes: {}. Val: {}".format(np.sum(mask), np.sum(var_vol)))
    print("Num certain nodes: {}. Val: {}".format(N - np.sum(mask), np.sum(valid_nodes) - np.sum(var_vol)))
    print("Num of positive samples: {}".format(np.sum(lb[np.where(mask == 0)[0]] == 1)))
    print("Num of negative samples: {}".format(np.sum(lb[np.where(mask == 0)[0]] == 0)))

    info = {
        "N" : N,
        "total_edges": N,
        "graph_shape": graph.shape,
        "weight_shape": weights.shape,
        "ft_shape": ft_graph.shape,
        "train_labels_shape": lb.shape,
        "ref_labels_shape": ref_lb.shape,
        "mask_uncertainty_shape": mask.shape,
        "num_nodes" : np.sum(valid_nodes),
        "num_uncertainty_nodes": np.sum(mask),
        "num_certainty_nodes": N - np.sum(mask),
        "num_positive_samples": np.sum(lb[np.where(mask == 0)[0]] == 1),
        "num_negative_samples": np.sum(lb[np.where(mask == 0)[0]] == 0)
    }

    return info


def reconstruct_segmentation():
    labels_path = dirs.LABELS_PATH
    vol_path = dirs.ROI_VOLUME

    label_graph = np.load(labels_path)
    vol = np.load(vol_path)
    graph_map = gut.vol_graph_map(vol.shape)
    seg_vol = gut.reconstruct_from_n6(label_graph, graph_map)
    y, x, z = seg_vol.shape
    for i in range(z):
        slice = seg_vol[:, :, i]
        cv2.imshow("Slice", (slice*255).astype(np.uint8))
        cv2.waitKey(300)


def load_data_test():
    d = gut.load_data()
    return d
