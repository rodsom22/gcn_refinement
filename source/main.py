import datetime
import run_mcdo as mc
import gen_graph as gg
import numpy as np
import train_gcn as gcn
import run_crf as crf
import path_config as dirs


MODEL_DIR = ""
WORKING_DIR = ""


def main(input_file):
    gcn_epochs = 200
    cf = 1
    w = 1
    th = 0.8
    mc_sample = 20
    dropout_rate = 0.3
    print("_____________________________ {} Inference and Monte Carlo evaluation".format(datetime.datetime.now()))
    ref_shape = mc.main(input_file, mc_samples=mc_sample, dropout_rate=dropout_rate, save_prefix="")
    mc.generate_graph_roi(th, ref_shape, save_prefix="")
    roi = np.load(dirs.ROI_LIMITS)
    print("_____________________________ {} Generating graph".format(datetime.datetime.now()))
    graph_info = gg.generate_components(cf, w)
    print("_____________________________ {} GCN refinement".format(datetime.datetime.now()))
    gcn_info = gcn.main(input_file, epochs=gcn_epochs)
    print("_____________________________ {} CRF refinement".format(datetime.datetime.now()))
    crf_info = crf.main(input_file, 10)
    print("--------------- Final Results --------------")
    print("Dice Sore:")
    print("CNN DSC: {}".format(gcn_info["cnn_vol_dsc"]))
    print("GCN Refinement DSC: {}".format(gcn_info["gcn_vol_dsc"]))
    print("CRF Refinement DSC: {}".format(crf_info["crf_vol_dsc"]))


if __name__ == "__main__":
    main(input_file=["vol.0012.nii.gz.npy", "ref.0012.nii.gz.npy"])
