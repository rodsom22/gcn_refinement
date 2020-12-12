# Uncertainty Based GCN Refinement Strategy

This repository contains the uncertainty analysis and GCN refinement used in our work 
["Uncertainty-based Graph Convolutional Networks for Organ
Segmentation Refinement"](https://openreview.net/pdf?id=UUie86nf5B).

## Overview
Organ segmentation in CT volumes is an important pre-processing step in many computer
assisted intervention and diagnosis methods. In recent years, convolutional neural networks
have dominated the state of the art in this task. However, since this problem presents a
challenging environment due to high variability in the organ’s shape and similarity between
tissues, the generation of false negative and false positive regions in the output segmentation
is a common issue. Recent works have shown that the uncertainty analysis of the model
can provide us with useful information about potential errors in the segmentation. In this
context, we proposed a segmentation refinement method based on uncertainty analysis
and graph convolutional networks. We employ the uncertainty levels of the convolutional
network in a particular input volume to formulate a semi-supervised graph learning problem
that is solved by training a graph convolutional network.

## Usage and Requirements

The code was tested with python3 and CUDA 9.2. using the Pytorch library. 
To run the code, clone the repository

```bash
$ git clone https://github.com/rodsom22/gcn_refinement.git
```
and install the required packages:

```bash
$ cd gcn_refinement
$ pip install -r requirements.txt
```
## Data Preparation
Download the [model and samples](https://campowncloud.in.tum.de/index.php/s/v7kNZvxDMCfalhV) 
and place it into the `source/models` folder (sample from the pancreas [NIH](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) dataset). Then, run the example code

```bash
$ cd source
$ python main.py
```

The process will output a set of files in the `source/models` folder (you can change by editing the `source/path_config.py` file)
. The `.npy` files are for internal use. 
It also outputs the following Nifti files for visualization:

* `volume.nii.gz` - The input volume.
* `reference.nii.gz` - The ground truth reference.
* `cnn_prediction.nii.gz` - The initial CNN prediction.
* `cnn_expectation..nii.gz` - CNN expectation.
* `cnn_entropy.nii.gz` - Obtained CNN entropy
* `binary_cnn_entropy.nii.gz` - Binarized entropy
* `gcn_prediction.nii.gz` - GCN refinement.
* `crf_prediction.nii.gz` - CRF Refinement. 

## Citation
If any part of our work is useful for your research, please cite our corresponding publications 
([MIDL20](https://openreview.net/pdf?id=UUie86nf5B), 
[MELBA]( https://www.melba-journal.org/article/18135-an-uncertainty-driven-gcn-refinement-strategy-for-organ-segmentation)):
```
@conference{soberanismukul2020refinement,
	    Author = {Roger D. Soberanis-Mukul and Nassir Navab and Shadi Albarqouni},
	    Booktitle = {Medical Imaging with Deep Learning (MIDL)},
	    Title = {Uncertainty-based Graph Convolutional Networks for Organ Segmentation Refinement},
            Year = {2020}
}

@article{soberanismukulmelba20,
	author = {Roger D. Soberanis-Mukul and Nassir Navab and Shadi Albarqouni},	
	journal = {Machine Learning for Biomedical Imaging (MELBA)},
	note = {MIDL 2020 Special Issue},
	title = {An Uncertainty-Driven GCN Refinement Strategy for Organ Segmentation},
	year = {2020}}
}
```

## Acknowledgement
Our GCN is based on the GCN implementation by [@tkipf](https://github.com/tkipf/pygcn).

We use the CRF implementation of [@lucasb-eyer](https://github.com/lucasb-eyer/pydensecrf).
Please refer to this repository to install pydensecrf.