import torch.optim as optim
import dlm.fcn_tools as tools
from dlm import unet
import path_config as dirs

# Settings for the Unet

class ModelUnetAxial1:
    def __init__(self):
        self.model = unet.UNet(in_channels=1, out_classes=1, padding=0)
        self.metric = tools.dice_score_tensor
        self.logits_to_predictions = tools.dsc_logits_to_predictions
        self.chkp_dir = dirs.WEIGHTS
        self.batch_size = 4


class ModelUnetAxial1Montecarlo(ModelUnetAxial1):
    def __init__(self, dropout_rate=0.1):
        super(ModelUnetAxial1Montecarlo, self).__init__()
        self.model = unet.UNet(in_channels=1, out_classes=1, padding=0, dropout_rate=dropout_rate)

