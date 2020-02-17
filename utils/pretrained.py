import os
import torch
import torchvision
from google_drive_downloader import GoogleDriveDownloader as gd

DIR = os.path.abspath(os.path.dirname(__file__))

def get_pretrained_model(model_type):
    model = eval("{}".format(model_type))
    return model().get_model()

class VGG16():
    def __init__(self, pretrained=True,
                 root=os.path.join(DIR, '../data/PreTrainedModels/vgg16_from_caffe.pth')):
        self.pretrained = pretrained
        self.root = root

    def get_model(self):
        model = torchvision.models.vgg16(pretrained=False)
        if not self.pretrained:
            return model
        self._fetch_vgg16_pretrained_model()
        state_dict = torch.load(self.root)
        model.load_state_dict(state_dict)
        return model

    def _fetch_vgg16_pretrained_model(self):
        gd.download_file_from_google_drive(
            file_id='0B9P1L--7Wd2vLTJZMXpIRkVVRFk',
            dest_path=self.root,
            unzip=False,
            showsize=True)
