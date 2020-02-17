import torch
from torch import nn
from semseg.models.models import get_model
from semseg.utils.initialization import weights_init
from utils.pretrained import get_pretrained_model

MODELS = ["FCN"]

def init_specific_model(model_type,
                        img_size,
                        num_classes,
                        init_from_pretrained=False,
                        pretrained_type=None):
    """Return an instance of a segmentation encoder from `model_type`."""
    # model_type = model_type.lower().capitalize()
    if model_type not in MODELS:
        err = "Unknown model_type={}. Possible values: {}"
        raise ValueError(err.format(model_type, MODELS))

    specific_model = get_model(model_type)

    if init_from_pretrained:
        pretrained_model = get_pretrained_model(pretrained_type)
    else:
        pretrained_model = None

    model = SegNet(specific_model,
                   img_size=img_size[0],
                   num_classes=num_classes,
                   init_from_pretrained=init_from_pretrained,
                   pretrained_model=pretrained_model)
    model.model_type = model_type  # store to help reloading
    return model

class SegNet(nn.Module):
    def __init__(self, encoder,
                 img_size,
                 num_classes,
                 init_from_pretrained=False,
                 pretrained_model=None):
        """
        :param encoder:
        :param img_size:
        :param num_classes:
        """
        super(SegNet, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.encoder = encoder(img_size, num_classes)
        self.encoder.initialize_weights(from_pretrained=init_from_pretrained,
                                        model=pretrained_model)

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        return self.encoder(x)

    def reset_parameters(self):
        """ Initializes the weights for each layer of the CNN"""
        self.apply(weights_init)
