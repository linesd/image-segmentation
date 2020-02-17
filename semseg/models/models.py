import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from utils.helpers import crop_tensor

# ALL encoders should be called Encoder<Model>
def get_model(model_type):
    # model_type = model_type#.lower().capitalize()
    return eval("{}".format(model_type))

def upsample_weights_init(in_channel, out_channel, filter_size):
    """
    For FCN
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        centre = factor - 1
    else:
        centre = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    filter =  (1 - abs(og[0] - centre) / factor) * (1 - abs(og[1] - centre) / factor)
    weights = np.zeros((in_channel, out_channel, filter_size, filter_size))
    weights[range(in_channel), range(out_channel),:,:] = filter
    return torch.from_numpy(weights).float()


class FCN(nn.Module):
    def __init__(self, n_chan, n_classes, version='32'):
        """
        """
        super(FCN, self).__init__()
        self.version = version # '32', '16' or '8'
        self.n_chan = n_chan
        self.n_classes = n_classes

        # conv1
        self.conv1_1 = nn.Conv2d(self.n_chan, 64, kernel_size=(3,3), padding=100)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True) # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True) # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3,3), padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3,3), padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3,3), padding=1)
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True) # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3,3), padding=1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True) # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1)
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True) # 1/32

        # fullyconv
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=(7, 7), padding=0)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout2d(p=0.5)

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=(1, 1), padding=0)
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout2d(p=0.5)
        self.score_fr = nn.Conv2d(4096, self.n_classes, kernel_size=(1, 1), padding=0)

        if self.version == '32':
            self.upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(64,64),
                                           stride=32, bias=False)

        if self.version == '16':
            self.upscore2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(4,4),
                                               stride=2, bias=False)
            self.score_pool4 = nn.Conv2d(512, self.n_classes, kernel_size=(1,1), padding=0)
            # need to sum a cropped version of upscore_pool4 and upscore2 - goes into upscore_pool4
            self.upscore16 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(32,32),
                                                    stride=16, bias=False)

        if self.version == '8':
            self.upscore2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(4, 4),
                                               stride=2, bias=False)

            self.score_pool4 = nn.Conv2d(512, self.n_classes, kernel_size=(1, 1), padding=0)
            self.upscore_pool4 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(4, 4),
                                                    stride=2, bias=False)

            self.score_pool3 = nn.Conv2d(256, self.n_classes, kernel_size=(1,1), padding=0)
            # need to sum a cropped version of score_pool3 and upscore_pool4 - goes into upscore8
            self.upscore8 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(16,16),
                                                    stride=8, bias=False)

    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        output_size = (x.shape[2], x.shape[3])

        a = self.relu1_1(self.conv1_1(x))
        a = self.relu1_2(self.conv1_2(a))
        a = self.pool1(a)

        a = self.relu2_1(self.conv2_1(a))
        a = self.relu2_2(self.conv2_2(a))
        a = self.pool2(a)

        a = self.relu3_1(self.conv3_1(a))
        a = self.relu3_2(self.conv3_2(a))
        a = self.relu3_3(self.conv3_3(a))
        a = self.pool3(a)
        pool_3 = a # save for skip (1/8)

        a = self.relu4_1(self.conv4_1(a))
        a = self.relu4_2(self.conv4_2(a))
        a = self.relu4_3(self.conv4_3(a))
        a = self.pool4(a)
        pool_4 = a # save for skip (1/16)

        a = self.relu5_1(self.conv5_1(a))
        a = self.relu5_2(self.conv5_2(a))
        a = self.relu5_3(self.conv5_3(a))
        a = self.pool5(a)

        # fully connected
        a = self.relu6(self.fc6(a))
        a = self.drop6(a)
        a = self.relu7(self.fc7(a))
        a = self.drop7(a)

        a = self.score_fr(a)

        if self.version == '32':
            upscore = self.upscore(a)
            output = interpolate(upscore, size=output_size, mode='bilinear', align_corners=True)

        elif self.version == '16':
            upscore2 = self.upscore2(a)
            score_pool4 = self.score_pool4(pool_4)
            score_pool4c = crop_tensor(score_pool4, upscore2)
            fuse_pool4 = upscore2 + score_pool4c
            upscore16 = self.upscore16(fuse_pool4)
            output = interpolate(upscore16, size=output_size, mode='bilinear', align_corners=True)

        elif self.version == '8':
            upscore2 = self.upscore2(a)
            score_pool4 = self.score_pool4(pool_4)
            score_pool4c = crop_tensor(score_pool4, upscore2)
            fuse_pool4 = upscore2 + score_pool4c
            upscore_pool4 = self.upscore_pool4(fuse_pool4)

            score_pool3 = self.score_pool3(pool_3)
            score_pool3c = crop_tensor(score_pool3, upscore_pool4)
            fuse_pool3 = upscore_pool4 + score_pool3c
            upscore8 = self.upscore8(fuse_pool3)
            output = interpolate(upscore8, size=output_size, mode='bilinear', align_corners=True)
        else:
            raise Exception('Selected version: {} not implemented, choose either "32", "16" or "8"'.format(self.version))
        return output

    def _initialize_upsample_weights(self, from_pretrained=False):
        for m in self.modules():
            if not from_pretrained:
                if isinstance(m, nn.Conv2d):
                    m.weight.data.zero_()
                    if m.bias is not None:
                        m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert (m.kernel_size[0] == m.kernel_size[1]), "Kernel must be square"
                weights = upsample_weights_init(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data = weights

    def _initialize_from_pretrained(self, model):
        self_features = list(self.modules())
        split = len(model.features) + 1
        for f1, f2 in zip(model.features, self_features[1:split]): # convolutional layers
            if isinstance(f1, nn.Conv2d) and isinstance(f2, nn.Conv2d):
                assert (f1.weight.shape == f2.weight.shape), "Conv layer weights have different shape"
                assert (f1.bias.shape == f2.bias.shape), "Conv layer biases have different shape"
                f2.weight.data = f1.weight.data
                f2.bias.data = f1.bias.data
        for fc1, fc2, in zip(model.classifier, self_features[split:split+4]): # fully connected layers
            if isinstance(fc1, nn.Linear):
                assert (len(fc1.weight) == len(fc2.weight)), "Linear layer weights have different shape"
                assert (fc1.bias.shape == fc2.bias.shape), "Linear layer biases have different shape"
                fc2.weight.data = fc1.weight.data.view(fc2.weight.shape)
                fc2.bias.data = fc1.bias.data

    def initialize_weights(self, from_pretrained=False, model=None):
        if from_pretrained:
            if model is None:
                raise Exception("A pretrained model must be included")
            self._initialize_from_pretrained(model)
        self._initialize_upsample_weights(from_pretrained=from_pretrained)